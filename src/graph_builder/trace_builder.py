"""Extract frozen VLM states and construct the heterogeneous TRACE graph.

Extraction uses teacher forcing over an image, prompt, and response.
Hooks retain the sampled-layer tensors used for graph construction.
"""
from __future__ import annotations

import json
import logging
import math
import tempfile
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from torch import Tensor
from torch_geometric.data import HeteroData
from tqdm import tqdm

from .edge_builders import build_context_edges, build_cross_edges, build_evolve_edges

logger = logging.getLogger(__name__)


def uniform_layers(total: int, count: int = 8) -> list[int]:
    """Sample equal-depth bin endpoints (zero based), including the final layer.

    For 32 layers this gives [3, 7, 11, 15, 19, 23, 27, 31]. For depths not
    divisible by K, floor(k * total / K) - 1 specifies the rounding convention.
    """
    if count < 1 or total < count:
        raise ValueError("Require 1 <= sampled layer count <= backbone depth")
    return [(k * total) // count - 1 for k in range(1, count + 1)]


def topk_visual_attention(attention: Tensor, top_k: int = 5) -> Tensor:
    """Reduce head-averaged [response, visual-token] attention (Eq. 4)."""
    if attention.ndim != 2 or attention.size(1) == 0 or top_k < 1:
        raise ValueError("Expected nonempty [response, visual-token] attention")
    return attention.float().topk(min(top_k, attention.size(1)), dim=-1).values.mean(-1)


class TRACEGraphBuilder:
    """Frozen VLM adapter with paper defaults K=8, tau=.05 and visual top-k=5.

    ``model_path`` may point to local model weights or a Hugging Face repository.
    Quantization is available through the 4-bit and 8-bit loading options.
    """

    MODEL_CONFIGS = {
        "llava-ov-8b": {
            "model_name": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "num_layers": 28, "model_class": "LlavaOnevisionForConditionalGeneration",
        },
        "llava-1.5-7b": {
            "model_name": "llava-hf/llava-1.5-7b-hf",
            "num_layers": 32, "model_class": "LlavaForConditionalGeneration",
        },
        "llava-ov-1.5-8b": {
            "model_name": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
            "num_layers": 36, "model_class": "AutoModel",
        },
        "qwen3-vl-8b": {
            "model_name": "Qwen/Qwen3-VL-8B-Instruct",
            "num_layers": 36, "model_class": "Qwen3VLForConditionalGeneration",
        },
        "qwen2.5-vl-7b": {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "num_layers": 28, "model_class": "Qwen2_5_VLForConditionalGeneration",
        },
        "qwen1-vl-7b": {
            "model_name": "Qwen/Qwen-VL-Chat",
            "num_layers": 32, "model_class": "AutoModelForCausalLM_QwenVL",
        },
    }

    def __init__(
        self,
        model_type: str = "llava-1.5-7b",
        selected_layers: Optional[list[int]] = None,
        attn_threshold: float = 0.05,
        device: str = "cuda:0",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_sampled_layers: int = 8,
        top_k: int = 5,
    ):
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model_type: {model_type}")
        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose at most one quantization mode")
        if not math.isfinite(attn_threshold) or not 0 <= attn_threshold <= 1:
            raise ValueError("attn_threshold must be in [0, 1]")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.model_type = model_type
        self.cfg = dict(self.MODEL_CONFIGS[model_type])
        self.model_path = model_path or self.cfg["model_name"]
        self.cache_dir = cache_dir
        self.attn_threshold = attn_threshold
        self.top_k = top_k
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.num_sampled_layers = num_sampled_layers
        self._explicit_layers = selected_layers is not None
        self.selected_layers = (
            list(selected_layers) if selected_layers is not None
            else uniform_layers(self.cfg["num_layers"], num_sampled_layers)
        )
        self._validate_layers(self.cfg["num_layers"])
        self.model = None
        self.processor = None
        self._hooks = []
        self._captured = {}
        self._temporary_images = []
        self._capture_response_indices = None
        self._capture_image_indices = None
        self._response_offsets = None

    def _validate_layers(self, depth: int) -> None:
        layers = self.selected_layers
        if not layers or layers != sorted(set(layers)) or layers[0] < 0 or layers[-1] >= depth:
            raise ValueError(f"selected_layers must be unique ascending indices in [0, {depth})")

    def load_model(self) -> None:
        """Load a frozen backbone with eager attention."""
        import transformers

        quant_config = None
        if self.load_in_4bit or self.load_in_8bit:
            options = dict(load_in_4bit=self.load_in_4bit, load_in_8bit=self.load_in_8bit)
            if self.load_in_4bit:
                options["bnb_4bit_compute_dtype"] = torch.float16
            if self.model_type == "qwen1-vl-7b":
                options["llm_int8_skip_modules"] = ["visual"]
            quant_config = transformers.BitsAndBytesConfig(**options)
        remote_code = self.cfg["model_class"] in {"AutoModel", "AutoModelForCausalLM_QwenVL"}
        processor_class = transformers.AutoTokenizer if self.model_type == "qwen1-vl-7b" else transformers.AutoProcessor
        self.processor = processor_class.from_pretrained(
            self.model_path, cache_dir=self.cache_dir, trust_remote_code=remote_code,
        )
        class_name = self.cfg["model_class"]
        if class_name == "AutoModelForCausalLM_QwenVL":
            class_name = "AutoModelForCausalLM"
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            raise RuntimeError(f"Installed transformers does not provide {class_name}; use a compatible backbone environment")
        self.model = model_class.from_pretrained(
            self.model_path, cache_dir=self.cache_dir, trust_remote_code=remote_code,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device_map="auto" if quant_config else self.device,
            quantization_config=quant_config, attn_implementation="eager",
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = str(next(self.model.parameters()).device)
        depth = len(self._get_transformer_layers())
        if not self._explicit_layers:
            self.selected_layers = uniform_layers(depth, self.num_sampled_layers)
        self._validate_layers(depth)
        logger.info("Loaded %s; sampled layers %s", self.model_type, self.selected_layers)

    def _get_transformer_layers(self):
        # Locate transformer blocks across supported model wrappers.
        for path in (
            "language_model.model.layers", "model.language_model.layers",
            "model.language_model.model.layers", "model.layers",
            "language_model.layers", "layers", "transformer.h",
        ):
            value = self.model
            for name in path.split("."):
                value = getattr(value, name, None)
            if value is not None and len(value):
                return value
        raise RuntimeError(f"Cannot locate transformer layers in {type(self.model).__name__}")

    def _find_vision_range(self, input_ids: Tensor) -> tuple[int, int]:
        """Locate the contiguous image-token block using processor or model IDs."""
        proc = self.processor
        tokenizer = getattr(proc, "tokenizer", proc)
        if hasattr(tokenizer, "img_start_id") and hasattr(tokenizer, "img_end_id"):
            start = (input_ids[0] == tokenizer.img_start_id).nonzero(as_tuple=True)[0]
            end = (input_ids[0] == tokenizer.img_end_id).nonzero(as_tuple=True)[0]
            if start.numel() == 1 and end.numel() == 1 and end[0] > start[0] + 1:
                return int(start[0]) + 1, int(end[0])
        config = getattr(self.model, "config", None)
        for obj in (proc, tokenizer, config):
            for name in ("image_token_id", "image_token_index"):
                token_id = getattr(obj, name, None)
                if token_id is None:
                    continue
                pos = (input_ids[0] == token_id).nonzero(as_tuple=True)[0]
                if pos.numel():
                    if not torch.equal(pos, torch.arange(pos[0], pos[-1] + 1, device=pos.device)):
                        raise ValueError("Expected one contiguous image-token block")
                    return int(pos[0]), int(pos[-1]) + 1
        raise ValueError("Cannot locate image tokens; this processor/version needs an explicit adapter")

    def _prepare_tf_inputs(self, image: Union[str, Image.Image], prompt: str, response: str):
        if self.model_type == "qwen1-vl-7b":
            return self._tf_qwenvl(image, prompt, response)
        return self._tf_standard(image, prompt, response)

    @staticmethod
    def _response_start(prompt_ids: Tensor, full_ids: Tensor) -> int:
        n = prompt_ids.size(1)
        if full_ids.size(1) <= n or not torch.equal(prompt_ids[0], full_ids[0, :n]):
            raise ValueError("Response does not preserve prompt token prefix; verify the processor/chat template")
        return n

    def _aligned_response_offsets(self, context: str, response: str, expected_ids: Tensor):
        """Obtain context-aware offsets only after verifying actual response IDs.

        Image-token expansion may change the prompt length in tokens, so we align
        the response suffix by token identity, not by the unexpanded prefix length.
        Slow tokenizers without offsets fall back to the strict HalLoc adapter.
        """
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        try:
            encoded = tokenizer(context + response, return_offsets_mapping=True)
        except (NotImplementedError, TypeError, ValueError):
            return None
        if "offset_mapping" not in encoded:
            return None
        boundary = len(context)
        selected = [(token_id, start, end) for token_id, (start, end) in
                    zip(encoded["input_ids"], encoded["offset_mapping"]) if end > boundary]
        ids = torch.tensor([entry[0] for entry in selected], dtype=torch.long)
        if not torch.equal(ids, expected_ids.detach().cpu().long()):
            return None
        offsets = [(max(0, start - boundary), end - boundary) for _, start, end in selected]
        if any(not 0 <= start < end <= len(response) for start, end in offsets):
            return None
        return torch.tensor(offsets, dtype=torch.long)

    def _tf_standard(self, image: Union[str, Image.Image], prompt: str, response: str):
        proc = self.processor
        if isinstance(image, str):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        prompt_text = proc.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        prompt_inputs = proc(images=image, text=prompt_text, return_tensors="pt")
        full_inputs = proc(images=image, text=prompt_text + response, return_tensors="pt")
        response_start = self._response_start(prompt_inputs["input_ids"], full_inputs["input_ids"])
        self._response_offsets = self._aligned_response_offsets(prompt_text, response, full_inputs["input_ids"][0, response_start:])
        full_inputs = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in full_inputs.items()}
        return full_inputs, response_start

    def _tf_qwenvl(self, image: Union[str, Image.Image], prompt: str, response: str):
        proc = self.processor
        if isinstance(image, str):
            image_path = image
        else:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary:
                image_path = temporary.name
            image.convert("RGB").save(image_path, "JPEG")
            # Qwen's model opens the image during forward, so cleanup occurs later.
            self._temporary_images.append(image_path)
        query = proc.from_list_format([{"image": image_path}, {"text": prompt}])
        prompt_ids = proc(query, return_tensors="pt")["input_ids"]
        full_inputs = proc(query + response, return_tensors="pt")
        response_start = self._response_start(prompt_ids, full_inputs["input_ids"])
        self._response_offsets = self._aligned_response_offsets(query, response, full_inputs["input_ids"][0, response_start:])
        dtype = next(self.model.parameters()).dtype
        full_inputs = {
            k: (v.to(self.device, dtype=dtype) if v.dtype == torch.uint8 or v.is_floating_point() else v.to(self.device))
            if isinstance(v, Tensor) else v for k, v in full_inputs.items()
        }
        return full_inputs, response_start

    def _register_hooks(self) -> None:
        self._clear_hooks()
        self._captured = {}
        layers = self._get_transformer_layers()
        self._validate_layers(len(layers))
        for layer_idx in self.selected_layers:
            layer = layers[layer_idx]

            def hidden_hook(module, inputs, output, idx=layer_idx):
                hs = (output[0] if isinstance(output, tuple) else output)[0].detach()
                response = self._capture_response_indices.to(hs.device)
                visual = self._capture_image_indices.to(hs.device)
                if hs.size(0) != self._input_length:
                    raise ValueError("Backbone expands image placeholders internally; use a processor that expands them before forward")
                cap = self._captured.setdefault(idx, {})
                cap["h_text"] = hs.index_select(0, response).float().cpu()
                cap["h_vision"] = hs.index_select(0, visual).float().mean(0).cpu()

            self._hooks.append(layer.register_forward_hook(hidden_hook))
            attn_module = next((getattr(layer, name) for name in ("self_attn", "attn", "attention") if hasattr(layer, name)), None)
            if attn_module is None:
                raise RuntimeError(f"Cannot locate attention module for layer {layer_idx}")

            def attention_hook(module, inputs, output, idx=layer_idx):
                # Qwen returns attention after its cache; other adapters generally
                # expose it at index 1. Select the square attention tensor.
                values = output if isinstance(output, tuple) else (output,)
                weights = next((v for v in reversed(values) if isinstance(v, Tensor)
                                and v.ndim == 4 and v.size(-2) == self._input_length
                                and v.size(-1) == self._input_length), None)
                if weights is None:
                    return
                weights = weights[0].detach()
                response = self._capture_response_indices.to(weights.device)
                visual = self._capture_image_indices.to(weights.device)
                rows = weights.index_select(1, response).float().mean(0)
                cap = self._captured.setdefault(idx, {})
                cap["attn_t2t"] = rows.index_select(1, response).cpu()
                cap["attn_t2v"] = topk_visual_attention(rows.index_select(1, visual), self.top_k).cpu()

            self._hooks.append(attn_module.register_forward_hook(attention_hook))

    def _clear_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def build_graph(self, image: Union[str, Image.Image], prompt: str, response: str, label: int) -> HeteroData:
        """Extract one graph, raising on missing states/attention or invalid alignment."""
        if self.model is None:
            raise RuntimeError("Call load_model() first")
        if label not in (0, 1):
            raise ValueError("Sentence label must be 0 (factual) or 1 (hallucination)")
        try:
            return self._build(image, prompt, response, label)
        finally:
            self._clear_hooks()
            self._captured.clear()
            for temporary in self._temporary_images:
                Path(temporary).unlink(missing_ok=True)
            self._temporary_images.clear()

    def _build(self, image, prompt, response, label):
        self._response_offsets = None
        inputs, response_start = self._prepare_tf_inputs(image, prompt, response)
        input_ids = inputs["input_ids"]
        vis_start, vis_end = self._find_vision_range(input_ids)
        if vis_end > response_start:
            raise ValueError("Image and response token ranges overlap")
        self._input_length = input_ids.size(1)
        self._capture_response_indices = torch.arange(response_start, self._input_length)
        self._capture_image_indices = torch.arange(vis_start, vis_end)
        self._register_hooks()
        # Request eager attention weights for the graph extraction hooks.
        with torch.no_grad():
            self.model(**inputs, output_attentions=True, use_cache=False)
        required = {"h_text", "h_vision", "attn_t2t", "attn_t2v"}
        for idx in self.selected_layers:
            missing = required - self._captured.get(idx, {}).keys()
            if missing:
                raise RuntimeError(f"Layer {idx} did not expose {sorted(missing)}; verify eager-attention compatibility")
        caps = [self._captured[idx] for idx in self.selected_layers]
        h_text = [cap["h_text"] for cap in caps]
        graph = HeteroData()
        graph["text"].x = torch.cat(h_text)
        graph["vision"].x = torch.stack([cap["h_vision"] for cap in caps])
        mask = torch.zeros(graph["text"].x.size(0), dtype=torch.bool)
        mask[-h_text[0].size(0):] = True
        graph["text"].last_layer_mask = mask
        edges = {
            ("text", "evolve", "text"): build_evolve_edges(h_text),
            ("text", "context", "text"): build_context_edges([cap["attn_t2t"] for cap in caps], self.attn_threshold),
            ("vision", "cross", "text"): build_cross_edges([cap["attn_t2v"] for cap in caps]),
        }
        for relation, (index, attr) in edges.items():
            graph[relation].edge_index = index
            graph[relation].edge_attr = attr
        graph.y = torch.tensor([label], dtype=torch.float)
        graph.response_token_ids = input_ids[0, response_start:].detach().cpu()
        if self._response_offsets is not None:
            graph.response_offsets = self._response_offsets
        graph.sampled_layers = torch.tensor(self.selected_layers)
        graph.attn_threshold = self.attn_threshold
        graph.top_k = self.top_k
        return graph

    def process_dataset(self, annotation_file: str, output_dir: str, image_dir: str = "", start_idx: int = 0, end_idx: Optional[int] = None) -> None:
        """Convert normalized JSONL to graph_*.pt files.

        Existing output files are rejected to prevent mixing stale graph settings.
        Use a new output directory or a disjoint [start_idx, end_idx) shard.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(annotation_file, encoding="utf-8") as stream:
            samples = [json.loads(line) for line in stream if line.strip()]
        end = len(samples) if end_idx is None else end_idx
        if not 0 <= start_idx <= end <= len(samples):
            raise ValueError("Invalid extraction range")
        destinations = [out / f"graph_{i:06d}.pt" for i in range(start_idx, end)]
        if any(path.exists() for path in destinations):
            raise FileExistsError("Output graphs already exist; choose a fresh directory or disjoint shard")
        for idx in tqdm(range(start_idx, end), desc="Building TRACE graphs"):
            sample = samples[idx]
            image = Path(sample.get("image_path", sample.get("image", "")))
            if not image.is_absolute():
                image = Path(image_dir) / image
            if not image.is_file():
                raise FileNotFoundError(image)
            graph = self.build_graph(str(image), sample["prompt"], sample["response"], sample["label"])
            graph.sample_id = str(sample.get("id", idx))
            torch.save(graph, out / f"graph_{idx:06d}.pt")
        logger.info("Saved %d graphs to %s", end - start_idx, out)
