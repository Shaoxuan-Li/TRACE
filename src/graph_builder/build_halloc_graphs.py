"""Extract full-response HalLoc graphs with exact character-to-token alignment.

A split contains unified/ (all samples), none/ (factual samples), and category/
(positive samples of that category). The unified graph carries the union of
all hallucination spans. Labels never select or crop the response text.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_builder.run_extraction import add_model_arguments, create_builder

HAL_TYPES = ("object", "attribute", "relationship", "scene")


def parse_char_spans(annotations: dict, hal_type: str) -> list[tuple[int, int]]:
    """Collect character spans for the requested hallucination category.

    ``char_index`` is a half-open ``start:end`` interval. Relationship spans
    include predicate, subject and object; attribute spans include its object.
    """
    spans = []
    for category in HAL_TYPES if hal_type == "unified" else (hal_type,):
        for annotation in annotations.get(category, []):
            for key in ("obj", "attribute", "predicate", "subject", "object", "scene"):
                if key in annotation and annotation[key].get("char_index"):
                    start, end = annotation[key]["char_index"].split(":")
                    spans.append((int(start), int(end)))
    return sorted(set(spans))


def char_spans_to_token_labels(response, spans, tokenizer, expected_token_ids=None, response_offsets=None):
    """Map overlapping spans, rejecting any mismatch with graph token identities."""
    for start, end in spans:
        if not 0 <= start < end <= len(response):
            raise ValueError(f"Invalid annotation span {(start, end)} for text of length {len(response)}")
    if response_offsets is None:
        encoded = tokenizer(response, return_offsets_mapping=True, add_special_tokens=False)
        ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        if expected_token_ids is not None and not torch.equal(ids, expected_token_ids.cpu().long()):
            raise ValueError("Response tokenization differs from graph tokens; no truncation/padding is permitted")
        offsets = encoded["offset_mapping"]
    else:
        if expected_token_ids is None:
            raise ValueError("Context-aware offsets require graph response token IDs")
        ids = expected_token_ids
        offsets = response_offsets.tolist()
    if len(offsets) != len(ids):
        raise ValueError("Tokenizer returned inconsistent offset mapping")
    if any(not 0 <= start <= end <= len(response) for start, end in offsets):
        raise ValueError("Tokenizer offsets lie outside response text")
    labels = torch.zeros(len(ids), dtype=torch.float)
    for i, (token_start, token_end) in enumerate(offsets):
        if token_end > token_start and any(token_end > start and token_start < end for start, end in spans):
            labels[i] = 1.0
    return labels


def find_image_path(sample, image_dirs):
    candidate = sample.get("image_path", sample.get("image"))
    if candidate and Path(candidate).is_absolute() and Path(candidate).is_file():
        return Path(candidate)
    names = [candidate] if candidate else [str(sample["image_id"]) + ".jpg"]
    for root in image_dirs:
        for name in names:
            path = Path(root) / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"Image for sample {sample.get('id')} not found in --image_dirs")


def process_sample(builder, sample, out_dirs, global_idx, image_dirs):
    response = sample["hallucinated_text"]
    annotations = sample["annotations"]
    spans = {category: parse_char_spans(annotations, category) for category in HAL_TYPES}
    for category in HAL_TYPES:
        if annotations.get(category) and not spans[category]:
            raise ValueError(f"Nonempty {category} annotations have no supported char_index entries")
    present = [category for category in HAL_TYPES if spans[category]]
    graph = builder.build_graph(
        str(find_image_path(sample, image_dirs)), sample["prompt"].replace("<image>", "").strip(),
        response, int(bool(present)),
    )
    tokenizer = getattr(builder.processor, "tokenizer", builder.processor)
    unified = parse_char_spans(annotations, "unified")
    destinations = ["unified"] + (present if present else ["none"])
    prepared = []
    for category in destinations:
        labels = char_spans_to_token_labels(
            response, unified if category == "unified" else spans.get(category, []),
            tokenizer, graph.response_token_ids, getattr(graph, "response_offsets", None),
        )
        if labels.numel() != int(graph["text"].last_layer_mask.sum()):
            raise ValueError("Token labels and deepest-layer node count differ")
        if category in present and not bool(labels.any()):
            raise ValueError(f"{category} spans do not overlap any response token")
        if category == "unified" and bool(present) != bool(labels.any()):
            raise ValueError("Sentence label and aligned token annotation disagree")
        item = graph.clone()
        item["text"].token_labels = labels
        item.hal_type = category
        item.hal_types = present
        item.sample_id = str(sample["id"])
        item.image_id = str(sample["image_id"])
        path = out_dirs[category] / f"graph_{global_idx:06d}.pt"
        if path.exists():
            raise FileExistsError(f"Graph already exists: {path}")
        prepared.append((path, item))
    # Validate every label variant before writing any graph for this response.
    for path, item in prepared:
        torch.save(item, path)
    return len(prepared)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--halloc_file", required=True, help="HalLoc JSON array with split and annotations fields")
    parser.add_argument("--image_dirs", nargs="+", required=True, help="Visual Genome image folders")
    parser.add_argument("--task", choices=("caption", "instruct", "vqa"), required=True)
    parser.add_argument("--save_root", default="data/graphs", help="Writes halloc/model/task/split underneath this root")
    parser.add_argument("--data_split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int)
    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    with open(args.halloc_file, encoding="utf-8") as stream:
        samples = [item for item in json.load(stream) if item["split"] == args.data_split]
    end = len(samples) if args.end_idx is None else args.end_idx
    if not 0 <= args.start_idx < end <= len(samples):
        raise ValueError("Requested split/shard is empty or out of range")
    base = Path(args.save_root) / "halloc" / args.model_type / args.task / args.data_split
    layout = {"schema_version": 1, "unified_contains_all": True}
    marker = base / "layout.json"
    if base.exists() and any(base.glob("*/graph_*.pt")) and (
        not marker.exists() or json.loads(marker.read_text()) != layout
    ):
        raise ValueError("Destination contains a different HalLoc layout; choose a fresh root")
    out_dirs = {name: base / name for name in (*HAL_TYPES, "none", "unified")}
    for directory in out_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(layout, indent=2) + "\n")
    for idx in range(args.start_idx, end):
        if (out_dirs["unified"] / f"graph_{idx:06d}.pt").exists():
            raise FileExistsError("Output graphs already exist; choose a fresh directory or disjoint shard")
    builder = create_builder(args)
    builder.load_model()
    saved = 0
    for idx in tqdm(range(args.start_idx, end), desc="HalLoc"):
        saved += process_sample(builder, samples[idx], out_dirs, idx, args.image_dirs)
    logging.info("Saved %d graph files to %s", saved, base)


if __name__ == "__main__":
    main()
