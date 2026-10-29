"""Shared model configuration, checkpoint loading, and inference utilities."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from models import TRACEModel, TRACEModelMIL

CHECKPOINT_VERSION = 1
MODEL_FIELDS = {"readout", "text_input_dim", "vis_input_dim", "hidden_dim", "num_layers", "num_heads", "dropout", "lse_r"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    return device


def build_model(config: dict):
    if set(config) != MODEL_FIELDS:
        raise ValueError(f"model_config must contain exactly {sorted(MODEL_FIELDS)}.")
    kwargs = {k: v for k, v in config.items() if k not in {"readout", "lse_r"}}
    if config["readout"] == "mil":
        return TRACEModelMIL(**kwargs, lse_r=config["lse_r"])
    if config["readout"] == "attention":
        return TRACEModel(**kwargs)
    raise ValueError(f"Unknown readout: {config['readout']}")


def check_graph_compatibility(graph, config: dict) -> None:
    if graph["text"].x.size(1) != config["text_input_dim"] or graph["vision"].x.size(1) != config["vis_input_dim"]:
        raise ValueError("Graph hidden dimensions do not match the checkpoint/model configuration.")
    # A single graph has one visual anchor per sampled transformer layer.
    if not hasattr(graph["vision"], "batch"):
        max_layers = graph["vision"].num_nodes
    else:
        max_layers = int(torch.bincount(graph["vision"].batch).max())
    if config["num_layers"] < max_layers:
        raise ValueError(f"Message-passing depth M={config['num_layers']} must be at least sampled graph depth K={max_layers}.")


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint format. Expected a versioned TRACE checkpoint with model_config and model weights.")
    if "model_config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Checkpoint must contain model_config and model weights.")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def get_rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def token_targets(batch, token_scores: torch.Tensor) -> torch.Tensor:
    if "token_labels" not in batch["text"]:
        raise ValueError("Token supervision/evaluation requires exact per-response-token labels.")
    labels = batch["text"].token_labels.reshape(-1).float()
    if labels.shape != token_scores.shape:
        raise ValueError(f"Token-label alignment mismatch: {labels.shape} vs {token_scores.shape}.")
    return labels
