"""Load pre-extracted TRACE graphs from explicit benchmark splits.

Supported layouts are ``root/split/graph_*.pt`` and HalLoc's
``root/split/{unified,none}/graph_*.pt``. A JSON split manifest can instead
map split names to graph paths relative to root. No random split is inferred.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import Dataset, HeteroData

RELATIONS = (
    ("text", "evolve", "text"),
    ("text", "context", "text"),
    ("vision", "cross", "text"),
)
HALLOC_CATEGORIES = ("unified", "object", "attribute", "relationship", "scene", "none")


class TRACEDataset(Dataset):
    """Lazy graph dataset; load only graph files created by a trusted source.

    ``token_labels=True`` requires one label per deepest-layer response token.
    Other metadata is deliberately kept out of PyG collation, so graphs from
    different sources can be combined without mismatched optional attributes.
    """

    def __init__(self, root: str, split: str = "train", split_file: str | None = None,
                 transform: Callable | None = None, token_labels: bool = False,
                 category: str = "unified", include_none: bool = True):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if category not in HALLOC_CATEGORIES:
            raise ValueError(f"Unknown HalLoc category: {category}")
        self.split = split
        self.load_token_labels = token_labels
        root_path = Path(root).expanduser().resolve()
        if split_file is not None:
            if category != "unified" or not include_none:
                raise ValueError("Category filtering must be encoded in the explicit split manifest.")
            mapping = json.loads(Path(split_file).read_text())
            if not isinstance(mapping, dict) or split not in mapping:
                raise ValueError(f"Manifest must contain a '{split}' list.")
            seen = {}
            resolved = {}
            for name, entries in mapping.items():
                if not isinstance(entries, list):
                    raise ValueError(f"Manifest split '{name}' must be a list of relative paths.")
                resolved[name] = []
                for entry in entries:
                    if not isinstance(entry, str) or Path(entry).is_absolute():
                        raise ValueError("Manifest entries must be relative graph-file paths.")
                    path = (root_path / entry).resolve()
                    if not path.is_relative_to(root_path):
                        raise ValueError(f"Manifest path escapes data root: {entry}")
                    if path in seen:
                        raise ValueError(f"Duplicate graph across/in splits '{seen[path]}' and '{name}': {entry}")
                    seen[path] = name
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    resolved[name].append(path)
            self.file_list = resolved[split]
        else:
            split_dir = root_path / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Missing {split_dir}. Provide explicit split directories or --split_file; random splitting is disabled.")
            flat = sorted(split_dir.glob("graph_*.pt"))
            has_categories = any((split_dir / name).is_dir() for name in HALLOC_CATEGORIES)
            if flat and has_categories:
                raise ValueError(f"Ambiguous mixed flat/category layout in {split_dir}; use a manifest.")
            if flat:
                if category != "unified" or not include_none:
                    raise ValueError("Category filters require HalLoc category subdirectories.")
                self.file_list = flat
            else:
                layout_file = split_dir / "layout.json"
                layout = json.loads(layout_file.read_text()) if layout_file.exists() else {}
                unified_all = layout.get("unified_contains_all", False)
                if category == "unified" and unified_all and not include_none:
                    raise ValueError("unified already contains all samples; use an explicit manifest to filter factual responses.")
                selected = [category]
                if include_none and category != "none" and not (category == "unified" and unified_all):
                    selected.append("none")
                self.file_list = [p for name in selected for p in sorted((split_dir / name).glob("graph_*.pt"))]
        if not self.file_list:
            raise FileNotFoundError(f"No graph_*.pt files in split '{split}' under {root_path}.")
        canonical = [p.resolve() for p in self.file_list]
        if len(canonical) != len(set(canonical)):
            raise ValueError("Split contains duplicate files or symlinks to the same graph.")
        super().__init__(root=str(root_path), transform=transform)

    def len(self) -> int:
        return len(self.file_list)

    def get(self, idx: int) -> HeteroData:
        source = self.file_list[idx]
        raw = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(raw, HeteroData):
            raise TypeError(f"Expected a PyG HeteroData graph: {source}")
        graph = HeteroData()
        for kind in ("text", "vision"):
            x = raw[kind].x
            if x.ndim != 2 or not x.size(0) or not x.size(1) or not torch.isfinite(x).all():
                raise ValueError(f"Invalid {kind} node features: {source}")
            graph[kind].x = x.float()
        mask = raw["text"].last_layer_mask.reshape(-1).bool()
        n_text, n_layers = graph["text"].x.size(0), graph["vision"].x.size(0)
        n_tokens = int(mask.sum())
        if mask.numel() != n_text or n_tokens == 0 or n_tokens * n_layers != n_text:
            raise ValueError(f"Invalid layer/token node layout: {source}")
        expected_mask = torch.arange(n_text) >= n_text - n_tokens
        if not torch.equal(mask, expected_mask):
            raise ValueError(f"last_layer_mask must select the deepest layer in layer-major order: {source}")
        graph["text"].last_layer_mask = mask
        for relation in RELATIONS:
            if relation not in raw.edge_types:
                raise ValueError(f"Missing relation {relation}: {source}")
            edge_index = raw[relation].edge_index
            edge_attr = raw[relation].edge_attr.reshape(-1, 1)
            if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
                raise ValueError(f"Invalid edge_index for {relation}: {source}")
            if edge_attr.size(0) != edge_index.size(1) or not torch.isfinite(edge_attr).all():
                raise ValueError(f"Invalid edge_attr for {relation}: {source}")
            if edge_index.numel():
                n_src, n_dst = graph[relation[0]].num_nodes, graph[relation[2]].num_nodes
                if edge_index.min() < 0 or edge_index[0].max() >= n_src or edge_index[1].max() >= n_dst:
                    raise ValueError(f"Out-of-bounds edge for {relation}: {source}")
            graph[relation].edge_index = edge_index
            graph[relation].edge_attr = edge_attr.float()
        label = raw.y.reshape(-1)
        if label.numel() != 1 or label.item() not in (0, 1):
            raise ValueError(f"Expected one binary sentence label: {source}")
        graph.y = label.float()
        if self.load_token_labels:
            if "token_labels" not in raw["text"]:
                raise ValueError(f"Missing token labels: {source}")
            labels = raw["text"].token_labels.reshape(-1)
            if labels.numel() != n_tokens or not torch.all((labels == 0) | (labels == 1)):
                raise ValueError(f"Token labels must align exactly with {n_tokens} deepest-layer tokens: {source}")
            graph["text"].token_labels = labels.float()
        return graph


def ensure_disjoint(train_set: TRACEDataset, val_set: TRACEDataset) -> None:
    """Reject the same physical graph in train and validation splits."""
    overlap = set(p.resolve() for p in train_set.file_list) & set(p.resolve() for p in val_set.file_list)
    if overlap:
        raise ValueError(f"Train/validation overlap: {next(iter(overlap))}")
