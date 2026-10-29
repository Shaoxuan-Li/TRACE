"""Evaluate sentence detection and MIL token localization."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from dataset import HALLOC_CATEGORIES, TRACEDataset
from metrics import compute_metrics
from runtime import check_graph_compatibility, load_checkpoint, resolve_device, token_targets

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--split_file")
    p.add_argument("--category", choices=HALLOC_CATEGORIES, default="unified")
    p.add_argument("--exclude_none", action="store_true", help="For category layouts, omit separate factual samples.")
    p.add_argument("--token_eval", action="store_true", help="Require exact token labels and report token localization metrics.")
    p.add_argument("--token_scope", choices=["all", "positive"], default="positive",
                   help="Evaluate tokens from hallucinated responses by default; all includes factual responses.")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for additional fixed-threshold sentence metrics.")
    p.add_argument("--token_threshold", type=float, default=0.5, help="Threshold for additional fixed-threshold token metrics.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", help="Default: checkpoint directory/evaluation/split/category.")
    p.add_argument("--save_predictions", action="store_true", help="Write per-graph probabilities and optional token scores as JSONL.")
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(args):
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if not 0 <= args.threshold <= 1 or not 0 <= args.token_threshold <= 1:
        raise ValueError("Thresholds must lie in [0, 1].")
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["model_config"]
    if args.token_eval and config["readout"] != "mil":
        raise ValueError("Attention readout returns attribution weights, not token hallucination probabilities; token evaluation requires an MIL checkpoint.")
    dataset = TRACEDataset(args.data_root, args.split, args.split_file,
                          token_labels=args.token_eval, category=args.category,
                          include_none=not args.exclude_none)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    output = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "evaluation" / args.split / args.category
    output.mkdir(parents=True, exist_ok=True)
    sent_labels, sent_scores, tok_labels, tok_scores = [], [], [], []
    prediction_file = (output / "predictions.jsonl").open("w") if args.save_predictions else None
    offset = 0
    try:
        for batch in loader:
            batch = batch.to(device)
            check_graph_compatibility(batch, config)
            labels = batch.y.reshape(-1)
            if args.token_eval:
                scores, token_scores, batch_last = model.predict_with_token_scores(batch)
                token_labels = token_targets(batch, token_scores)
                keep = torch.ones_like(token_labels, dtype=torch.bool)
                if args.token_scope == "positive":
                    keep = labels[batch_last].bool()
                tok_labels.extend(token_labels[keep].cpu().int().tolist())
                tok_scores.extend(token_scores[keep].cpu().tolist())
            else:
                scores = model(batch).sigmoid()
            if scores.shape != labels.shape:
                raise ValueError("Sentence scores and labels do not align.")
            sent_labels.extend(labels.cpu().int().tolist())
            sent_scores.extend(scores.cpu().tolist())
            if prediction_file:
                for idx in range(labels.numel()):
                    row = {"graph_file": str(dataset.file_list[offset + idx]),
                           "label": int(labels[idx]), "probability": float(scores[idx])}
                    if args.token_eval:
                        token_mask = batch_last == idx
                        row["token_labels"] = token_labels[token_mask].cpu().int().tolist()
                        row["token_probabilities"] = token_scores[token_mask].cpu().tolist()
                    prediction_file.write(json.dumps(row, allow_nan=False) + "\n")
            offset += labels.numel()
    finally:
        if prediction_file:
            prediction_file.close()
    results = {"split": args.split, "category": args.category, "readout": config["readout"],
               "model_config": config, "checkpoint_epoch": checkpoint.get("epoch", -1) + 1,
               "checkpoint_supervision": checkpoint.get("train_config", {}).get("supervision"),
               "sentence": compute_metrics(sent_labels, sent_scores, args.threshold, select_threshold=True),
               "metric_convention": "aupr = trapezoidal precision-recall area; average_precision is reported separately; tied scores grouped by scikit-learn"}
    if args.token_eval:
        results["token_scope"] = args.token_scope
        results["token"] = compute_metrics(tok_labels, tok_scores, args.token_threshold, select_threshold=True) if tok_labels else None
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    logger.info("%s", json.dumps(results, indent=2, allow_nan=False))
    for level in ("sentence", "token"):
        metrics = results.get(level)
        if metrics:
            logger.info("%s Best F1=%s; AUPR=%s", level.capitalize(),
                        metrics["best_f1"], metrics["aupr"])
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate(parse_args())
