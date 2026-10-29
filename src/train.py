"""Train TRACE with sentence/token-level supervision."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from dataset import TRACEDataset, ensure_disjoint
from metrics import compute_metrics
from runtime import (CHECKPOINT_VERSION, build_model, check_graph_compatibility,
                     get_rng_state, load_checkpoint, resolve_device,
                     seed_everything, set_rng_state, token_targets)

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", nargs="+", required=True,
                   help="One or more graph roots with train/val splits; use multiple roots to combine M-HalDetect and HalLoc.")
    p.add_argument("--split_file", help="Explicit JSON split manifest (one data root only).")
    p.add_argument("--output_dir", default="runs/trace")
    p.add_argument("--readout", choices=["mil", "attention"], default="mil")
    p.add_argument("--supervision", choices=["sentence", "token"], default="sentence",
                   help="Token supervision is the explicitly supervised upper-bound variant.")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_gnn_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lse_r", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    exclusive = p.add_mutually_exclusive_group()
    exclusive.add_argument("--resume", help="Resume this release's last.pt, including optimizer and scheduler.")
    exclusive.add_argument("--init_checkpoint", help="Initialize model weights for fine-tuning; restart optimizer/scheduler.")
    return p.parse_args(argv)


def run_epoch(model, loader, config, device, supervision, optimizer=None, max_grad_norm=1.0):
    training = optimizer is not None
    model.train(training)
    total_loss = total_items = 0
    all_labels, all_scores = [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            check_graph_compatibility(batch, config)
            if supervision == "sentence":
                logits = model(batch)
                labels = batch.y.reshape(-1).float()
            else:
                logits, _ = model.forward_token_logits(batch)
                labels = token_targets(batch, logits)
            if logits.shape != labels.shape:
                raise ValueError(f"Prediction/label shape mismatch: {logits.shape} vs {labels.shape}.")
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; inspect graph features and training settings.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
            total_loss += float(loss.detach()) * labels.numel()
            total_items += labels.numel()
            all_labels.extend(labels.detach().cpu().int().tolist())
            all_scores.extend(logits.detach().sigmoid().cpu().tolist())
    return total_loss / total_items, compute_metrics(all_labels, all_scores)


def train(args):
    if args.supervision == "token" and args.readout != "mil":
        raise ValueError("Token supervision uses the MIL token prediction head; select --readout mil.")
    if args.split_file and len(args.data_root) != 1:
        raise ValueError("--split_file accepts exactly one --data_root.")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative.")
    if not 0 <= args.min_lr <= args.lr or args.lr <= 0 or args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise ValueError("Invalid learning-rate, weight-decay, or gradient-clipping setting.")
    output = Path(args.output_dir)
    if not args.resume and any((output / name).exists() for name in ("best.pt", "last.pt", "config.json", "history.json")):
        raise FileExistsError(f"Run artifacts already exist in {output}; choose a new --output_dir or use --resume.")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_sets = [TRACEDataset(root, "train", args.split_file, token_labels=args.supervision == "token") for root in args.data_root]
    val_sets = [TRACEDataset(root, "val", args.split_file, token_labels=args.supervision == "token") for root in args.data_root]
    for train_set in train_sets:
        for val_set in val_sets:
            ensure_disjoint(train_set, val_set)
    for sets in (train_sets, val_sets):
        files = [p.resolve() for ds in sets for p in ds.file_list]
        if len(files) != len(set(files)):
            raise ValueError("Repeated data roots or overlapping manifests duplicate samples.")
    sample = train_sets[0][0]
    config = {"readout": args.readout, "text_input_dim": sample["text"].x.size(1),
              "vis_input_dim": sample["vision"].x.size(1), "hidden_dim": args.hidden_dim,
              "num_layers": args.num_gnn_layers, "num_heads": args.num_heads,
              "dropout": args.dropout, "lse_r": args.lse_r}
    for ds in train_sets + val_sets:
        check_graph_compatibility(ds[0], config)
    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    start_epoch, best_aupr, history = 0, float("-inf"), []
    checkpoint_path = args.resume or args.init_checkpoint
    if checkpoint_path:
        restored, checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint["model_config"] != config:
            raise ValueError("Checkpoint architecture differs from requested configuration; use matching model options.")
        model.load_state_dict(restored.state_dict(), strict=True)
        del restored
        if args.resume:
            required = {"optimizer", "scheduler", "epoch", "best_aupr", "rng_state", "train_config", "history"}
            if not required.issubset(checkpoint):
                raise ValueError("Resume requires a complete training checkpoint (last.pt).")
            for key in ("epochs", "batch_size", "lr", "min_lr", "weight_decay", "max_grad_norm", "seed", "supervision", "data_root", "split_file", "num_workers"):
                if checkpoint["train_config"].get(key) != getattr(args, key):
                    raise ValueError(f"Cannot resume with a changed {key}; use --init_checkpoint for a new training run.")
            if Path(checkpoint_path).resolve().parent != output.resolve():
                raise ValueError("--resume must use the original run directory as --output_dir.")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch, best_aupr = checkpoint["epoch"] + 1, checkpoint["best_aupr"]
            history = checkpoint["history"]
            set_rng_state(checkpoint["rng_state"])
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"model": config, "training": vars(args)}, indent=2) + "\n")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_metrics = run_epoch(model, train_loader, config, device, args.supervision, optimizer, args.max_grad_norm)
        val_loss, val_metrics = run_epoch(model, val_loader, config, device, args.supervision)
        if val_metrics["aupr"] is None:
            raise ValueError("Validation requires both classes for AUPR checkpoint selection.")
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
               "train": train_metrics, "validation": val_metrics, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        is_best = val_metrics["aupr"] > best_aupr
        best_aupr = max(best_aupr, val_metrics["aupr"])
        checkpoint = {"format_version": CHECKPOINT_VERSION, "model_config": config,
                      "model": model.state_dict(), "epoch": epoch, "best_aupr": best_aupr,
                      "train_config": vars(args), "history": history,
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "rng_state": get_rng_state()}
        if is_best:
            torch.save(checkpoint, output / "best.pt")
        torch.save(checkpoint, output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        logger.info("Epoch %d/%d: train loss %.4f, val loss %.4f, val AUPR %.4f", epoch + 1, args.epochs, train_loss, val_loss, val_metrics["aupr"])
    logger.info("Best validation AUPR: %.4f; checkpoint: %s", best_aupr, output / "best.pt")
    return output / "best.pt"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(parse_args())
