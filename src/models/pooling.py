"""Response readouts: token-first MIL and optional attention pooling."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter, softmax


def _validate_batch(values: Tensor, batch: Tensor) -> int:
    """Require one nonempty token bag per contiguous, zero-based graph ID."""
    if values.ndim < 1 or batch.ndim != 1 or values.size(0) != batch.numel():
        raise ValueError("batch must contain one graph ID per token")
    if not batch.numel():
        raise ValueError("readout requires at least one response token per graph")
    if batch.dtype != torch.long or batch.device != values.device:
        raise ValueError("batch must be int64 and on the same device as token features")
    if bool((batch < 0).any()):
        raise ValueError("graph IDs must be nonnegative")
    num_graphs = int(batch.max()) + 1
    counts = torch.bincount(batch, minlength=num_graphs)
    if bool((counts == 0).any()):
        raise ValueError("every graph must contain at least one response token")
    return num_graphs


def normalized_logsumexp(token_logits: Tensor, batch: Tensor, r: float = 0.5) -> Tensor:
    """Pool logits by Eq. (6): ``(logsumexp(r*z) - log(n)) / r`` per graph.

    Subtracting each bag's maximum before exponentiation avoids overflow. The
    detached maximum is only a numerical shift; gradients remain softmax(r*z).
    Half precision inputs are accumulated in float32.
    """
    if not math.isfinite(r) or r <= 0:
        raise ValueError("MIL lse_r must be finite and strictly positive")
    if token_logits.ndim != 1:
        raise ValueError("token_logits must have shape [number_of_response_tokens]")
    num_graphs = _validate_batch(token_logits, batch)
    logits = token_logits.float() if token_logits.dtype in (torch.float16, torch.bfloat16) else token_logits
    scaled = r * logits
    maxima = scatter(scaled.detach(), batch, dim=0, dim_size=num_graphs, reduce="max")
    sums = scatter((scaled - maxima[batch]).exp(), batch, dim=0, dim_size=num_graphs, reduce="sum")
    counts = torch.bincount(batch, minlength=num_graphs).to(dtype=logits.dtype)
    return (maxima + sums.log() - counts.log()) / r


class AttentionReadout(nn.Module):
    """Pool features and return per-graph normalized token attribution weights."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        num_graphs = _validate_batch(x, batch)
        weights = softmax(self.attn_mlp(x).squeeze(-1), batch, num_nodes=num_graphs)
        features = scatter(x * weights.unsqueeze(-1), batch, dim=0, dim_size=num_graphs, reduce="sum")
        return features, weights


class PredictionHead(nn.Module):
    """Paper MLP: 256 -> 128 -> 1 at the default hidden width."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(features).squeeze(-1)


class TokenFirstReadout(nn.Module):
    """Predict token logits, then pool them using normalized LSE (r=0.5)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, lse_r: float = 0.5):
        super().__init__()
        if not math.isfinite(lse_r) or lse_r <= 0:
            raise ValueError("MIL lse_r must be finite and strictly positive")
        self.pred_head = PredictionHead(hidden_dim, dropout)
        self.lse_r = float(lse_r)

    def forward(self, x: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        token_logits = self.pred_head(x)
        graph_logits = normalized_logsumexp(token_logits, batch, self.lse_r)
        return graph_logits, token_logits
