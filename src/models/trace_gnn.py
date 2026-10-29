"""TRACE heterogeneous graph encoder and response-level detector interfaces.

Graphs contain text nodes at each sampled layer, one visual anchor per layer,
scalar attributes for evolve/context/cross edges, and a text last_layer_mask.
Vision anchors are source-only. Only deepest-layer response nodes reach readout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter, softmax

from .pooling import AttentionReadout, PredictionHead, TokenFirstReadout, _validate_batch


class GatedGATConv(nn.Module):
    """
    GAT attention times an edge-conditioned scalar sigmoid gate (Eq. 7).
    """

    def __init__(
        self,
        in_channels: int | tuple[int, int],
        out_channels: int,
        heads: int = 4,
        edge_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if out_channels < 1 or heads < 1 or edge_dim < 1:
            raise ValueError("out_channels, heads, and edge_dim must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        in_src, in_dst = in_channels if isinstance(in_channels, tuple) else (in_channels, in_channels)
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.lin_src = nn.Linear(in_src, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_dst, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        # One scalar gate per edge, shared across attention heads.
        self.gate_mlp = nn.Sequential(
            nn.Linear(edge_dim, max(8, out_channels)),
            nn.GELU(),
            nn.Linear(max(8, out_channels), 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.lin_src, self.lin_dst):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        for layer in self.gate_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor | tuple[Tensor, Tensor], edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x_src, x_dst = x if isinstance(x, tuple) else (x, x)
        n_dst = x_dst.size(0)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, number_of_edges]")
        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError("edge_attr must contain one attribute per edge")
        if edge_index.numel() == 0:
            return x_dst.new_zeros((n_dst, self.heads * self.out_channels))
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.ndim == 1 else edge_attr
        edge_attr = edge_attr.to(dtype=x_src.dtype, device=x_src.device)
        src, dst = edge_index
        source = self.lin_src(x_src).view(-1, self.heads, self.out_channels)
        target = self.lin_dst(x_dst).view(-1, self.heads, self.out_channels)
        alpha = (source[src] * self.att_src).sum(-1) + (target[dst] * self.att_dst).sum(-1)
        alpha = softmax(F.leaky_relu(alpha, negative_slope=0.2), dst, num_nodes=n_dst)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        gate = torch.sigmoid(self.gate_mlp(edge_attr))
        messages = (alpha * gate).unsqueeze(-1) * source[src]
        return scatter(messages, dst, dim=0, dim_size=n_dst, reduce="sum").reshape(n_dst, -1)


class TRACEGNNLayer(nn.Module):
    """Relation sum -> GELU -> residual -> LayerNorm; vision stays unchanged."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim < 2 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be >= 2 and divisible by num_heads")
        head_dim = hidden_dim // num_heads
        self.evolve_conv = GatedGATConv(hidden_dim, head_dim, num_heads, dropout=dropout)
        self.context_conv = GatedGATConv(hidden_dim, head_dim, num_heads, dropout=dropout)
        self.cross_conv = GatedGATConv((hidden_dim, hidden_dim), head_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(
        self, x_text: Tensor, x_vision: Tensor,
        evolve_edge_index: Tensor, evolve_edge_attr: Tensor,
        context_edge_index: Tensor, context_edge_attr: Tensor,
        cross_edge_index: Tensor, cross_edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        messages = (
            self.evolve_conv(x_text, evolve_edge_index, evolve_edge_attr)
            + self.context_conv(x_text, context_edge_index, context_edge_attr)
            + self.cross_conv((x_vision, x_text), cross_edge_index, cross_edge_attr)
        )
        return self.norm(self.act(messages) + x_text), x_vision


class _TRACEEncoder(nn.Module):
    """Shared heterogeneous graph encoder for both readout models."""

    def __init__(self, text_input_dim: int, vis_input_dim: int, hidden_dim: int,
                 num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        if text_input_dim < 1 or vis_input_dim < 1 or num_layers < 1:
            raise ValueError("input dimensions and num_layers must be positive")
        self.text_proj = nn.Sequential(nn.Linear(text_input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.vis_proj = nn.Sequential(nn.Linear(vis_input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList(TRACEGNNLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers))

    @staticmethod
    def _get_edge(data: HeteroData, src_type: str, rel: str, dst_type: str) -> tuple[Tensor, Tensor]:
        key = (src_type, rel, dst_type)
        if key not in data.edge_types:
            raise ValueError(f"missing graph relation {key}; represent an empty relation explicitly")
        store = data[key]
        attr = store.edge_attr
        return store.edge_index, attr.unsqueeze(-1) if attr.ndim == 1 else attr

    def _encode_last(self, data: HeteroData) -> tuple[Tensor, Tensor]:
        x_text = self.text_proj(data["text"].x)
        x_vision = self.vis_proj(data["vision"].x)
        edges = [
            self._get_edge(data, "text", "evolve", "text"),
            self._get_edge(data, "text", "context", "text"),
            self._get_edge(data, "vision", "cross", "text"),
        ]
        for layer in self.layers:
            x_text, x_vision = layer(x_text, x_vision, *edges[0], *edges[1], *edges[2])
        mask = data["text"].last_layer_mask
        if mask.ndim != 1 or mask.numel() != x_text.size(0) or mask.dtype != torch.bool:
            raise ValueError("text.last_layer_mask must be a boolean mask for all text nodes")
        batch = data["text"].get("batch")
        if batch is None:
            batch = torch.zeros(x_text.size(0), dtype=torch.long, device=x_text.device)
        num_graphs = _validate_batch(x_text, batch)
        x_last, batch_last = x_text[mask], batch[mask]
        _validate_batch(x_last, batch_last)
        if torch.bincount(batch_last, minlength=num_graphs).eq(0).any():
            raise ValueError("every graph must contain deepest-layer response tokens")
        return x_last, batch_last


class TRACEModel(_TRACEEncoder):
    """Attention readout with response prediction and token attribution.

    Per-graph normalized attention weights describe token contributions
    to the pooled representation.
    """

    def __init__(self, text_input_dim: int, vis_input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 8, num_heads: int = 4, dropout: float = 0.1):
        super().__init__(text_input_dim, vis_input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.readout = AttentionReadout(hidden_dim, dropout)
        self.pred_head = PredictionHead(hidden_dim, dropout)

    def forward(self, data: HeteroData) -> Tensor:
        return self.forward_with_attn(data)[0]

    def forward_with_attn(self, data: HeteroData) -> tuple[Tensor, Tensor, Tensor]:
        """Return response logits, attention attribution weights, and token bag IDs."""
        features, batch = self._encode_last(data)
        pooled, weights = self.readout(features, batch)
        return self.pred_head(pooled), weights, batch

    def predict_with_token_scores(self, data: HeteroData) -> tuple[Tensor, Tensor, Tensor]:
        """Return response probabilities, attention attribution, and token bag IDs.

        Call model.eval() for deterministic inference. The second return value
        contains per-graph normalized attention weights.
        """
        logits, attribution, batch = self.forward_with_attn(data)
        return logits.sigmoid(), attribution, batch


class TRACEModelMIL(_TRACEEncoder):
    """Paper detector: per-token prediction followed by normalized LSE pooling."""

    def __init__(self, text_input_dim: int, vis_input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 8, num_heads: int = 4, dropout: float = 0.1,
                 lse_r: float = 0.5):
        super().__init__(text_input_dim, vis_input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.readout = TokenFirstReadout(hidden_dim, dropout, lse_r)

    def _encode(self, data: HeteroData) -> tuple[Tensor, Tensor]:
        """Encode deepest-layer text nodes and their graph IDs."""
        return self._encode_last(data)

    def forward(self, data: HeteroData) -> Tensor:
        return self.forward_with_token_logits(data)[0]

    def forward_with_token_logits(self, data: HeteroData) -> tuple[Tensor, Tensor, Tensor]:
        """Return response logits, token logits, and token bag IDs in one pass."""
        features, batch = self._encode_last(data)
        graph_logits, token_logits = self.readout(features, batch)
        return graph_logits, token_logits, batch

    def forward_token_logits(self, data: HeteroData) -> tuple[Tensor, Tensor]:
        """Optional token-supervised extension; returns raw token logits and IDs."""
        features, batch = self._encode_last(data)
        return self.readout.pred_head(features), batch

    def predict_with_token_scores(self, data: HeteroData) -> tuple[Tensor, Tensor, Tensor]:
        """Return response probabilities, token hallucination probabilities, IDs.

        Call model.eval() for deterministic inference and use torch.no_grad()
        when gradients are not needed.
        """
        graph_logits, token_logits, batch = self.forward_with_token_logits(data)
        return graph_logits.sigmoid(), token_logits.sigmoid(), batch
