"""
Pure, model-independent functions for constructing the three TRACE edge types.

Prerequisite: the following per-layer features have been extracted from the VLM:
  - h_text_list   : List[Tensor]  Response token hidden states per layer [n, D]
  - h_vision_list : List[Tensor]  Visual node features per layer [D]
  - attn_t2t_list : List[Tensor]  Text-to-text attention per layer [n, n]
  - attn_t2v_list : List[Tensor]  Text-to-vision attention per layer [n] (top-k visual attention scores already averaged)

Global node indexing convention
-----------------
text   node(layer=k, token=i) = k * n + i
vision node(layer=k)          = k
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def build_evolve_edges(
    h_text_list: List[Tensor],
) -> Tuple[Tensor, Tensor]:
    """
    Cross-layer evolution edges E_evolve: t_i^{l_{k-1}} → t_i^{l_k}

    Edge feature = CosineSimilarity(h_i^{l_{k-1}}, h_i^{l_k}), a scalar.

    Args:
        h_text_list: K tensors of shape [n, D] on the same device.

    Returns:
        edge_index : [2, (K-1)*n]  long
        edge_attr  : [(K-1)*n, 1]  float
    """
    K = len(h_text_list)
    n = h_text_list[0].size(0)

    device = h_text_list[0].device
    if K <= 1:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, 1, device=device),
        )

    token_ids = torch.arange(n, device=device)
    src_list, dst_list, attr_list = [], [], []

    for k in range(K - 1):
        h_prev = h_text_list[k].float()       # [n, D]
        h_next = h_text_list[k + 1].float()   # [n, D]
        cos_sim = F.cosine_similarity(h_prev, h_next, dim=-1)  # [n]

        src_list.append(k * n + token_ids)
        dst_list.append((k + 1) * n + token_ids)
        attr_list.append(cos_sim)

    edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)])
    edge_attr  = torch.cat(attr_list).unsqueeze(-1)
    return edge_index, edge_attr


def build_context_edges(
    attn_t2t_list: List[Tensor],
    attn_threshold: float = 0.05,
) -> Tuple[Tensor, Tensor]:
    """
    Within-layer context edges E_context: t_j^l → t_i^l (j ≤ i and α_{i,j}^l > τ)

    Edge feature = self-attention score α_{i,j}^l, averaged across attention heads.

    Args:
        attn_t2t_list  : K tensors of shape [n, n]; attn[i, j] is the attention from token i to token j.
        attn_threshold : Sparsification threshold τ; retain edges with attention scores greater than τ.

    Returns:
        edge_index : [2, E_context]  long  (src=j, dst=i)
        edge_attr  : [E_context, 1]  float
    """
    K   = len(attn_t2t_list)
    n   = attn_t2t_list[0].size(0)
    src_list, dst_list, attr_list = [], [], []

    for k, attn in enumerate(attn_t2t_list):
        attn = attn.float()
        mask = (attn > attn_threshold).tril()                # [n, n]
        i_idx, j_idx = mask.nonzero(as_tuple=True)  # dst, src
        if i_idx.numel() == 0:
            continue
        src_list.append(k * n + j_idx)
        dst_list.append(k * n + i_idx)
        attr_list.append(attn[i_idx, j_idx])

    if not src_list:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=attn_t2t_list[0].device),
            torch.zeros(0, 1, device=attn_t2t_list[0].device),
        )

    edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)])
    edge_attr  = torch.cat(attr_list).unsqueeze(-1)
    return edge_index, edge_attr


def build_cross_edges(
    attn_t2v_list: List[Tensor],
) -> Tuple[Tensor, Tensor]:
    """
    Cross-modal bridging edges E_cross: v^l → t_i^l (one visual node connects to all response tokens in each layer)

    Edge feature = mean attention from response token i to the top-k visual tokens (a precomputed scalar).

    Args:
        attn_t2v_list: K tensors of shape [n]; each entry is a response token's mean top-k attention to image regions in that layer.

    Returns:
        edge_index : [2, K*n]  long  (src=vision node k, dst=text node k*n+i)
        edge_attr  : [K*n, 1]  float
    """
    K = len(attn_t2v_list)
    n = attn_t2v_list[0].size(0)
    device = attn_t2v_list[0].device
    token_ids = torch.arange(n, device=device)

    src_list, dst_list, attr_list = [], [], []
    for k, attn_vec in enumerate(attn_t2v_list):
        src_list.append(torch.full((n,), k, dtype=torch.long, device=device))
        dst_list.append(k * n + token_ids)
        attr_list.append(attn_vec.float())

    edge_index = torch.stack([torch.cat(src_list), torch.cat(dst_list)])
    edge_attr  = torch.cat(attr_list).unsqueeze(-1)
    return edge_index, edge_attr
