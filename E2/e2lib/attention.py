"""Multi-head self-attention twice: written from basic ops (MSAScratch) and PyTorch's own (MSATorch).
Both have the same interface and can be loaded with IDENTICAL weights (load_canonical), so any difference in results comes
from the implementation, not from the initialisation.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_head)) V         head_i uses its own slice of W_q, W_k, W_v
    MSA(x) = Concat(head_1 ... head_h) W_o
"""
import math

import torch
import torch.nn as nn


def canonical_weights(dim, seed):
    """Deterministic Xavier-uniform W_q, W_k, W_v, W_o of shape (dim, dim) from a private generator. Biases are zero."""
    g = torch.Generator().manual_seed(seed)
    bound = math.sqrt(6.0 / (dim + dim))
    return [(torch.rand(dim, dim, generator=g) * 2 - 1) * bound for _ in range(4)]


class MSAScratch(nn.Module):
    """No nn.MultiheadAttention: only Linear, einsum, softmax and reshape."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by the number of heads"
        self.h, self.dh = heads, dim // heads
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x, return_attn=False):                     # x: (B, N, D)
        B, N, D = x.shape
        q, k, v = (w(x).view(B, N, self.h, self.dh) for w in (self.wq, self.wk, self.wv))   # (B, N, h, dh)
        scores = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(self.dh)                 # (B, h, N, N)
        attn = scores.softmax(dim=-1)                              # each query's weights over all keys sum to 1
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v).reshape(B, N, D)                     # concatenate the heads
        out = self.wo(out)
        return (out, attn) if return_attn else out

    @torch.no_grad()
    def load_canonical(self, wq, wk, wv, wo):
        for lin, w in zip((self.wq, self.wk, self.wv, self.wo), (wq, wk, wv, wo)):
            lin.weight.copy_(w)
            lin.bias.zero_()


class MSATorch(nn.Module):
    """Reference: nn.MultiheadAttention (batch_first). Without return_attn PyTorch may use its fused fast kernels."""

    def __init__(self, dim, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x, return_attn=False):
        out, attn = self.mha(x, x, x, need_weights=return_attn, average_attn_weights=False)
        return (out, attn) if return_attn else out

    @torch.no_grad()
    def load_canonical(self, wq, wk, wv, wo):
        self.mha.in_proj_weight.copy_(torch.cat([wq, wk, wv], dim=0))   # rows: [W_q; W_k; W_v]
        self.mha.in_proj_bias.zero_()
        self.mha.out_proj.weight.copy_(wo)
        self.mha.out_proj.bias.zero_()


IMPLS = {"scratch": MSAScratch, "torch": MSATorch}
