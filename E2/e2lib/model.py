"""Shallow Transformer encoder classifier: tokenizer -> [CLS] + positional embedding -> depth x (LN, MSA, LN, FFN) -> LN -> head."""
import torch
import torch.nn as nn

from .attention import IMPLS, canonical_weights
from .tokenizers import make_tokenizer


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout, impl):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = IMPLS[impl](dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout))

    def forward(self, x, return_attn=False):
        if return_attn:
            a, attn = self.attn(self.ln1(x), return_attn=True)
        else:
            a, attn = self.attn(self.ln1(x)), None
        x = x + a                                   # residual around attention
        x = x + self.mlp(self.ln2(x))               # residual around FFN
        return x, attn


class TinyTransformer(nn.Module):
    def __init__(self, tokenizer="patch4", attn="scratch", dim=128, depth=4, heads=4, mlp_ratio=4, dropout=0.1,
                 use_pos=True, num_classes=10, img=32, seed=0):
        super().__init__()
        self.tok = make_tokenizer(tokenizer, dim, img)
        self.n_tokens = self.tok.n + 1                              # + CLS
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim), requires_grad=use_pos)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, dropout, attn) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.use_pos = use_pos
        # Same initial weights for both attention implementations. The two classes draw a different number of random numbers
        # in their constructors, so everything that is NOT attention is re-initialised here from a fixed seed, and the
        # attention weights come from `canonical_weights` (depends only on seed and block index).
        torch.manual_seed(seed * 7 + 1)
        nn.init.trunc_normal_(self.pos, std=0.02)                   # drawn AFTER seeding, otherwise pos/cls differ per model
        nn.init.trunc_normal_(self.cls, std=0.02)
        if not use_pos:
            with torch.no_grad():
                self.pos.zero_()
        for name, m in self.named_modules():
            if ".attn" not in name and isinstance(m, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d)):
                m.reset_parameters()
        for i, b in enumerate(self.blocks):
            b.attn.load_canonical(*canonical_weights(dim, seed * 1000 + i))

    def forward(self, x, return_attn=False):
        t = self.tok(x)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1) + self.pos
        t = self.drop(t)
        attns = []
        for b in self.blocks:
            t, a = b(t, return_attn)
            attns.append(a)
        logits = self.head(self.norm(t)[:, 0])                      # classify from the CLS token
        return (logits, attns) if return_attn else logits
