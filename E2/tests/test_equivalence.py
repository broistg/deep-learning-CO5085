"""Run from the repo root:  PYTHONPATH=.:E2 python -m unittest discover -s E2/tests -v"""
import unittest

import numpy as np

try:
    import torch
    from e2lib.attention import MSAScratch, MSATorch, canonical_weights
    from e2lib.model import TinyTransformer
    from e2lib.tokenizers import TOKENIZERS, make_tokenizer
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

from common.split import split_hash, stratified_split


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class Attention(unittest.TestCase):
    def pair(self, dim=32, heads=4, seed=3):
        a, b = MSAScratch(dim, heads).double(), MSATorch(dim, heads).double()
        w = [t.double() for t in canonical_weights(dim, seed)]
        a.load_canonical(*w)
        b.load_canonical(*w)
        return a, b

    def test_same_output(self):
        a, b = self.pair()
        x = torch.randn(5, 11, 32, dtype=torch.double)
        self.assertTrue(torch.allclose(a(x), b(x), atol=1e-10))

    def test_same_attention_weights(self):
        a, b = self.pair()
        x = torch.randn(2, 7, 32, dtype=torch.double)
        _, pa = a(x, return_attn=True)
        _, pb = b(x, return_attn=True)
        self.assertEqual(pa.shape, (2, 4, 7, 7))
        self.assertTrue(torch.allclose(pa, pb, atol=1e-10))
        self.assertTrue(torch.allclose(pa.sum(-1), torch.ones(2, 4, 7, dtype=torch.double)))

    def test_same_gradients(self):
        a, b = self.pair()
        x = torch.randn(3, 9, 32, dtype=torch.double)
        a(x).square().sum().backward()
        b(x).square().sum().backward()
        self.assertTrue(torch.allclose(a.wq.weight.grad, b.mha.in_proj_weight.grad[:32], atol=1e-9))
        self.assertTrue(torch.allclose(a.wo.weight.grad, b.mha.out_proj.weight.grad, atol=1e-9))

    def test_same_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in MSAScratch(64, 4).parameters()), sum(p.numel() for p in MSATorch(64, 4).parameters()))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class Model(unittest.TestCase):
    def test_token_counts_and_shapes(self):
        expect = {"patch4": 64, "patch8": 16, "row": 32, "cnn": 64}
        x = torch.randn(2, 3, 32, 32)
        for t in TOKENIZERS:
            tok = make_tokenizer(t, 128)
            out = tok(x)
            self.assertEqual(out.shape, (2, expect[t], 128), t)
            self.assertEqual(tok.n, expect[t])

    def test_scratch_and_torch_models_start_identical(self):
        x = torch.randn(4, 3, 32, 32)
        for t in TOKENIZERS:
            a = TinyTransformer(t, "scratch", dim=64, depth=2, dropout=0.0, seed=1).eval()
            b = TinyTransformer(t, "torch", dim=64, depth=2, dropout=0.0, seed=1).eval()
            self.assertTrue(torch.allclose(a(x), b(x), atol=1e-4), t)

    def test_forward_and_attn_return(self):
        m = TinyTransformer("patch8", "scratch", dim=64, depth=2, seed=0).eval()
        logits, attns = m(torch.randn(2, 3, 32, 32), return_attn=True)
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(len(attns), 2)
        self.assertEqual(attns[0].shape, (2, 4, 17, 17))

    def test_no_pos_is_frozen_zero(self):
        m = TinyTransformer("row", "scratch", dim=64, depth=1, use_pos=False)
        self.assertEqual(float(m.pos.abs().sum()), 0.0)
        self.assertFalse(m.pos.requires_grad)


class Split(unittest.TestCase):
    def test_stratified_disjoint_and_stable(self):
        y = np.repeat(np.arange(10), 100)
        s = stratified_split(y, val_per_class=10, seed=42)
        self.assertEqual(len(s["val"]), 100)
        self.assertEqual(len(set(s["train"]) & set(s["val"])), 0)
        self.assertEqual(np.bincount(y[s["val"]]).tolist(), [10] * 10)
        self.assertEqual(split_hash(s), split_hash(stratified_split(y, 10, 42)))
        self.assertNotEqual(split_hash(s), split_hash(stratified_split(y, 10, 43)))


if __name__ == "__main__":
    unittest.main()
