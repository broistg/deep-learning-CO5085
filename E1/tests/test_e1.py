"""Run from the repo root:  PYTHONPATH=.:E1 python -m unittest discover -s E1/tests -v
The summary test needs only numpy; the others are skipped when torch is not installed."""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
    from common.cifar import channel_stats, get_splits
    from e1lib.data import CifarArrays, make_loaders
    from e1lib.models import MODELS
    from e1lib.run import Cfg, run
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

from e1lib import summary
from common.repro import config_hash


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class Models(unittest.TestCase):
    def test_parameter_counts(self):
        # softmax: 3072*10 + 10 | mlp 3072-64-32-10 | minivgg: conv blocks (3->16->16, 16->32->32) with BatchNorm + Linear(2048 -> 10)
        expect = {"softmax": 30_730, "mlp": 199_082, "minivgg": 37_338}
        for name, n in expect.items():
            self.assertEqual(sum(p.numel() for p in MODELS[name]().parameters()), n, name)

    def test_forward_shapes(self):
        x = torch.randn(4, 3, 32, 32)
        for name, make in MODELS.items():
            self.assertEqual(make().eval()(x).shape, (4, 10), name)

    def test_models_live_on_cpu_until_moved(self):
        for make in MODELS.values():
            self.assertTrue(all(p.device.type == "cpu" for p in make().parameters()))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class Data(unittest.TestCase):
    def setUp(self):
        self.splits, _ = get_splits(fake=True)
        self.mean, self.std = channel_stats(self.splits["train"][0])

    def test_only_train_augments(self):
        train, val, test, d = make_loaders(self.splits, self.mean, self.std, batch_size=16, workers=0)
        self.assertIsNotNone(d["train"].aug)
        self.assertIsNone(d["val"].aug)
        self.assertIsNone(d["test"].aug)
        self.assertIsNone(make_loaders(self.splits, self.mean, self.std, 16, 0, augment=False)[3]["train"].aug)

    def test_eval_images_are_deterministic_and_train_images_vary(self):
        d = CifarArrays(*self.splits["val"], self.mean, self.std, train=False)
        self.assertTrue(torch.equal(d[3][0], d[3][0]))
        t = CifarArrays(*self.splits["train"], self.mean, self.std, train=True)
        draws = [t[3][0] for _ in range(20)]
        self.assertFalse(all(torch.equal(draws[0], x) for x in draws[1:]))
        self.assertEqual(draws[0].shape, (3, 32, 32))

    def test_normalisation_uses_train_statistics(self):
        d = CifarArrays(*self.splits["train"], self.mean, self.std, train=False)
        x = torch.stack([d[i][0] for i in range(len(d))])
        self.assertTrue(torch.allclose(x.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-3))
        self.assertTrue(torch.allclose(x.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-2))

    def test_same_seed_same_batches(self):
        torch.manual_seed(0)                      # with workers=0 the augmentation draws from the global RNG (run() calls set_seed first)
        a = next(iter(make_loaders(self.splits, self.mean, self.std, 16, 0, seed=5)[0]))
        torch.manual_seed(0)
        b = next(iter(make_loaders(self.splits, self.mean, self.std, 16, 0, seed=5)[0]))
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(torch.equal(a[0], b[0]))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class EndToEnd(unittest.TestCase):
    def test_smoke_run_on_fake_data(self):
        for name in MODELS:
            r = run(Cfg(model=name, epochs=1), fake=True, smoke=True)
            self.assertTrue(0.0 <= r["final"]["test_acc"] <= 1.0)
            self.assertEqual(len(r["history"]), 1)
            self.assertEqual(r["split_sha256_16"], "fake")

    def test_cached_run_requires_matching_configuration_and_predictions(self):
        cfg = Cfg(model="mlp", seed=0)
        saved = {"run_id": cfg.run_id, "config": dataclasses.asdict(cfg),
                 "config_hash": config_hash(dataclasses.asdict(cfg)), "fake": False, "smoke": False}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = out / f"{cfg.run_id}.json"
            preds = out / f"{cfg.run_id}_preds.npz"
            result.write_text(json.dumps(saved), encoding="utf8")
            preds.touch()
            with patch("e1lib.run.get_splits", side_effect=AssertionError("cache should skip data loading")):
                self.assertEqual(run(cfg, out_dir=out), saved)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    run(Cfg(model="mlp", seed=0, epochs=10), out_dir=out)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    run(Cfg(model="mlp", seed=0, workers=2), out_dir=out)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    run(cfg, out_dir=out, fake=True)
                preds.unlink()
                with self.assertRaisesRegex(ValueError, "predictions are missing"):
                    run(cfg, out_dir=out)


class Summary(unittest.TestCase):
    def test_mean_std_over_seeds_and_table(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for seed, (fin, best) in enumerate([(0.70, 0.71), (0.72, 0.72), (0.74, 0.73)]):
                r = {"config": {"model": "mlp", "seed": seed, "augment": True}, "n_params": 199_082, "smoke": False, "fake": False,
                     "final": {"test_acc": fin}, "best": {"test_acc": best}, "train_s_per_epoch_median": 2.0,
                     "inference_ms_per_image": 0.01, "split_sha256_16": "abc"}
                (td / f"mlp_s{seed}.json").write_text(json.dumps(r), encoding="utf8")
            summary.main(td)
            s = json.loads((td / "summary.json").read_text(encoding="utf8"))["mlp"]
            self.assertAlmostEqual(s["test_acc_final_mean_std"][0], 0.72)
            self.assertAlmostEqual(s["test_acc_final_mean_std"][1], 0.02)            # sample std, ddof = 1
            tab = (td / "tab_e1_summary.tex").read_text(encoding="utf8")
            self.assertIn("199\\,082", tab)
            self.assertIn("72{,}00", tab)


if __name__ == "__main__":
    unittest.main()
