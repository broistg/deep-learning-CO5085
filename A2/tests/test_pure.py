"""Torch-free tests:  PYTHONPATH=.:A1:A2 python -m unittest discover -s A2/tests -v   (from the repo root)"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from a2lib import data as D
from a2lib import metrics as M
from a2lib import plan as P
from a2lib.config import load_config, dump_config, make_config, resolve_lr, validate
from a2lib.schedule import lr_factor


class Schedule(unittest.TestCase):
    def test_segformer(self):
        f = lambda s: lr_factor(s, 24000, 1.0, 1.0, 0.0, 0.0625, 1e-6)
        self.assertAlmostEqual(f(0), 1e-6)
        self.assertAlmostEqual(f(1500), 0.9375, places=3)
        self.assertAlmostEqual(f(12000), 0.5, places=3)
        self.assertLess(f(23999), 1e-3)

    def test_deeplab_min_lr(self):
        v = lr_factor(23999, 24000, 0.01, 0.9, 1e-4, 0.0)
        self.assertGreaterEqual(v * 0.01, 1e-4 - 1e-9)


class Metrics(unittest.TestCase):
    def test_perfect_and_ignore(self):
        conf = np.zeros((3, 3), dtype=np.int64)
        label = np.array([[0, 1], [2, 255]])
        M.update_confusion(conf, label.copy(), label, 3)
        s = M.summarize(conf)
        self.assertAlmostEqual(s["mIoU"], 1.0)
        self.assertEqual(s["n_pixels"], 3)

    def test_half(self):
        conf = np.zeros((2, 2), dtype=np.int64)
        M.update_confusion(conf, np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
        s = M.summarize(conf)
        self.assertAlmostEqual(s["pixel_acc"], 0.75)
        self.assertAlmostEqual(s["mIoU"], (0.5 + 2 / 3) / 2)


class Split(unittest.TestCase):
    def test_sizes_disjoint_deterministic(self):
        s = D.make_split()
        self.assertEqual((len(s["train"]), len(s["dev"]), len(s["test"])), (19210, 1000, 2000))
        self.assertFalse(set(s["train"]) & set(s["dev"]))
        self.assertEqual(D.make_split()["train"][:5], s["train"][:5])

    def test_committed_split_matches(self):
        p = Path(__file__).resolve().parents[1] / "split.json"
        self.assertEqual(D.load_split(p)["sha256_16"], D.split_hash(D.make_split()))


class Plan(unittest.TestCase):
    def test_runs(self):
        runs = P.build_runs("x")
        ids = [c.run_id for c, _ in runs]
        self.assertEqual(len(ids), 20)
        self.assertEqual(len(set(ids)), 20)
        self.assertEqual(sum(c.kind == "sweep" for c, _ in runs), 12)
        for c, _ in runs:
            validate(c)

    def test_assign_balanced(self):
        plan = P.assign(P.build_runs(""), ["dat", "ly"])
        t = P.summarize_plan({"runs": plan})
        a, b = (sum(v.values()) for v in t.values())
        self.assertLess(abs(a - b), 1.0)

    def test_config_roundtrip_and_lr(self):
        c = P.main_run("segformer_b0", "pretrained", 0, "x")
        with tempfile.TemporaryDirectory() as d:
            dump_config(c, Path(d) / "c.yaml")
            c2 = load_config(Path(d) / "c.yaml")
        self.assertEqual(c.run_id, c2.run_id)
        self.assertEqual(resolve_lr(c2, {"segformer_b0|pretrained": {"lr": 6e-5}}).lr_peak, 6e-5)
        with self.assertRaises(LookupError):
            resolve_lr(c2, {})

    def test_committed_configs_exist(self):
        plan = P.load_plan(Path(__file__).resolve().parents[1] / "plan.json")
        for e in plan["runs"]:
            self.assertTrue((Path(__file__).resolve().parents[1] / e["config"]).exists())


class Data(unittest.TestCase):
    def test_fake_dataset_deterministic(self):
        from a2lib.config import default_augmentation
        with tempfile.TemporaryDirectory() as d:
            names = D.make_fake_ade(d, n_train=4, n_val=2)
            ds = D.ADEDataset(d, names["train"], "training", default_augmentation(), True, 0)
            a, b = ds[1], ds[1]
            for x, y in zip(a, b):
                self.assertTrue(np.array_equal(x, y))
            self.assertEqual(a[0].shape[:2], (512, 512))
            ds.set_epoch(1)
            self.assertFalse(np.array_equal(ds[1][0], a[0]))


if __name__ == "__main__":
    unittest.main()
