"""Torch-free tests:  python -m unittest discover -s A1/tests -v   (run from the repo root, after `pip install -e .`)"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from a1lib import metrics as M
from a1lib import plan as P
from a1lib.config import RunConfig, load_config, dump_config, lr_group, resolve_lr, validate
from a1lib.schedule import lr_factor


class Schedule(unittest.TestCase):
    def test_shape(self):
        T = 1000
        f = [lr_factor(s, T, 0.05, 0.01, 0.01) for s in range(T + 1)]
        self.assertAlmostEqual(f[0], 0.01)
        self.assertAlmostEqual(f[50], 1.0, places=6)          # end of warmup (5% of 1000 steps)
        self.assertAlmostEqual(f[T], 0.01, places=6)
        self.assertTrue(all(f[i] >= f[i + 1] - 1e-12 for i in range(50, T)))  # monotone decay after warmup


class Config(unittest.TestCase):
    def test_roundtrip_and_validate(self):
        c = P.sweep_run("resnet50.a1_in1k", "head", 1e-3)
        with tempfile.TemporaryDirectory() as d:
            dump_config(c, Path(d) / "x.yaml")
            self.assertEqual(load_config(Path(d) / "x.yaml").to_dict(), c.to_dict())
        with self.assertRaises(Exception):
            validate(RunConfig(run_id="a", model="m", mode="bogus", epochs=1))

    def test_resolve_lr(self):
        c = RunConfig(run_id="a", model="resnet50.a1_in1k", mode="partial", epochs=10)
        r = resolve_lr(c, {"resnet50.a1_in1k|finetune": {"lr": 1e-4}})
        self.assertEqual(float(r.lr_peak), 1e-4)
        self.assertEqual(lr_group("full"), lr_group("partial"))


class Plan(unittest.TestCase):
    def setUp(self):
        self.runs = P.build_runs()

    def test_counts(self):
        k = {}
        for r in self.runs:
            k[r.kind] = k.get(r.kind, 0) + 1
        self.assertEqual(k, {"sweep": 36, "main": 8, "seed": 12, "ablation": 12, "stretch": 9})
        self.assertEqual(len({r.run_id for r in self.runs}), len(self.runs))

    def test_assignment_balanced(self):
        plan = P.assign(self.runs, ["a", "b"])
        tab = P.summarize_plan({"runs": plan})
        tot = {o: sum(v.values()) for o, v in tab.items()}
        self.assertLess(abs(tot["a"] - tot["b"]), 2.0)
        committed = sum(e["est_hours"] for e in plan if e["kind"] != "stretch")
        self.assertTrue(20 < committed < 32, committed)

    def test_choose_lr(self):
        rows = [{"lr": 1e-4, "val_top1": .80, "val_nll": 1}, {"lr": 3e-4, "val_top1": .85, "val_nll": 1}, {"lr": 1e-3, "val_top1": .82, "val_nll": 1}]
        r = P.choose_lr(rows, [1e-4, 3e-4, 1e-3])
        self.assertEqual(r["lr"], 3e-4)
        self.assertIsNone(r["extend_to"])
        rows[2]["val_top1"] = .9                      # best at the upper edge -> extend x3
        r = P.choose_lr(rows, [1e-4, 3e-4, 1e-3])
        self.assertTrue(math.isclose(r["extend_to"], 3e-3))
        rows.append({"lr": 3e-3, "val_top1": .91, "val_nll": 1})   # still best at the edge, but already extended once -> accept
        r = P.choose_lr(rows, [1e-4, 3e-4, 1e-3])
        self.assertIsNone(r["extend_to"])
        self.assertEqual(r["lr"], 3e-3)

    def test_choose_lr_tie_break(self):
        rows = [{"lr": 1e-4, "val_top1": .85, "val_nll": 0.9}, {"lr": 3e-4, "val_top1": .85, "val_nll": 0.5}, {"lr": 1e-3, "val_top1": .7, "val_nll": 2}]
        self.assertEqual(P.choose_lr(rows, [1e-4, 3e-4, 1e-3])["lr"], 3e-4)


class Metrics(unittest.TestCase):
    def test_perfect_and_uniform(self):
        y = np.array([0, 1, 2, 1])
        perfect = np.eye(3)[y] * 20
        s = M.summarize(perfect, y, 3)
        self.assertEqual(s["top1"], 1.0)
        self.assertLess(s["nll"], 1e-6)
        u = M.summarize(np.zeros((4, 3)), y, 3)
        self.assertAlmostEqual(u["nll"], math.log(3), places=6)


if __name__ == "__main__":
    unittest.main()
