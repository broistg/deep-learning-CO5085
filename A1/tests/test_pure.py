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


class FollowUp(unittest.TestCase):
    def test_llrd_scale(self):
        from a1lib.schedule import llrd_scale
        self.assertEqual(llrd_scale("head.weight", 12, 0.75), 1.0)
        self.assertEqual(llrd_scale("norm.bias", 12, 0.75), 1.0)
        self.assertAlmostEqual(llrd_scale("blocks.11.attn.qkv.weight", 12, 0.75), 0.75)
        self.assertAlmostEqual(llrd_scale("blocks.0.mlp.fc1.weight", 12, 0.75), 0.75 ** 12)
        self.assertAlmostEqual(llrd_scale("patch_embed.proj.weight", 12, 0.75), 0.75 ** 13)
        self.assertEqual(llrd_scale("cls_token", 12, 1.0), 1.0)

    def test_hash_unchanged_by_llrd_default(self):
        from a1lib.config import stable_hash
        cfg = P._cfg("x", P.PAIR1[0], "full", 10, "main", lr=1e-3).to_dict()
        old = {k: v for k, v in cfg.items() if k != "llrd"}          # a result written before the field existed
        self.assertEqual(stable_hash(cfg), stable_hash(old))
        self.assertNotEqual(stable_hash(cfg), stable_hash({**cfg, "llrd": 0.75}))

    def test_diag_runs(self):
        runs = P.build_diag_runs()
        ids = [r.run_id for r in runs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(r.kind == "diag_sweep" for r in runs), 10)
        self.assertEqual(sum(r.kind == "diag" for r in runs), 5)
        by = {r.run_id: r for r in runs}
        self.assertEqual(by["dg_resnet50_full_nomix"].augmentation["mixup"]["prob"], 0.0)
        self.assertEqual(by["dg_resnet50_full_nomix"].augmentation["mixup"]["label_smoothing"], 0.1)
        self.assertEqual(by["dg_deit_s_full_plain"].augmentation["mixup"]["label_smoothing"], 0.0)
        self.assertEqual(by["dg_deit_s_full_llrd75"].llrd, 0.75)
        self.assertTrue(all(r.lr_peak != "auto" for r in runs if r.kind == "diag_sweep"))

    def test_append_to_plan_is_idempotent_and_leaves_old_entries(self):
        with tempfile.TemporaryDirectory() as d:
            a1 = Path(d)
            old = P.build_runs()
            plan = {"version": 1, "owners": ["dat", "ly"], "grids": P.LR_GRID, "runs": P.assign(old, ["dat", "ly"])}
            before = list(plan["runs"])
            new1 = P._append_to_plan(plan, P.build_diag_runs(), a1)
            new2 = P._append_to_plan(plan, P.build_diag_runs(), a1)
            self.assertEqual(len(new1), 15)
            self.assertEqual(new2, [])
            self.assertTrue(all(e in plan["runs"] for e in before))
            self.assertEqual(len(plan["runs"]), len(before) + 15)
            self.assertTrue((a1 / "configs" / "dg_deit_s_full_llrd75.yaml").exists())
            self.assertEqual(load_config(a1 / "configs" / "dg_deit_s_full_llrd75.yaml").llrd, 0.75)
