"""The A1 analysis on SYNTHETIC results (no torch, no real data): checks that every table/figure is produced, the numbers equal an
independent computation, and the output is deterministic (what the replication notebook relies on)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from a1lib import analyze as AN
from a1lib import metrics as M
from a1lib import plan as P

N_VAL, N_TEST = 505, 1010


def fake_logits(rng, n, acc, n_classes=101):
    y = rng.integers(0, n_classes, n)
    z = rng.normal(0, 1, (n, n_classes)).astype(np.float32)
    hit = rng.random(n) < acc
    z[np.arange(n), y] += np.where(hit, 4.0, 1.0)
    return z, y


def make_results(d, art):
    rng = np.random.default_rng(0)
    base = {"head": 0.70, "partial": 0.80, "full": 0.85, "scratch": 0.45}
    for r in P.build_runs():
        c = r.to_dict()
        c["lr_peak"] = 1e-3 if c["lr_peak"] == "auto" else c["lr_peak"]
        acc = base[c["mode"]] * (0.97 if "deit" in c["model"] else 1.0) * (0.5 + 0.5 * c["data_fraction"]) + 0.002 * c["seed"]
        zv, yv = fake_logits(rng, N_VAL, acc)
        zt, yt = fake_logits(rng, N_TEST, acc)
        fin = {"val": M.summarize(zv, yv, 101)}
        if c["kind"] != "sweep":
            fin["test"] = M.summarize(zt, yt, 101)
            (art / r.run_id).mkdir(parents=True)
            np.savez(art / r.run_id / "logits_val.npz", logits=zv.astype(np.float16), labels=yv)
            np.savez(art / r.run_id / "logits_test.npz", logits=zt.astype(np.float16), labels=yt)
        hist = [{"epoch": e + 1, "val_top1": acc * (e + 1) / c["epochs"], "train_s": 100.0, "val_s": 10.0} for e in range(c["epochs"])]
        res = {"run_id": r.run_id, "kind": c["kind"], "config": c, "final": fin, "history": hist, "trainable_M": 1.0, "trainable_pct": 12.5,
               "timing": {"train_s_total": 100.0 * c["epochs"], "val_s_total": 10.0 * c["epochs"]}, "peak_mem_mb_max": 1234, "smoke": False}
        (d / f"{r.run_id}.json").write_text(json.dumps(res))
    (d / "lr_choice.json").write_text(json.dumps({"resnet50.a1_in1k|head": {"lr": 1e-3}}))


class Analyze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.res, cls.art, cls.out, cls.out2 = t / "results", t / "artifacts", t / "out", t / "out2"
        cls.res.mkdir()
        make_results(cls.res, cls.art)
        AN.main(["--results", str(cls.res), "--artifacts", str(cls.art), "--out", str(cls.out)])
        AN.main(["--results", str(cls.res), "--artifacts", str(cls.art), "--out", str(cls.out2)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_exist(self):
        names = {p.name for p in self.out.glob("tab_*.tex")}
        for n in ("tab_acc", "tab_acc_ext", "tab_seeds", "tab_traincost", "tab_freeze", "tab_ablation", "tab_lr", "tab_stretch",
                  "tab_calibration", "tab_ensemble", "tab_confpairs", "tab_hardclasses"):
            self.assertIn(n + ".tex", names)
        for f in ("curves", "ablation", "pareto", "lr_sweep", "reliability", "perclass"):
            self.assertTrue((self.out / "figures" / f"{f}.png").exists(), f)
        s = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(s["kinds"], {"sweep": 36, "main": 8, "seed": 12, "ablation": 12, "stretch": 9})

    def test_no_trailing_backslashes_and_column_count(self):
        for p in self.out.glob("tab_*.tex"):
            lines = p.read_text().strip().splitlines()
            self.assertFalse(lines[-1].rstrip().endswith("\\\\"), p.name)
        row = (self.out / "tab_acc.tex").read_text().splitlines()[0]
        self.assertEqual(row.count("&"), 6)                       # 7 columns as in the report's table
        self.assertEqual((self.out / "tab_acc.tex").read_text().count("\n"), 8)

    def test_numbers_match_independent_computation(self):
        s = json.loads((self.out / "summary.json").read_text())
        v = [json.loads((self.res / f"m_resnet50_full_s{k}.json").read_text())["final"]["test"]["top1"] for k in (0, 1, 2)]
        self.assertEqual(s["main"]["ResNet-50|full"]["top1"], v)
        row = [l for l in (self.out / "tab_acc.tex").read_text().splitlines() if "Full fine-tuning" in l][0]
        self.assertIn(AN.pm(v), row)
        z, y = AN.load_logits(self.art, "m_deit_s_full_s0", "test")
        self.assertAlmostEqual(s["main"]["DeiT-S|full"]["top1"][0], float((z.argmax(1) == y).mean()), places=2)  # fp16 round-trip tolerance

    def test_deterministic(self):
        for p in sorted(self.out.glob("tab_*.tex")):
            self.assertEqual(p.read_text(), (self.out2 / p.name).read_text(), p.name)
        self.assertEqual((self.out / "summary.json").read_text(), (self.out2 / "summary.json").read_text())

    def test_temperature_scaling_reduces_nll(self):
        rng = np.random.default_rng(1)
        z, y = fake_logits(rng, 4000, 0.6)
        z = z * 3.0                                              # over-confident
        T = AN.fit_temperature(z, y)
        self.assertGreater(T, 1.5)
        self.assertLess(M.nll(z / T, y), M.nll(z, y))

    def test_ablation_has_all_points(self):
        s = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(set(s["ablation"]["ResNet-50|full"]), {"0.1", "0.25", "0.5", "1.0"})

    def test_works_without_logits(self):
        o = Path(self.tmp.name) / "nolog"
        AN.main(["--results", str(self.res), "--artifacts", str(Path(self.tmp.name) / "nope"), "--out", str(o)])
        self.assertTrue((o / "tab_acc.tex").exists())
        self.assertFalse((o / "tab_calibration.tex").exists())


if __name__ == "__main__":
    unittest.main()
