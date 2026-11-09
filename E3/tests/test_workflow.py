"""Checks for the E3 runner, result analysis, diagnostics, and notebook."""

import dataclasses
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from common.repro import config_hash
from e3lib import analyze, diagnostics, plan
from e3lib.train import E3_DIR


def result_for(cfg, val_acc, test_acc):
    config = dataclasses.asdict(cfg)
    return {
        "run_id": cfg.run_id,
        "config": config,
        "config_hash": config_hash(config),
        "split_sha256_16": "same-split",
        "channel_mean": [0.5, 0.5, 0.5],
        "channel_std": [0.25, 0.25, 0.25],
        "n_params": 40_138,
        "n_tokens": 32,
        "history": [{"epoch": epoch, "train_loss": 2.0 - epoch / 100,
                     "val_acc": val_acc, "val_loss": 0.8}
                    for epoch in range(1, cfg.epochs + 1)],
        "final": {"val_acc": val_acc, "val_loss": 0.8,
                  "test_acc": test_acc, "test_loss": 0.9},
        "train_s_per_epoch_median": 2.0,
        "smoke": False,
    }


class TestWorkflow(unittest.TestCase):

    def test_plan_has_twelve_recurrent_runs(self):
        configs = plan.all_cfgs()
        self.assertEqual(len(configs), 12)
        self.assertEqual(len({cfg.run_id for cfg in configs}), 12)
        self.assertEqual({cfg.seed for cfg in configs}, {0, 1, 2})
        self.assertEqual({cfg.model for cfg in configs}, {"lstm", "gru"})
        self.assertEqual({cfg.epochs for cfg in configs}, {30})
        self.assertEqual({cfg.embed_dim for cfg in configs}, {64})
        self.assertEqual({cfg.hidden_size for cfg in configs}, {128})

    def test_hidden64_results_require_explicit_legacy_selection(self):
        cfg = next(c for c in plan.all_cfgs(hidden_size=64)
                   if c.run_id == "gru_patch4_s0")
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / f"{cfg.run_id}.json").write_text(
                json.dumps(result_for(cfg, 0.6, 0.6)))
            with self.assertRaisesRegex(ValueError, "different config"):
                analyze.load_results(results)
            self.assertEqual(len(analyze.load_results(results, hidden_size=64)), 1)

    def test_analysis_uses_only_recurrent_runs(self):
        configs = {cfg.run_id: cfg for cfg in plan.all_cfgs()}
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            output = Path(tmp) / "output"
            results.mkdir()
            pairs = [("lstm_row_s0", 0.6, 0.62),
                     ("gru_row_s0", 0.55, 0.58),
                     ("lstm_row_s1", 0.61, 0.64)]
            for run_id, val, test in pairs:
                (results / f"{run_id}.json").write_text(
                    json.dumps(result_for(configs[run_id], val, test)))
            analyze.main(["--results", str(results), "--out", str(output),
                          "--no-plots"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["n_runs"], 3)
            self.assertNotIn("n_baseline_runs", summary)
            self.assertEqual(len(summary["missing_run_ids"]), 9)
            self.assertEqual(set(summary["groups"]), {"lstm_row", "gru_row"})
            self.assertEqual(summary["groups"]["lstm_row"]["source"], "E3")
            self.assertNotIn("paired_comparisons", summary)
            self.assertEqual(summary["groups"]["lstm_row"]["seeds"], [0, 1])
            table = (output / "tab_e3_summary.tex").read_text()
            self.assertIn("LSTM / row", table)
            self.assertNotIn("MLP", table)
            self.assertNotIn("MiniVGG", table)
            self.assertTrue(table.splitlines()[0].endswith("\\\\"))
            self.assertFalse((output / "tab_e3_pairs.tex").exists())

            changed = result_for(configs["gru_row_s0"], 0.55, 0.58)
            changed["channel_mean"][0] = 0.4
            (results / "gru_row_s0.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "different CIFAR splits"):
                analyze.load_results(results)

            (results / "gru_row_s0.json").write_text(
                json.dumps(result_for(configs["gru_row_s0"], 0.55, 0.58)))

    def test_peak_gpu_is_mean_of_three_seeds_only(self):
        configs = {cfg.run_id: cfg for cfg in plan.all_cfgs()}
        runs = []
        for seed, peak in enumerate((100.0, 200.0, 300.0)):
            run = result_for(configs[f"lstm_row_s{seed}"], 0.6, 0.62)
            run["peak_mem_mb"] = peak
            runs.append(run)
        self.assertIsNone(analyze.summarize(runs[:2])["groups"]["lstm_row"]["peak_mem_mb_mean"])
        summary = analyze.summarize(runs)
        self.assertEqual(summary["groups"]["lstm_row"]["peak_mem_mb_mean"], 200.0)
        self.assertIn("200{,}0", analyze.tables(summary)["tab_e3_summary.tex"])

    def test_analysis_requires_e3_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            out = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "No completed E3 runs"):
                analyze.main(["--results", str(results), "--out", str(out),
                              "--no-plots"])

    def test_diagnostics_from_saved_logits(self):
        cfg = next(c for c in plan.all_cfgs() if c.run_id == "lstm_row_s0")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            artifacts = root / "artifacts"
            results.mkdir()
            logits = torch.zeros(10000, 10)
            labels = torch.zeros(10000, dtype=torch.long)
            logits[:100, 1] = 3  # 100 mistakes: true class 0 -> class 1
            run = result_for(cfg, val_acc=0.7, test_acc=0.99)
            (results / f"{cfg.run_id}.json").write_text(json.dumps(run))
            path = artifacts / cfg.run_id / "logits_test.pt"
            path.parent.mkdir(parents=True)
            torch.save({"logits": logits, "labels": labels}, path)
            diagnostics.main(["--results", str(results), "--artifacts",
                              str(artifacts), "--out", str(root / "out"),
                              "--no-plots"])
            report = json.loads((root / "out" / "diagnostics.json").read_text())
            self.assertEqual(report["run_id"], cfg.run_id)
            self.assertEqual(report["confusion_matrix"][0][1], 100)
            self.assertAlmostEqual(report["per_class_acc"]["airplane"], 0.99)

    def test_diagnostics_selects_highest_test_accuracy(self):
        configs = {cfg.run_id: cfg for cfg in plan.all_cfgs()}
        runs = [result_for(configs["gru_patch4_s0"], 0.70, 0.68),
                result_for(configs["gru_patch4_s2"], 0.69, 0.69)]
        self.assertEqual(diagnostics.choose_run(runs)["run_id"], "gru_patch4_s2")

    def test_notebook_has_load_and_rerun_modes(self):
        notebook = json.loads((E3_DIR / "notebooks" / "E3.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code = ["".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code"]
        self.assertIn('MODE = "load"', code[0])
        self.assertIn('assert MODE in {"load", "rerun"}', code[0])
        self.assertTrue(any('analyze.main(["--results", str(published)' in cell
                            for cell in code))
        self.assertTrue(any('nu.compare_outputs(regenerated, published)' in cell
                            for cell in code))
        self.assertTrue(any('run(cfg, out_dir=rerun_dir)' in cell for cell in code))
        for cell in code:
            compile(cell, "E3.ipynb", "exec")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"),
                         "matplotlib not installed")
    def test_analysis_and_diagnostics_export_figures(self):
        cfg = next(c for c in plan.all_cfgs() if c.run_id == "lstm_row_s0")
        run = result_for(cfg, 0.7, 0.8)
        with tempfile.TemporaryDirectory() as tmp:
            figures = Path(tmp) / "figures"
            analyze.plot_curves([run], figures)
            self.assertTrue((figures / "curves_e3.png").is_file())
            self.assertTrue((figures / "curves_e3.pdf").is_file())
            matrix = [[0] * 10 for _ in range(10)]
            matrix[0][0] = 100
            diagnostics.plot_confusion(matrix, figures, run["config"], 0.8)
            self.assertTrue((figures / "confusion_e3.png").is_file())
            self.assertTrue((figures / "confusion_e3.pdf").is_file())
            images = np.zeros((12, 32, 32, 3), dtype=np.uint8)
            logits = torch.zeros(12, 10)
            labels = torch.zeros(12, dtype=torch.long)
            diagnostics.plot_error_grid(images, logits, labels, range(12),
                                        "gru_patch4_s2", 0.6861, figures)
            self.assertTrue((figures / "error_grid_gru_patch4.png").is_file())
            self.assertTrue((figures / "error_grid_gru_patch4.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
