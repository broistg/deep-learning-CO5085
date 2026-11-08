"""E3 training, checkpoint, and run lifecycle tests using small in-memory data."""

import argparse
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from e3lib import train as training
from e3lib.data import ImageBatches

from common.ckpt import atomic_save, build_state
from common.repro import config_hash, set_seed

torch.set_num_threads(1)


def small_data():
    rng = np.random.default_rng(7)
    def batch():
        images = rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
        labels = np.array([0, 1, 2, 3])
        return ImageBatches(images, labels, "cpu", (0.5,) * 3, (0.25,) * 3)
    metadata = {"split_sha256_16": "unit", "channel_mean": (0.5,) * 3,
                "channel_std": (0.25,) * 3}
    return batch(), batch(), batch(), metadata


def quick_eval(model, dataset, batch_size=500, max_batches=None,
               return_predictions=False):
    result = {"loss": 0.7, "accuracy": 0.5, "time_s": 0.01}
    if return_predictions:
        logits = torch.zeros(10000, 10)
        result.update(logits=logits, labels=torch.zeros(10000, dtype=torch.long),
                      accuracy=1.0)
    return result


def tiny_cfg(**kwargs):
    return training.Cfg(batch_size=2, eval_batch_size=2, epochs=1,
                        embed_dim=8, hidden_size=8, **kwargs)


class TestSteps(unittest.TestCase):

    def test_training_updates_weights_with_adam(self):
        dataset = small_data()[0]
        for model_name, tokenizer in (("lstm", "row"), ("lstm", "patch4"),
                                      ("gru", "row"), ("gru", "patch4")):
            with self.subTest(model=model_name, tokenizer=tokenizer):
                cfg = tiny_cfg(model=model_name, tokenizer=tokenizer)
                model = training.make_model(cfg)
                optimizer = training.make_optimizer(model, cfg)
                self.assertIsInstance(optimizer, torch.optim.Adam)
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0)
                self.assertEqual(optimizer.param_groups[0]["lr"], cfg.lr)
                before = {name: p.detach().clone()
                          for name, p in model.named_parameters()}
                training.train_one_epoch(model, dataset, optimizer, cfg, epoch=0)
                self.assertTrue(any(not torch.equal(p, before[name])
                                    for name, p in model.named_parameters()))
                self.assertEqual(optimizer.param_groups[0]["lr"], cfg.lr)

    def test_evaluation_does_not_update_weights_or_batchnorm(self):
        dataset = small_data()[1]
        for name in ("lstm", "gru"):
            with self.subTest(model=name):
                model = training.make_model(tiny_cfg(model=name))
                before = {key: value.clone() for key, value in model.state_dict().items()}
                result = training.evaluate(model, dataset, batch_size=2)
                self.assertTrue(0 <= result["accuracy"] <= 1)
                self.assertTrue(all(torch.equal(value, before[key])
                                    for key, value in model.state_dict().items()))

    def test_adam_has_no_weight_decay(self):
        cfg = tiny_cfg(model="lstm")
        model = training.make_model(cfg)
        optimizer = training.make_optimizer(model, cfg)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertTrue(all(group["weight_decay"] == 0 for group in optimizer.param_groups))


class TestRunLifecycle(unittest.TestCase):

    def test_cli_only_accepts_recurrent_models(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            training.validate_cfg(training.Cfg(model="minivgg"))
        options = []
        original = argparse.ArgumentParser.add_argument
        def track(parser, *names, **kwargs):
            options.extend(name for name in names if name.startswith("--")
                           and name != "--help")
            return original(parser, *names, **kwargs)
        with patch.object(argparse.ArgumentParser, "add_argument", track):
            with patch.object(training, "run", return_value={}) as run:
                training.main(["--model", "gru", "--tokenizer", "patch4",
                               "--smoke"])
        self.assertEqual(set(options), {"--model", "--tokenizer", "--seed",
                                        "--smoke", "--force"})
        self.assertEqual(len(options), 5)
        self.assertEqual(run.call_args.args[0].run_id, "gru_patch4_s0")

    def test_smoke_never_reads_or_changes_official_files(self):
        cfg = tiny_cfg(model="gru")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            files = [out / f"{cfg.run_id}.json",
                     out / "artifacts" / cfg.run_id / "ckpt" / "last.pt",
                     out / "artifacts" / cfg.run_id / "model_final.pt",
                     out / "artifacts" / cfg.run_id / "logits_test.pt"]
            for file in files:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"official sentinel")
            with patch.object(training, "make_datasets", return_value=small_data()):
                with patch.object(training, "read_json", side_effect=AssertionError("read")):
                    result = training.run(cfg, out_dir=out, smoke=True, force=True)
            self.assertTrue(result["smoke"])
            self.assertEqual(len(result["history"]), 1)
            self.assertTrue(all(file.read_bytes() == b"official sentinel"
                                for file in files))

    def test_skip_checks_all_completed_artifacts(self):
        cfg = tiny_cfg(model="gru")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            art = out / "artifacts" / cfg.run_id
            art.mkdir(parents=True)
            torch.save(training.make_model(cfg).state_dict(), art / "model_final.pt")
            torch.save({"logits": torch.zeros(10000, 10),
                        "labels": torch.zeros(10000, dtype=torch.long)},
                       art / "logits_test.pt")
            history = [{"epoch": 1, "val_acc": 0.5, "val_loss": 0.7}]
            saved = {"run_id": cfg.run_id, "config": dataclasses.asdict(cfg),
                     "config_hash": config_hash(dataclasses.asdict(cfg)),
                     "split_sha256_16": "unit",
                     "channel_mean": [0.5] * 3, "channel_std": [0.25] * 3,
                     "n_params": sum(p.numel() for p in training.make_model(cfg).parameters()),
                     "n_tokens": 32,
                     "history": history,
                     "final": {"val_acc": 0.5, "val_loss": 0.7,
                               "test_acc": 1.0, "test_loss": 0.7}}
            result_file = out / f"{cfg.run_id}.json"
            result_file.write_text(json.dumps(saved))
            with patch.object(training, "make_datasets",
                              side_effect=AssertionError("loaded CIFAR")):
                self.assertEqual(training.run(cfg, out_dir=out), saved)
                with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                    training.run(dataclasses.replace(cfg, epochs=2), out_dir=out)
                (art / "model_final.pt").write_bytes(b"corrupt")
                with self.assertRaisesRegex(RuntimeError, "corrupt artifact"):
                    training.run(cfg, out_dir=out)

    def test_checkpoint_mismatch_is_blocked_without_overwrite(self):
        cfg = tiny_cfg(model="lstm")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ckpt = out / "artifacts" / cfg.run_id / "ckpt" / "last.pt"
            state = {"extra": {"run_id": cfg.run_id, "config_hash": "other"}}
            atomic_save(state, ckpt)
            original = ckpt.read_bytes()
            with patch.object(training, "make_datasets",
                              side_effect=AssertionError("loaded CIFAR")):
                with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                    training.run(cfg, out_dir=out)
            self.assertEqual(ckpt.read_bytes(), original)

    def test_resume_restores_adam_state_and_finishes(self):
        cfg = dataclasses.replace(tiny_cfg(model="gru"), epochs=2)
        train, val, test, metadata = small_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            set_seed(cfg.seed)
            model = training.make_model(cfg)
            optimizer = training.make_optimizer(model, cfg)
            training.train_one_epoch(model, train, optimizer, cfg, epoch=0)
            history = [{"epoch": 1, "train_loss": 0.8, "train_acc": 0.5,
                        "val_loss": 0.7, "val_acc": 0.5, "lr": cfg.lr,
                        "train_s": 0.01, "val_s": 0.01, "peak_mem_mb": None}]
            extra = {"config_hash": config_hash(dataclasses.asdict(cfg)),
                     "run_id": cfg.run_id, "history": history,
                     "device_type": "cpu", "split_sha256_16": "unit",
                     "channel_mean": list(metadata["channel_mean"]),
                     "channel_std": list(metadata["channel_std"])}
            ckpt = out / "artifacts" / cfg.run_id / "ckpt" / "last.pt"
            atomic_save(build_state(model, optimizer, None, None, 1, extra), ckpt)
            saved_weight = next(model.parameters()).detach().clone()
            original_train = training.train_one_epoch
            def check_restored(model, dataset, optimizer, cfg, epoch,
                               max_batches=None):
                self.assertEqual(epoch, 1)
                self.assertTrue(torch.equal(next(model.parameters()), saved_weight))
                self.assertTrue(optimizer.state)
                self.assertEqual(optimizer.param_groups[0]["lr"], cfg.lr)
                return original_train(model, dataset, optimizer, cfg, epoch, max_batches)
            with patch.object(training, "make_datasets",
                              return_value=(train, val, test, metadata)):
                with patch.object(training, "evaluate", side_effect=quick_eval):
                    with patch.object(training, "env_info", return_value={}):
                        with patch.object(training, "train_one_epoch",
                                          side_effect=check_restored):
                            result = training.run(cfg, out_dir=out)
            self.assertEqual([row["epoch"] for row in result["history"]], [1, 2])
            restored = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertEqual(restored["epoch"], 2)
            self.assertIsNone(restored["sched"])
            self.assertIn("rng", restored)

    def test_rebuilds_missing_result_from_final_checkpoint(self):
        cfg = tiny_cfg(model="gru")
        train, val, test, metadata = small_data()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            set_seed(cfg.seed)
            model = training.make_model(cfg)
            optimizer = training.make_optimizer(model, cfg)
            training.train_one_epoch(model, train, optimizer, cfg, epoch=0)
            history = [{"epoch": 1, "train_loss": 0.8, "train_acc": 0.5,
                        "val_loss": 0.7, "val_acc": 0.5, "lr": cfg.lr,
                        "train_s": 0.01, "val_s": 0.01,
                        "peak_mem_mb": None}]
            extra = {"config_hash": config_hash(dataclasses.asdict(cfg)),
                     "run_id": cfg.run_id, "history": history,
                     "device_type": "cpu", "split_sha256_16": "unit",
                     "channel_mean": list(metadata["channel_mean"]),
                     "channel_std": list(metadata["channel_std"])}
            artifact_dir = out / "artifacts" / cfg.run_id
            atomic_save(build_state(model, optimizer, None, None, 1, extra),
                        artifact_dir / "ckpt" / "last.pt")
            weights = artifact_dir / "model_final.pt"
            weights.write_bytes(b"incomplete final save")
            with patch.object(torch.cuda, "is_available", return_value=False):
                with patch.object(training, "make_datasets",
                                  return_value=(train, val, test, metadata)):
                    with patch.object(training, "evaluate", side_effect=quick_eval):
                        with patch.object(training, "env_info", return_value={}):
                            with patch.object(training, "train_one_epoch",
                                              side_effect=AssertionError("trained again")):
                                result = training.run(cfg, out_dir=out)
            self.assertEqual(len(result["history"]), 1)
            self.assertEqual(result["config"]["epochs"], 1)
            self.assertTrue((out / f"{cfg.run_id}.json").is_file())
            restored_weights = torch.load(weights, map_location="cpu",
                                          weights_only=True)
            self.assertTrue(all(torch.equal(value, restored_weights[key])
                                for key, value in model.state_dict().items()))
            self.assertTrue((artifact_dir / "logits_test.pt").is_file())

    def test_force_resets_only_selected_run(self):
        cfg = tiny_cfg(model="gru")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            selected = out / "artifacts" / cfg.run_id
            other = out / "artifacts" / "lstm_row_s1"
            for art in (selected, other):
                art.mkdir(parents=True)
                (art / "old.bin").write_bytes(b"old")
            selected_json = out / f"{cfg.run_id}.json"
            other_json = out / "lstm_row_s1.json"
            selected_json.write_bytes(b"old")
            other_json.write_bytes(b"other")
            with patch.object(training, "make_datasets", return_value=small_data()):
                with patch.object(training, "evaluate", side_effect=quick_eval):
                    with patch.object(training, "env_info", return_value={}):
                        result = training.run(cfg, out_dir=out, force=True)
            self.assertEqual(result["history"][0]["epoch"], 1)
            self.assertFalse((selected / "old.bin").exists())
            self.assertEqual(other_json.read_bytes(), b"other")
            self.assertEqual((other / "old.bin").read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
