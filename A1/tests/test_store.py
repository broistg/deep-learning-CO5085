"""Torch-free tests of the store, locks, squash guard, seeding and the worker's run selection.
    python -m unittest discover -s A1/tests -v"""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from a1lib import store as S
from a1lib.seeding import sample_seed


class Seeding(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        self.assertEqual(sample_seed(0, 3, 17), sample_seed(0, 3, 17))
        seeds = {sample_seed(0, e, i) for e in range(5) for i in range(2000)}
        self.assertEqual(len(seeds), 5 * 2000)                       # no collisions on this grid
        self.assertNotEqual(sample_seed(0, 1, 1), sample_seed(1, 1, 1))
        self.assertTrue(all(0 <= s < 2**32 for s in list(seeds)[:100]))

    def test_independent_of_loader_layout(self):
        """The seed of a sample never depends on which worker/batch it lands in: it is a pure function of (seed, epoch, index)."""
        order = list(range(50))
        for workers in (0, 2, 4):
            by_worker = {i: sample_seed(7, 2, i) for i in order}
            self.assertEqual(by_worker, {i: sample_seed(7, 2, i) for i in order}, workers)


class StoreBasics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.st = S.LocalStore(Path(self.tmp.name) / "store")

    def tearDown(self):
        self.tmp.cleanup()

    def test_put_get_delete_list(self):
        f = Path(self.tmp.name) / "x.bin"
        f.write_bytes(b"abc")
        self.st.put({"runs/r/a.json": '{"k": 1}', "runs/r/last.pt": f, "tmp/z.json": b"{}"})
        self.assertEqual(self.st.read_json("runs/r/a.json"), {"k": 1})
        self.assertEqual(dict(self.st.list_files("runs/")), {"runs/r/a.json": 8, "runs/r/last.pt": 3})
        self.st.get("runs/r/last.pt", Path(self.tmp.name) / "d" / "l.pt")
        self.assertEqual((Path(self.tmp.name) / "d" / "l.pt").read_bytes(), b"abc")
        self.st.put({"tmp/z.json": None})
        self.assertFalse(self.st.exists("tmp/z.json"))
        self.assertEqual(self.st.pushed_commits, 2)
        self.assertEqual(self.st.pushed_bytes, 8 + 3 + 2)

    def test_conditional_put(self):
        h = self.st.head()
        self.st.put({"a.json": "1"}, parent=h)
        with self.assertRaises(S.StoreConflict):
            self.st.put({"a.json": "2"}, parent=h)                   # head moved
        self.assertEqual(self.st.read_json("a.json"), 1)


class Locks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.st = S.LocalStore(Path(self.tmp.name) / "store")

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_owner_two_workers_exclude_each_other(self):
        a = S.RunLock(self.st, "r1", "dat", "dat-s1-g0")
        b = S.RunLock(self.st, "r1", "dat", "dat-s1-g1")
        a.acquire()
        with self.assertRaises(RuntimeError):
            b.acquire()
        a.acquire()                                                  # the holder may re-acquire (restart)
        a.release()
        b.acquire()                                                  # free now
        self.assertTrue(b.still_mine() and not a.still_mine())

    def test_stale_lock_is_taken_over_and_force(self):
        self.st.put({"runs/r2/run.lock.json": json.dumps({"owner": "ly", "worker": "ly-s9-g0", "heartbeat": time.time() - S.STALE_S - 5})})
        b = S.RunLock(self.st, "r2", "dat", "dat-s1-g0")
        b.acquire()                                                  # stale -> allowed
        c = S.RunLock(self.st, "r2", "ly", "ly-s2-g0")
        with self.assertRaises(RuntimeError):
            c.acquire()
        c.acquire(force=True)
        self.assertTrue(c.still_mine() and not b.still_mine())

    def test_race_has_one_winner(self):
        wins, errs = [], []

        def go(i):
            lk = S.RunLock(self.st, "r3", "dat", f"dat-s1-g{i}")
            try:
                lk.acquire()
                wins.append(i)
            except RuntimeError as e:
                errs.append(str(e))
        ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(len(wins), 1, (wins, errs))
        self.assertEqual(len(errs), 7)


class Sync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.st = S.LocalStore(self.root / "store")

    def tearDown(self):
        self.tmp.cleanup()

    def _sync(self, name, worker, push_every_s=0.0):
        d = self.root / name
        (d / "ckpt").mkdir(parents=True)
        return S.RunSync(self.st, "r1", "dat", worker, d, push_every_s=push_every_s), d

    def test_beat_and_ckpt_handover_between_sessions(self):
        s1, d1 = self._sync("s1", "dat-s1-g0")
        s1.lock.acquire()
        (d1 / "status.json").write_text('{"state": "running", "epoch": 2}')
        (d1 / "ckpt" / "last.pt").write_bytes(b"epoch-2")
        s1.beat()
        s1.push_ckpt(2)
        self.assertEqual(self.st.read_json("runs/r1/status.json")["epoch"], 2)
        self.assertEqual(s1.remote_epoch(), 2)
        s1.lock.release()
        s2, d2 = self._sync("s2", "dat-s2-g0")                       # a NEW session, empty local disk
        s2.lock.acquire()
        self.assertEqual(s2.pull_ckpt(0), 2)
        self.assertEqual((d2 / "ckpt" / "last.pt").read_bytes(), b"epoch-2")
        self.assertIsNone(s2.pull_ckpt(2))                           # local already as new as the store

    def test_beat_detects_takeover(self):
        s1, d1 = self._sync("s1", "dat-s1-g0")
        s1.lock.acquire()
        S.RunLock(self.st, "r1", "ly", "ly-s1-g0").acquire(force=True)
        with self.assertRaises(RuntimeError):
            s1.beat()
        self.assertTrue(s1.lock.lost)

    def test_push_interval(self):
        s1, _ = self._sync("s1", "dat-s1-g0", push_every_s=3600)
        self.assertFalse(s1.should_push_ckpt())
        s1.last_ckpt_push -= 3601
        self.assertTrue(s1.should_push_ckpt())

    def test_final_marks_done(self):
        s1, d1 = self._sync("s1", "dat-s1-g0")
        f = d1 / "m.pt"
        f.write_bytes(b"w")
        s1.push_final({"model_final.pt": f}, '{"run_id": "r1"}')
        self.assertEqual(self.st.read_json("runs/r1/result.json")["run_id"], "r1")
        self.assertTrue(self.st.exists("runs/r1/model_final.pt"))


class Maintenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.st = S.LocalStore(Path(self.tmp.name) / "store")
        self.st.put({"runs/r1/result.json": "{}", "stats/w1.json": json.dumps({"worker": "w1", "bytes_pushed": 5e9})})

    def tearDown(self):
        self.tmp.cleanup()

    def test_squash_refused_while_a_worker_is_live(self):
        S.RunLock(self.st, "r1", "dat", "dat-s1-g0").acquire()
        with self.assertRaises(RuntimeError) as cm:
            S.squash(self.st, "dat")
        self.assertIn("refusing", str(cm.exception))
        self.assertFalse(self.st.exists("maintenance.log"))

    def test_squash_logs_and_resets_baseline(self):
        s0 = S.stats(self.st)
        self.assertAlmostEqual(s0["pushed_since_squash"], 5e9)
        S.squash(self.st, "dat")
        self.assertIn("verify OK", (self.st.mirror / "maintenance.log").read_text())
        self.assertAlmostEqual(S.stats(self.st)["pushed_since_squash"], 0)
        self.st.put({"stats/w1.json": json.dumps({"worker": "w1", "bytes_pushed": 7e9})})
        self.assertAlmostEqual(S.stats(self.st)["pushed_since_squash"], 2e9)

    def test_force_overrides_and_is_logged(self):
        S.RunLock(self.st, "r1", "dat", "dat-s1-g0").acquire()
        S.squash(self.st, "dat", force=True)
        self.assertIn("FORCED", (self.st.mirror / "maintenance.log").read_text())

    def test_verification_catches_a_changed_store(self):
        orig = self.st.squash_history
        self.st.squash_history = lambda: self.st.put({"runs/r1/result.json": None})   # simulate a squash that lost a file
        with self.assertRaises(RuntimeError):
            S.squash(self.st, "dat")
        self.assertIn("MISMATCH", (self.st.mirror / "maintenance.log").read_text())
        self.st.squash_history = orig

    def test_warning_threshold(self):
        self.st.put({"stats/w2.json": json.dumps({"worker": "w2", "bytes_pushed": 40e9})})
        self.assertTrue(S.stats(self.st)["warn"])


class WorkerSelection(unittest.TestCase):
    def test_runnable_skips_runs_held_by_other_workers(self):
        from a1lib.paths import Paths
        from a1lib.worker import runnable
        with tempfile.TemporaryDirectory() as d:
            st = S.LocalStore(Path(d) / "store")
            paths = Paths(st, Path(d) / "local")
            plan = {"_a1": d, "runs": [{"run_id": f"r{i}", "owner": "dat", "needs_lr": False} for i in range(3)]}
            S.RunLock(st, "r0", "dat", "dat-s1-g0").acquire()
            st.put({"runs/r1/result.json": "{}"})
            ready, blocked = runnable(plan, paths, "dat", me="dat-s1-g1")
            self.assertEqual([e["run_id"] for e in ready], ["r2"])
            ready, _ = runnable(plan, paths, "dat", me="dat-s1-g0")      # my own live lock does not hide the run (restart)
            self.assertEqual([e["run_id"] for e in ready], ["r0", "r2"])
            self.assertEqual(S.worker_id("dat", 1).split("-")[-1], "g1")


if __name__ == "__main__":
    unittest.main()


class DataCopy(unittest.TestCase):
    def test_find_and_copy_unpacked_folder(self):
        import os
        from a1lib import data as D
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "input" / "ds" / "food101_256"
            for c in ("a", "b"):
                (src / "images" / c).mkdir(parents=True)
                for i in range(3):
                    (src / "images" / c / f"{i}.jpg").write_bytes(b"x" * (i + 1))
            (src / "split.json").write_text("{}")
            os.environ["A1_DATA_SRC"] = str(d / "input")
            try:
                self.assertEqual(D._find_source(), src)
                dst = d / "local" / "food101_256"
                dst.mkdir(parents=True)
                D._copy_tree(src, dst, threads=4)
            finally:
                del os.environ["A1_DATA_SRC"]
            self.assertEqual(D.count_images(dst / "images"), 6)
            self.assertTrue((dst / "split.json").exists())
