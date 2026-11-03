"""The commands a team member runs on Kaggle (A2). Same behaviour as a1lib.worker: works through the run list in plan order, resumes interrupted
runs, one worker process per GPU, per-run locks in the shared store.

    python -m a2lib.worker launch --owner dat --gpus 2 --max-hours 8.5    # one worker per GPU in this session (the normal way)
    python -m a2lib.worker run --owner dat --max-hours 8.5                # a single worker on the visible GPU
    python -m a2lib.worker status                                         # table of every run (reads the store)
    python -m a2lib.worker list --owner dat                               # what would run next
    python -m a2lib.worker prepare                                        # download + verify + extract ADE20K to the local disk
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from a1lib.store import lock_is_live, worker_id
from a1lib.worker import cmd_status, is_done, live_lock, log

from a2lib.config import load_config
from a2lib.paths import Paths
from a2lib.plan import load_plan


def runnable(plan, paths, owner, steal=False, only=None, me=None):
    """-> (ready, blocked) from the MIRROR (call paths.store.refresh() first). blocked: waiting for lr_choice_a2.json."""
    lr = paths.store.read_json("lr_choice_a2.json", {}) or {}
    ready, blocked = [], []
    for e in plan["runs"]:
        rid = e["run_id"]
        if (only and not rid.startswith(only)) or is_done(paths, rid):
            continue
        info = live_lock(paths, rid)
        if lock_is_live(info) and info.get("worker") != me:
            continue
        if e["owner"] != owner and not steal:
            continue
        if e["needs_lr"]:
            cfg = load_config(Path(plan["_dir"]) / e["config"])
            if f"{cfg.model}|{cfg.lr_group}" not in lr:
                blocked.append(e)
                continue
        ready.append(e)
    return ready, blocked


def cmd_run(paths, plan, a, me):
    from a2lib.train import TimeUp, run_training    # imported late: `status`/`list` must work without torch
    failed, busy = set(), set()
    deadline = time.time() + a.max_hours * 3600 if a.max_hours else None
    wait_until = time.time() + a.wait_minutes * 60
    while True:
        paths.store.refresh()
        ready, blocked = runnable(plan, paths, a.owner, a.steal, a.only, me)
        ready = [e for e in ready if e["run_id"] not in failed | busy]
        if ready:
            e = ready[0]
            if deadline and time.time() + 600 > deadline:
                log("less than 10 minutes of the time budget left: not starting another run")
                return
            cfg = load_config(Path(plan["_dir"]) / e["config"])
            log(f"=== {e['run_id']}  (est {e['est_hours']:.2f} h; {len(ready)} ready, {len(blocked)} blocked) [{me}] ===")
            try:
                run_training(cfg, paths, a.owner, me, data_root=a.data_root, force=a.force, deadline=deadline)
            except KeyboardInterrupt:
                log("stopped (checkpoint pushed)")
                return
            except TimeUp as ex:
                log(str(ex))
                return
            except RuntimeError as ex:
                if "locked by worker" in str(ex) or "could not acquire the lock" in str(ex):
                    log(f"   {e['run_id']} was taken by another worker meanwhile -> next")
                    busy.add(e["run_id"])
                else:
                    log(f"!!! {e['run_id']} failed: {type(ex).__name__}: {str(ex)[:300]}")
                    failed.add(e["run_id"])
            except Exception as ex:  # noqa: BLE001
                log(f"!!! {e['run_id']} failed: {type(ex).__name__}: {str(ex)[:300]}")
                failed.add(e["run_id"])
            continue
        if blocked and time.time() < wait_until:
            log(f"{len(blocked)} run(s) wait for lr_choice_a2.json; sleeping 5 min (idle time still counts against the quota - prefer stopping)")
            time.sleep(300)
            continue
        break
    if blocked:
        keys = sorted({f"{c.model}|{c.lr_group}" for c in (load_config(Path(plan["_dir"]) / e["config"]) for e in blocked)})
        log(f"blocked on lr_choice_a2 for {len(blocked)} run(s); missing LR groups: {keys}\n"
            "   -> finish the sweep runs (`worker status`, rows a2_sw_*), then the coordinator runs `python -m a2lib.plan choose-lr`")
    if failed:
        log(f"FAILED: {sorted(failed)}  -> read status.json message, tell the team; rerun the worker to retry only after the cause is fixed")
    if not ready and not blocked and not failed:
        log("nothing left to do for this owner (other workers may still be running their last runs)")


def cmd_prepare():
    from a2lib.data import ensure_data
    from a2lib.paths import default_local
    root = ensure_data(default_local() / "data")
    log(f"data ready at {root}")


def cmd_launch(a, session):
    """Prepare the data ONCE, then start one worker process per GPU, each pinned to its GPU and with a unique worker id."""
    cmd_prepare()
    procs = []
    for g in range(a.gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), A1_SESSION=session)
        argv = [sys.executable, "-m", "a2lib.worker", "run", "--owner", a.owner, "--a2-dir", a.a2_dir, "--wait-minutes", str(a.wait_minutes)]
        for flag, val in (("--max-hours", a.max_hours), ("--only", a.only), ("--data-root", a.data_root)):
            if val:
                argv += [flag, str(val)]
        if a.steal:
            argv.append("--steal")
        p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs.append(p)

        def pump(proc=p, tag=f"[g{g}]"):
            for line in proc.stdout:
                print(tag, line, end="", flush=True)
        threading.Thread(target=pump, daemon=True).start()
        time.sleep(20)   # stagger so the workers do not race for the same run
    try:
        codes = [p.wait() for p in procs]
    except KeyboardInterrupt:
        for p in procs:
            p.send_signal(2)
        codes = [p.wait() for p in procs]
    log(f"workers finished with exit codes {codes}")
    return max(codes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["launch", "run", "status", "list", "prepare"])
    ap.add_argument("--owner", default=None)
    ap.add_argument("--a2-dir", default="A2")
    ap.add_argument("--steal", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--wait-minutes", type=float, default=0)
    ap.add_argument("--max-hours", type=float, default=None, help="stop cleanly (checkpoint pushed) so a session ends before Kaggle's 12 h limit")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--session", default=None)
    a = ap.parse_args()
    if a.cmd == "prepare":
        return cmd_prepare()
    paths = Paths.from_env()
    plan = load_plan(Path(a.a2_dir) / "plan.json")
    plan["_dir"] = a.a2_dir
    if a.cmd == "status":
        plan["runs"] = plan["runs"]
        return cmd_status(paths, plan)
    if not a.owner or a.owner not in plan["owners"]:
        ap.error(f"--owner must be one of {plan['owners']}")
    session = a.session or os.environ.get("A1_SESSION") or time.strftime("%m%d-%H%M")
    os.environ["A1_SESSION"] = session
    if a.cmd == "launch":
        sys.exit(cmd_launch(a, session))
    me = worker_id(a.owner)
    if a.cmd == "list":
        paths.store.refresh()
        ready, blocked = runnable(plan, paths, a.owner, a.steal, a.only, me)
        for e in ready:
            print("ready  ", e["run_id"], f"{e['est_hours']:.2f}h")
        for e in blocked:
            print("blocked", e["run_id"], "(needs lr_choice_a2)")
        return
    cmd_run(paths, plan, a, me)


if __name__ == "__main__":
    main()
