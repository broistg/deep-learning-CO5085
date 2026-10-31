"""The commands a team member runs on Kaggle. Works through the run list in plan order, resumes interrupted runs.

    python -m a1lib.worker launch --owner dat --gpus 2 --max-hours 11    # one worker per GPU in this session (the normal way)
    python -m a1lib.worker run --owner dat --max-hours 11                # a single worker on the visible GPU
    python -m a1lib.worker run --owner dat --steal                        # ALSO runs owned by others that are unfinished and unlocked
    python -m a1lib.worker run --owner dat --only sw_resnet50             # only run_ids starting with this
    python -m a1lib.worker status                                         # table of every run (reads the store)
    python -m a1lib.worker list --owner dat                               # what would run next

Several sessions (and both accounts) can work on the same list at the same time: every worker takes a per-run lock in the store, so
no run is trained twice. Each worker has a unique id <owner>-<session>-g<gpu>.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from a1lib.config import load_config, lr_group
from a1lib.paths import Paths
from a1lib.plan import load_plan
from a1lib.status import human_age, read_all
from a1lib.store import STALE_S, lock_is_live, worker_id


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def is_done(paths, rid):
    return (paths.shared / "runs" / rid / "result.json").exists()


def live_lock(paths, rid):
    return paths.store.read_json(f"runs/{rid}/run.lock.json", {}) or {}


def runnable(plan, paths, owner, steal=False, only=None, me=None):
    """-> (ready, blocked) from the MIRROR (call paths.store.refresh() first).
    ready: unfinished runs whose LR is known and that no other live worker holds. blocked: waiting for lr_choice.json."""
    lr = paths.store.read_json("lr_choice.json", {}) or {}
    ready, blocked = [], []
    for e in plan["runs"]:
        rid = e["run_id"]
        if only and not rid.startswith(only):
            continue
        if is_done(paths, rid):
            continue
        info = live_lock(paths, rid)
        held_by_other = lock_is_live(info) and info.get("worker") != me
        if held_by_other:
            continue
        if e["owner"] != owner and not steal:
            continue
        if e["needs_lr"]:
            cfg = load_config(Path(plan["_a1"]) / e["config"])
            if f"{cfg.model}|{lr_group(cfg.mode)}" not in lr:
                blocked.append(e)
                continue
        ready.append(e)
    return ready, blocked


def cmd_status(paths, plan):
    paths.store.refresh()
    st = {d["run_id"]: d for d in read_all(paths.shared)}
    now = time.time()
    print(f"{'run_id':34s} {'owner':6s} {'state':11s} {'epoch':>7s} {'age':>6s}  worker / note")
    n_done = 0
    for e in plan["runs"]:
        rid = e["run_id"]
        d = st.get(rid, {})
        state = "done" if is_done(paths, rid) else d.get("state", "-")
        n_done += state == "done"
        info = live_lock(paths, rid)
        if state == "running" and not lock_is_live(info):
            state = "STALE?"
        ep = f"{d.get('epoch', 0)}/{d.get('epochs', '?')}" if d else ""
        age = human_age(now - d["updated"]) if d.get("updated") else ""
        note = d.get("worker") or ""
        if state in ("failed", "STALE?"):
            note = ((d.get("message") or "").strip().splitlines() or [""])[-1][:60]
        print(f"{rid:34s} {e['owner']:6s} {state:11s} {ep:>7s} {age:>6s}  {note}")
    live = sorted({i.get("worker") for i in (live_lock(paths, e["run_id"]) for e in plan["runs"]) if lock_is_live(i)})
    print(f"\n{n_done}/{len(plan['runs'])} runs finished; est. remaining "
          f"{sum(e['est_hours'] for e in plan['runs'] if not is_done(paths, e['run_id'])):.1f} GPU-h; live workers: {', '.join(live) or 'none'}")


def cmd_run(paths, plan, a, me):
    from a1lib.train import TimeUp, run_training  # imported late: `status`/`list` must work without torch
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
            cfg = load_config(Path(plan["_a1"]) / e["config"])
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
            except Exception as ex:  # noqa: BLE001  one bad run must not stop the queue; it stays 'failed' and is reported
                log(f"!!! {e['run_id']} failed: {type(ex).__name__}: {str(ex)[:300]}")
                failed.add(e["run_id"])
            continue
        if blocked and time.time() < wait_until:
            log(f"{len(blocked)} run(s) wait for lr_choice.json; sleeping 5 min (idle time still counts against the quota - prefer stopping)")
            time.sleep(300)
            continue
        break
    if blocked:
        keys = sorted({f"{c.model}|{lr_group(c.mode)}" for c in (load_config(Path(plan["_a1"]) / e["config"]) for e in blocked)})
        log(f"blocked on lr_choice for {len(blocked)} run(s); missing LR groups: {keys}\n"
            "   -> their sweep runs must be done first (`worker status`, rows sw_*); then the coordinator runs `python -m a1lib.plan choose-lr` "
            "(it lists every group: 'sweep not finished yet' means some sw_* runs of that group are still missing or failed)")
    if failed:
        log(f"FAILED: {sorted(failed)}  -> read status.json message, tell the team; rerun the worker to retry only after the cause is fixed")
    if not ready and not blocked and not failed:
        log("nothing left to do for this owner (other workers may still be running their last runs)")


def cmd_prepare(paths):
    from a1lib.data import ensure_data
    root = ensure_data(paths.data_local)
    log(f"data ready at {root}")


def cmd_launch(paths, a, session):
    """Prepare the data ONCE, then start one worker process per GPU, each pinned to its GPU and with a unique worker id."""
    cmd_prepare(paths)
    procs = []
    for g in range(a.gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), A1_SESSION=session)
        argv = [sys.executable, "-m", "a1lib.worker", "run", "--owner", a.owner, "--a1-dir", a.a1_dir, "--wait-minutes", str(a.wait_minutes)]
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
        time.sleep(20)   # stagger: worker 1 starts after worker 0 has taken its first lock, so they do not race for the same run
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
    ap.add_argument("--a1-dir", default="A1")
    ap.add_argument("--steal", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--wait-minutes", type=float, default=0)
    ap.add_argument("--max-hours", type=float, default=None, help="stop cleanly (checkpoint pushed) so a session ends before Kaggle's 12 h limit")
    ap.add_argument("--gpus", type=int, default=2, help="launch: one worker per GPU")
    ap.add_argument("--session", default=None, help="session tag in worker ids (default: a timestamp)")
    a = ap.parse_args()
    paths = Paths.from_env()
    if a.cmd == "prepare":
        return cmd_prepare(paths)
    plan = load_plan(Path(a.a1_dir) / "plan.json")
    plan["_a1"] = a.a1_dir
    if a.cmd == "status":
        return cmd_status(paths, plan)
    if not a.owner or a.owner not in plan["owners"]:
        ap.error(f"--owner must be one of {plan['owners']}")
    session = a.session or os.environ.get("A1_SESSION") or time.strftime("%m%d-%H%M")
    os.environ["A1_SESSION"] = session
    if a.cmd == "launch":
        sys.exit(cmd_launch(paths, a, session))
    me = worker_id(a.owner)
    if a.cmd == "list":
        paths.store.refresh()
        ready, blocked = runnable(plan, paths, a.owner, a.steal, a.only, me)
        for e in ready:
            print("ready  ", e["run_id"], f"{e['est_hours']:.2f}h")
        for e in blocked:
            print("blocked", e["run_id"], "(needs lr_choice)")
        return
    cmd_run(paths, plan, a, me)


if __name__ == "__main__":
    main()
