"""The only command a team member runs on Colab. Works through that member's runs in plan order, resumes interrupted ones.

    python -m a1lib.worker run --owner dat            # own runs, in plan order
    python -m a1lib.worker run --owner dat --steal    # ALSO runs owned by others that are unfinished and not locked (agree in chat first)
    python -m a1lib.worker run --owner dat --only sw_resnet50   # only run_ids starting with this
    python -m a1lib.worker status                     # table of every run (reads the shared folder)
    python -m a1lib.worker list --owner dat           # what would run next
"""
import argparse
import time
from pathlib import Path

from common.repro import read_json
from a1lib.config import load_config, lr_group
from a1lib.paths import Paths
from a1lib.plan import load_plan
from a1lib.status import human_age, read_all


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def is_done(paths, rid):
    return (paths.remote_run(rid) / "result.json").exists()


def lock_is_live(paths, rid, stale_s=1200):
    info = read_json(paths.remote_run(rid) / "run.lock.json", {}) or {}
    return bool(info) and time.time() - info.get("heartbeat", 0) < stale_s


def runnable(plan, paths, owner, steal=False, only=None):
    """-> (ready, blocked). ready: unfinished runs whose LR is known. blocked: waiting for lr_choice.json."""
    lr = read_json(paths.lr_choice, {}) or {}
    ready, blocked = [], []
    for e in plan["runs"]:
        rid = e["run_id"]
        if only and not rid.startswith(only):
            continue
        if e["owner"] != owner and not (steal and not lock_is_live(paths, rid)):
            continue
        if is_done(paths, rid):
            continue
        if e["needs_lr"]:
            cfg = load_config(Path(plan["_a1"]) / e["config"])
            if f"{cfg.model}|{lr_group(cfg.mode)}" not in lr:
                blocked.append(e)
                continue
        ready.append(e)
    return ready, blocked


def cmd_status(paths, plan):
    st = {d["run_id"]: d for d in read_all(paths.shared)}
    now = time.time()
    print(f"{'run_id':34s} {'owner':8s} {'state':11s} {'epoch':>7s} {'age':>6s}  note")
    n_done = 0
    for e in plan["runs"]:
        d = st.get(e["run_id"], {})
        state = "done" if is_done(paths, e["run_id"]) else d.get("state", "-")
        n_done += state == "done"
        if state == "running" and not lock_is_live(paths, e["run_id"]):
            state = "STALE?"
        ep = f"{d.get('epoch', 0)}/{d.get('epochs', '?')}" if d else ""
        age = human_age(now - d["updated"]) if d.get("updated") else ""
        msg = (d.get("message") or "").strip().splitlines()[-1:] or [""]
        print(f"{e['run_id']:34s} {e['owner']:8s} {state:11s} {ep:>7s} {age:>6s}  {msg[0][:60] if state in ('failed', 'STALE?') else ''}")
    print(f"\n{n_done}/{len(plan['runs'])} runs finished; est. remaining "
          f"{sum(e['est_hours'] for e in plan['runs'] if not is_done(paths, e['run_id'])):.1f} GPU-h")


def cmd_run(paths, plan, a):
    from a1lib.train import run_training  # imported late: `status`/`list` must work without torch
    failed = []
    deadline = time.time() + a.wait_minutes * 60
    while True:
        ready, blocked = runnable(plan, paths, a.owner, a.steal, a.only)
        ready = [e for e in ready if e["run_id"] not in failed]
        if ready:
            e = ready[0]
            cfg = load_config(Path(plan["_a1"]) / e["config"])
            log(f"=== {e['run_id']}  (est {e['est_hours']:.2f} h; {len(ready)} ready, {len(blocked)} blocked) ===")
            try:
                run_training(cfg, paths, a.owner, data_root=a.data_root, force=a.force)
            except KeyboardInterrupt:
                log("stopped by user")
                return
            except Exception as ex:  # noqa: BLE001  one bad run must not stop the queue; it stays 'failed' and is reported
                log(f"!!! {e['run_id']} failed: {type(ex).__name__}: {str(ex)[:300]}")
                failed.append(e["run_id"])
            continue
        if blocked and time.time() < deadline:
            log(f"{len(blocked)} run(s) wait for lr_choice.json; sleeping 5 min (this counts as idle time on Colab - prefer stopping)")
            time.sleep(300)
            continue
        break
    if blocked:
        log(f"blocked on lr_choice: {[e['run_id'] for e in blocked][:6]} ... (run `python -m a1lib.plan choose-lr` once the sweep is complete)")
    if failed:
        log(f"FAILED: {failed}  -> read status.json message, tell the team; rerun the worker to retry only after the cause is fixed")
    if not ready and not blocked and not failed:
        log("nothing left to do for this owner")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "status", "list"])
    ap.add_argument("--owner", default=None)
    ap.add_argument("--a1-dir", default="A1")
    ap.add_argument("--steal", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--wait-minutes", type=float, default=0)
    a = ap.parse_args()
    paths = Paths.from_env()
    plan = load_plan(Path(a.a1_dir) / "plan.json")
    plan["_a1"] = a.a1_dir
    if a.cmd == "status":
        return cmd_status(paths, plan)
    if not a.owner or a.owner not in plan["owners"]:
        ap.error(f"--owner must be one of {plan['owners']}")
    if a.cmd == "list":
        ready, blocked = runnable(plan, paths, a.owner, a.steal, a.only)
        for e in ready:
            print("ready  ", e["run_id"], f"{e['est_hours']:.2f}h")
        for e in blocked:
            print("blocked", e["run_id"], "(needs lr_choice)")
        return
    cmd_run(paths, plan, a)


if __name__ == "__main__":
    main()
