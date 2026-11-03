"""Maintenance of the A2 store (uses A2_STORE, never A1_STORE):

    python -m a2lib.store_cli init | ls | stats | squash [--force]

`squash` rewrites the Hugging Face history to reclaim LFS quota. It REFUSES while any run has a live lock (a worker is running), because a
squash during a checkpoint push could lose it. Stop the workers (or wait until their locks go stale, ~15 min after the last heartbeat) first."""
import argparse
import os

from a1lib import store as S
from a2lib.paths import Paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "ls", "stats", "squash"])
    ap.add_argument("--actor", default=os.environ.get("A1_OWNER", "unknown"))
    ap.add_argument("--force", action="store_true", help="squash even with live locks (dangerous; only if the locks are known to be dead)")
    a = ap.parse_args()
    store = Paths.from_env(create=(a.cmd == "init")).store
    if a.cmd == "init":
        print("store ready:", store.describe())
    elif a.cmd == "ls":
        for rel, size in store.list_files():
            print(f"{size:>14,d}  {rel}")
    elif a.cmd == "stats":
        S.print_stats(store)
    else:
        live = S.live_workers(store)
        if live and not a.force:
            print("refusing to squash: workers are running on these runs:")
            for rid, w, age in live:
                print(f"   {rid}  worker {w}  last heartbeat {age / 60:.1f} min ago")
            print("Stop the workers first (or use --force only if you know these locks are dead).")
            raise SystemExit(2)
        S.squash(store, a.actor, a.force)


if __name__ == "__main__":
    main()
