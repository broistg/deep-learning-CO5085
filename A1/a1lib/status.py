"""Per-run status.json on the shared folder. SINGLE WRITER: only the run's current owner writes it. Everyone else only reads."""
import socket
import time
from pathlib import Path

from common.repro import read_json, write_json


class Status:
    def __init__(self, run_dir, run_id, owner, epochs):
        self.path = Path(run_dir) / "status.json"
        old = read_json(self.path, {}) or {}
        self.d = {"run_id": run_id, "owner": owner, "state": "pending", "epoch": 0, "epochs": epochs, "host": socket.gethostname(),
                  "gpu": None, "started": old.get("started"), "updated": time.time(), "message": "", "eta_min": None}
        if old.get("state") in ("running", "interrupted", "failed"):
            self.d["epoch"] = old.get("epoch", 0)

    def set(self, **kw):
        self.d.update(kw)
        self.d["updated"] = time.time()
        write_json(self.path, self.d)


def read_all(shared_dir):
    out = []
    runs = Path(shared_dir) / "runs"
    if runs.exists():
        for p in sorted(runs.glob("*/status.json")):
            d = read_json(p)
            if d:
                out.append(d)
    return out


def human_age(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"
