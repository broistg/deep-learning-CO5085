"""Where things live. Machine-specific, so it comes from environment variables, not from the YAML configs.

A1_STORE : 'hf:<org>/<repo>' (Kaggle, shared by both accounts) or a folder (tests)   -- see a1lib/store.py
A1_LOCAL : fast local disk of the session (default /kaggle/temp/a1_local, else /tmp/a1_local)
           -> extracted dataset and working checkpoints. NOT /kaggle/working (20 GB output cap, saved on commit).
"""
import os
from dataclasses import dataclass
from pathlib import Path

from a1lib.store import Store, open_store


def default_local():
    if os.environ.get("A1_LOCAL"):
        return Path(os.environ["A1_LOCAL"])
    return Path("/kaggle/temp/a1_local") if Path("/kaggle/temp").exists() else Path("/tmp/a1_local")


@dataclass
class Paths:
    store: Store
    local: Path

    @classmethod
    def from_env(cls, create=False):
        local = default_local()
        return cls(open_store(local_dir=local, create=create), local)

    @property
    def shared(self):
        """Read-only mirror of the store's small files (call store.refresh() first). Writes go through store.put()."""
        return self.store.mirror

    def local_run(self, run_id):
        return self.local / "runs" / run_id

    def run_rel(self, run_id, name=""):
        return f"runs/{run_id}/{name}".rstrip("/")

    @property
    def data_local(self):
        return self.local / "data"
