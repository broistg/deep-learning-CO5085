"""Where things live (A2). Own shared Hugging Face store (`dl4dl/a2-store`, separate from A1's final store), own local folder.

A2_STORE : 'hf:<org>/<repo>' or a folder (tests). Falls back to A1_STORE only for local tests.
A2_LOCAL : fast local disk of the session (default /kaggle/temp/a2_local, else /tmp/a2_local): extracted ADE20K and working checkpoints.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from a1lib.store import Store, open_store


def default_local():
    if os.environ.get("A2_LOCAL"):
        return Path(os.environ["A2_LOCAL"])
    return Path("/kaggle/temp/a2_local") if Path("/kaggle/temp").exists() else Path("/tmp/a2_local")


@dataclass
class Paths:
    store: Store
    local: Path

    @classmethod
    def from_env(cls, create=False):
        local = default_local()
        spec = os.environ.get("A2_STORE") or os.environ.get("A1_STORE")
        return cls(open_store(spec, local_dir=local, create=create), local)

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
