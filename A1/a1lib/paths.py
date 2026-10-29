"""Where things live. Machine-specific, so it comes from environment variables, not from the YAML configs.

A1_SHARED : the shared Drive folder (default /content/drive/MyDrive/a1_shared)   -- results, checkpoints, status, data tar
A1_LOCAL  : fast local disk of the Colab VM (default /content/a1_local)           -- extracted dataset, working checkpoints
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    shared: Path
    local: Path

    @classmethod
    def from_env(cls):
        return cls(Path(os.environ.get("A1_SHARED", "/content/drive/MyDrive/a1_shared")),
                   Path(os.environ.get("A1_LOCAL", "/content/a1_local")))

    def remote_run(self, run_id):
        return self.shared / "runs" / run_id

    def local_run(self, run_id):
        return self.local / "runs" / run_id

    @property
    def lr_choice(self):
        return self.shared / "lr_choice.json"

    @property
    def data_local(self):
        return self.local / "data"
