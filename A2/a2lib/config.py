"""Run configuration for A2. One YAML per run (written by `python -m a2lib.plan make`), committed to the repo.
Every value that changes the science is in the YAML; machine settings (paths, GPU) are not."""
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from common.repro import config_hash

MODELS = ("segformer_b0", "deeplabv3p_mnv2")
N_CLASSES = 150
IGNORE = 255
# recipe per model (protocol v3 section 4; NVlabs/SegFormer B0 config and mmseg schedule_160k + M-V2-D8 config)
MODEL_DEFAULTS = {
    "segformer_b0": dict(optimizer="adamw", weight_decay=0.01, head_lr_mult=10.0, poly_power=1.0, min_lr=0.0, warmup_frac=0.0625,
                         warmup_start_frac=1e-6, micro_batch=16, aux_weight=0.0),
    "deeplabv3p_mnv2": dict(optimizer="sgd", weight_decay=5e-4, head_lr_mult=1.0, poly_power=0.9, min_lr=1e-4, warmup_frac=0.0,
                            warmup_start_frac=1.0, micro_batch=8, aux_weight=0.4),
}


def default_augmentation():
    """mmseg ade20k.py train pipeline (verified from source; see A2/README.md)."""
    return {"scale": [2048, 512], "ratio_range": [0.5, 2.0], "crop": 512, "cat_max_ratio": 0.75, "flip_p": 0.5,
            "photometric": True, "mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375],
            "eval_scale": [2048, 512]}


@dataclass
class SegConfig:
    run_id: str
    model: str
    pretrained: bool
    epochs: int
    seed: int = 0
    kind: str = "main"             # sweep | main | seed | stretch
    data_fraction: float = 1.0     # random nested subset of the 19,210 training images (seed 42 order)
    lr_peak: object = "auto"       # float, or "auto" = look up the sweep result for (model, init)
    batch_size: int = 16           # effective batch = micro_batch x accumulation
    micro_batch: int = 16
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    momentum: float = 0.9
    head_lr_mult: float = 10.0
    poly_power: float = 1.0
    min_lr: float = 0.0            # absolute LR floor of the poly schedule
    warmup_frac: float = 0.0625
    warmup_start_frac: float = 1e-6
    aux_weight: float = 0.0
    amp: str = "fp16"
    workers: int = 2
    num_classes: int = N_CLASSES
    augmentation: dict = field(default_factory=default_augmentation)
    split_sha256_16: str = ""
    notes: str = ""

    @property
    def init(self):
        return "pretrained" if self.pretrained else "scratch"

    @property
    def lr_group(self):
        return self.init

    @property
    def accum(self):
        return self.batch_size // self.micro_batch

    def to_dict(self):
        return dataclasses.asdict(self)


def make_config(run_id, model, pretrained, epochs, kind, seed=0, fraction=1.0, lr="auto", notes="", split_sha=""):
    d = dict(MODEL_DEFAULTS[model])
    return validate(SegConfig(run_id=run_id, model=model, pretrained=pretrained, epochs=epochs, seed=seed, kind=kind, data_fraction=fraction,
                              lr_peak=lr, notes=notes, split_sha256_16=split_sha, **d))


def validate(cfg):
    if cfg.model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {cfg.model!r}")
    if cfg.optimizer not in ("adamw", "sgd"):
        raise ValueError("optimizer must be adamw or sgd")
    if not (0 < cfg.data_fraction <= 1):
        raise ValueError("data_fraction must be in (0, 1]")
    if cfg.batch_size % cfg.micro_batch:
        raise ValueError("batch_size must be a multiple of micro_batch")
    if cfg.lr_peak != "auto" and not (isinstance(cfg.lr_peak, (int, float)) and cfg.lr_peak > 0):
        raise ValueError("lr_peak must be a positive number or 'auto'")
    if cfg.epochs < 1:
        raise ValueError("epochs must be >= 1")
    return cfg


def from_dict(d):
    known = {f.name for f in dataclasses.fields(SegConfig)}
    extra = set(d) - known
    if extra:
        raise ValueError(f"unknown config keys: {sorted(extra)}")
    return validate(SegConfig(**d))


def load_config(path):
    return from_dict(yaml.safe_load(Path(path).read_text(encoding="utf8")))


def dump_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf8")


def resolve_lr(cfg, lr_choice):
    """Copy of cfg with a numeric lr_peak. lr_choice = {"<model>|<init>": {"lr": float, ...}}."""
    if cfg.lr_peak != "auto":
        return cfg
    key = f"{cfg.model}|{cfg.lr_group}"
    if not lr_choice or key not in lr_choice:
        raise LookupError(f"no LR chosen yet for '{key}' (run the LR sweep, then `python -m a2lib.plan choose-lr`)")
    v = lr_choice[key]
    return dataclasses.replace(cfg, lr_peak=float(v["lr"] if isinstance(v, dict) else v))


def stable_hash(d):
    return config_hash({k: v for k, v in d.items() if k != "notes"})
