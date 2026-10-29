"""Run configuration. One YAML per run (written by `python -m a1lib.plan make`), committed to the repo.
Every value that changes the science is in the YAML; machine settings (paths, workers' count is a recipe value, GPU) are not."""
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODES = ("scratch", "head", "partial", "full")
LR_GROUPS = ("head", "finetune", "scratch")


def default_augmentation():
    return {
        "random_resized_crop": {"size": 224, "scale": [0.25, 1.0], "ratio": [0.75, 1.3333333333333333], "interpolation": "bilinear"},
        "hflip_p": 0.5,
        "randaugment": {"num_ops": 2, "magnitude": 9, "num_magnitude_bins": 31, "interpolation": "nearest"},
        "mixup": {"mixup_alpha": 0.8, "cutmix_alpha": 1.0, "prob": 1.0, "switch_prob": 0.5, "mode": "batch",
                  "correct_lam": True, "label_smoothing": 0.1},
        "eval": {"resize": 256, "crop": 224},
    }


@dataclass
class RunConfig:
    run_id: str
    model: str
    mode: str                      # scratch | head | partial | full
    epochs: int
    seed: int = 0
    kind: str = "main"             # sweep | main | seed | ablation | stretch
    pretrained: bool = True
    data_fraction: float = 1.0     # class-balanced nested subset of the 70,700 training images
    lr_peak: object = "auto"       # float, or "auto" = look up the sweep result for (model, lr_group)
    batch_size: int = 64
    warmup_frac: float = 0.05
    warmup_start_frac: float = 0.01
    min_lr_frac: float = 0.01
    weight_decay: float = 0.05     # AdamW, applied to parameters with >= 2 dims only
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    amp: str = "fp16"
    workers: int = 2
    img_size: int = 224
    augmentation: dict = field(default_factory=default_augmentation)
    split_json_sha256_16: str = "75ed81bfe380c089"
    notes: str = ""

    @property
    def lr_group(self):
        return lr_group(self.mode)

    def to_dict(self):
        return dataclasses.asdict(self)


def lr_group(mode):
    return {"scratch": "scratch", "head": "head", "partial": "finetune", "full": "finetune"}[mode]


def validate(cfg):
    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {cfg.mode!r}")
    if (cfg.mode == "scratch") == bool(cfg.pretrained):
        raise ValueError("mode 'scratch' must have pretrained: false, every other mode pretrained: true")
    if not (0 < cfg.data_fraction <= 1):
        raise ValueError("data_fraction must be in (0, 1]")
    if cfg.batch_size % 2:
        raise ValueError("batch_size must be even (Mixup/CutMix pairs each sample with the flipped batch)")
    if cfg.lr_peak != "auto" and not (isinstance(cfg.lr_peak, (int, float)) and cfg.lr_peak > 0):
        raise ValueError("lr_peak must be a positive number or 'auto'")
    if cfg.epochs < 1:
        raise ValueError("epochs must be >= 1")
    return cfg


def from_dict(d):
    known = {f.name for f in dataclasses.fields(RunConfig)}
    extra = set(d) - known
    if extra:
        raise ValueError(f"unknown config keys: {sorted(extra)}")
    return validate(RunConfig(**d))


def load_config(path):
    return from_dict(yaml.safe_load(Path(path).read_text(encoding="utf8")))


def dump_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf8")


def resolve_lr(cfg, lr_choice):
    """Return a copy of cfg with a numeric lr_peak. lr_choice = {"<model>|<lr_group>": {"lr": float, ...}} or {... : float}."""
    if cfg.lr_peak != "auto":
        return cfg
    key = f"{cfg.model}|{cfg.lr_group}"
    if not lr_choice or key not in lr_choice:
        raise LookupError(f"no LR chosen yet for '{key}' (run the LR sweep, then `python -m a1lib.plan choose-lr`)")
    v = lr_choice[key]
    lr = float(v["lr"] if isinstance(v, dict) else v)
    return dataclasses.replace(cfg, lr_peak=lr)
