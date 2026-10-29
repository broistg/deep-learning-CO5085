"""Learning-rate schedule: per-iteration linear warmup (from start_frac to 1) then cosine to min_frac. Pure Python."""
import math


def lr_factor(step, total_steps, warmup_frac=0.05, start_frac=0.01, min_frac=0.01):
    """Multiplier of the peak LR at optimizer step `step` (0-based). total_steps = epochs * steps_per_epoch."""
    warm = max(1, round(warmup_frac * total_steps))
    if step < warm:
        return start_frac + (1.0 - start_frac) * step / warm
    progress = min(1.0, (step - warm) / max(1, total_steps - warm))
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(opt, total_steps, cfg):
    import torch
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, total_steps, cfg.warmup_frac, cfg.warmup_start_frac, cfg.min_lr_frac))
