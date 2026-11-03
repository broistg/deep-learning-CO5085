"""Learning-rate schedule of the paper's recipe: mmseg PolyLR (power, absolute min_lr) times a linear warmup. Pure Python.
SegFormer-B0: power 1.0, min_lr 0, 1500 warmup iterations from ratio 1e-6 (= 6.25% of 24k). DeepLab (mmseg schedule_160k): power 0.9, min_lr 1e-4, no warmup."""


def lr_factor(step, total_steps, lr_peak, power=1.0, min_lr=0.0, warmup_frac=0.0, start_frac=1e-6):
    """Multiplier of lr_peak at optimizer step `step` (0-based, the step about to be taken)."""
    min_frac = min(1.0, min_lr / lr_peak) if lr_peak > 0 else 0.0
    t = min(1.0, step / max(1, total_steps))
    poly = min_frac + (1.0 - min_frac) * (1.0 - t) ** power
    warm = round(warmup_frac * total_steps)
    if warm > 0 and step < warm:
        poly *= start_frac + (1.0 - start_frac) * step / warm
    return poly


def make_scheduler(opt, total_steps, cfg, lr_peak):
    import torch
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, total_steps, lr_peak, cfg.poly_power, cfg.min_lr, cfg.warmup_frac, cfg.warmup_start_frac))
