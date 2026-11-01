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


def vit_layer_id(name, depth):
    """Layer index of a timm ViT parameter: 0 = embeddings, 1..depth = blocks, depth + 1 = final norm / head."""
    if name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token", "dist_token")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return depth + 1


def llrd_scale(name, depth, llrd):
    """LR multiplier of a parameter under layer-wise decay: the head gets 1, each block below it gets `llrd` times the one above."""
    return llrd ** (depth + 1 - vit_layer_id(name, depth))
