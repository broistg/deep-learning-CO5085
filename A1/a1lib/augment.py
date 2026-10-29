"""Training/eval transforms, Mixup/CutMix, soft-target loss, GPU normalisation. Exactly the recipe in the protocol (cfg.augmentation)."""
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM

MEAN255 = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD255 = (0.229 * 255, 0.224 * 255, 0.225 * 255)
_INTERP = {"bilinear": IM.BILINEAR, "nearest": IM.NEAREST, "bicubic": IM.BICUBIC}


def train_transform(aug):
    r, ra = aug["random_resized_crop"], aug["randaugment"]
    return T.Compose([
        T.RandomResizedCrop(r["size"], scale=tuple(r["scale"]), ratio=tuple(r["ratio"]), interpolation=_INTERP[r["interpolation"]]),
        T.RandomHorizontalFlip(aug["hflip_p"]),
        T.RandAugment(num_ops=ra["num_ops"], magnitude=ra["magnitude"], num_magnitude_bins=ra["num_magnitude_bins"],
                      interpolation=_INTERP[ra["interpolation"]], fill=None),
        T.PILToTensor(),
    ])


def eval_transform(aug):
    e = aug["eval"]
    return T.Compose([T.Resize(e["resize"], interpolation=IM.BILINEAR), T.CenterCrop(e["crop"]), T.PILToTensor()])


def make_mixup(aug, num_classes):
    """Label smoothing is applied HERE, once, inside the Mixup target. The loss must NOT smooth again."""
    import timm
    m = aug["mixup"]
    return timm.data.Mixup(mixup_alpha=m["mixup_alpha"], cutmix_alpha=m["cutmix_alpha"], cutmix_minmax=None, prob=m["prob"],
                           switch_prob=m["switch_prob"], mode=m["mode"], correct_lam=m["correct_lam"],
                           label_smoothing=m["label_smoothing"], num_classes=num_classes)


def soft_target_ce(logits, target):
    """Soft-target cross-entropy in fp32 (no extra label_smoothing argument anywhere)."""
    return torch.sum(-target * torch.log_softmax(logits.float(), dim=-1), dim=-1).mean()


def gpu_normalize(x_uint8, device, channels_last=False):
    mean = torch.tensor(MEAN255, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD255, device=device).view(1, 3, 1, 1)
    x = (x_uint8.to(device, non_blocking=True).float() - mean) / std
    return x.contiguous(memory_format=torch.channels_last) if channels_last else x
