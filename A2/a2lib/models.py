"""The two models of A2 and their optimiser parameter groups.

A  segformer_b0     SegFormer-B0: MiT-B0 encoder (ImageNet-1k, `nvidia/mit-b0`, ENCODER ONLY) + all-MLP decoder, via HF transformers.
B  deeplabv3p_mnv2  DeepLabV3+ MobileNetV2-D8: our re-implementation of the mmsegmentation model of the paper's Table 2 (15.35M parameters with the
                    auxiliary head, 69.4 GMACs at 512x512; measured on a T4). Backbone: torchvision MobileNetV2 (ImageNet).

NEVER initialise from `nvidia/segformer-b0-finetuned-ade-*` (already trained on ADE20K): only `nvidia/mit-b0` is used for training.
Both models return logits at 1/4 of the input resolution; the loss and the evaluation upsample them (bilinear)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2lib.config import N_CLASSES

SEGFORMER_ENCODER = "nvidia/mit-b0"


def cbr(i, o, k=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=0 if k == 1 else d * (k // 2), dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


def sep(i, o, d=1):
    return nn.Sequential(nn.Conv2d(i, i, 3, padding=d, dilation=d, groups=i, bias=False), nn.BatchNorm2d(i), nn.ReLU(True),
                         nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class DeepLabMMSeg(nn.Module):
    """mmseg configs/mobilenet_v2/mobilenet-v2-d8 + DepthwiseSeparableASPPHead (verified from source): strides (1,2,2,1,1,1,1), dilations
    (1,1,1,2,2,4,4) -> output stride 8, 320 channels; image pool + ASPP (1x1, then depthwise-separable 3x3 with dilations 12/24/36), all 512
    channels; a FULL 3x3 bottleneck 5*512 -> 512 (11.8M parameters); low-level 24 -> 48; two depthwise-separable 3x3 convs; 1x1 classifier;
    auxiliary FCN head on the 96-channel stage (training only)."""

    def __init__(self, pretrained=True, num_classes=N_CLASSES, ch=512, c1=48, dilations=(12, 24, 36)):
        super().__init__()
        import torchvision
        w = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        f = torchvision.models.mobilenet_v2(weights=w).features[:18]          # stem + 17 blocks, ends with the 320-channel block
        for idx, dil in [(range(7, 11), 2), (range(11, 14), 2), (range(14, 17), 4), (range(17, 18), 4)]:
            for i in idx:
                for m in f[i].modules():
                    if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):   # the depthwise conv: no stride, dilate instead
                        m.stride, m.dilation, m.padding = (1, 1), (dil, dil), (dil, dil)
        self.low, self.mid, self.high = f[:4], f[4:14], f[14:]               # 24ch @ 1/4 | 96ch @ 1/8 | 320ch @ 1/8
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), cbr(320, ch))
        self.aspp = nn.ModuleList([cbr(320, ch)] + [sep(320, ch, d) for d in dilations])
        self.bottleneck = cbr(5 * ch, ch, 3)
        self.c1 = cbr(24, c1)
        self.sep_bottleneck = nn.Sequential(sep(ch + c1, ch), sep(ch, ch))
        self.cls = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(ch, num_classes, 1))
        self.aux = nn.Sequential(cbr(96, 256, 3), nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, 1))   # training only

    def forward(self, x):
        c1 = self.low(x)
        mid = self.mid(c1)
        x4 = self.high(mid)
        p = F.interpolate(self.pool(x4), size=x4.shape[2:], mode="bilinear", align_corners=False)
        y = self.bottleneck(torch.cat([p] + [m(x4) for m in self.aspp], dim=1))
        c1f = self.c1(c1)
        y = F.interpolate(y, size=c1f.shape[2:], mode="bilinear", align_corners=False)
        out = self.cls(self.sep_bottleneck(torch.cat([y, c1f], dim=1)))     # logits at 1/4 resolution, like mmseg
        if self.training:
            return out, self.aux(mid)                                        # (main, aux): aux is 1/8 resolution
        return out


class SegFormerLogits(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).logits


def make_segformer(pretrained, num_classes=N_CLASSES):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    if pretrained:
        m = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ENCODER, num_labels=num_classes)   # encoder weights only; decoder is new
    else:
        cfg = SegformerConfig.from_pretrained(SEGFORMER_ENCODER, num_labels=num_classes)                    # config only (architecture), no weights
        m = SegformerForSemanticSegmentation(cfg)
    return SegFormerLogits(m)


def make_model(cfg):
    if cfg.model == "segformer_b0":
        return make_segformer(cfg.pretrained, cfg.num_classes)
    if cfg.model == "deeplabv3p_mnv2":
        return DeepLabMMSeg(cfg.pretrained, cfg.num_classes)
    raise ValueError(cfg.model)


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad), tot


def param_groups(model, cfg, lr):
    """SegFormer (NVlabs/SegFormer B0 config, paramwise_cfg): decoder ('head') lr x10; weight decay 0 for every parameter whose name contains
    'norm' (LayerNorm); mmseg applies only ONE rule per parameter and 'head' wins over 'norm', so decoder norms keep their decay.
    DeepLab (mmseg schedule_160k): one group, SGD weight decay 5e-4 on everything, no multipliers."""
    if cfg.model != "segformer_b0":
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": cfg.weight_decay}]
    groups = {"enc_decay": [], "enc_nodecay": [], "head": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "decode_head" in n:
            groups["head"].append(p)
        elif "norm" in n:
            groups["enc_nodecay"].append(p)
        else:
            groups["enc_decay"].append(p)
    return [{"params": groups["enc_decay"], "lr": lr, "weight_decay": cfg.weight_decay},
            {"params": groups["enc_nodecay"], "lr": lr, "weight_decay": 0.0},
            {"params": groups["head"], "lr": lr * cfg.head_lr_mult, "weight_decay": cfg.weight_decay}]


def make_optimizer(model, cfg, lr):
    g = param_groups(model, cfg, lr)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(g, lr=lr, betas=tuple(cfg.betas))
    return torch.optim.SGD(g, lr=lr, momentum=cfg.momentum)


def seg_loss(out, target, ignore=255):
    """Cross-entropy (ignore 255) on logits bilinearly upsampled to the label size: the mmseg / paper way, for BOTH models."""
    up = F.interpolate(out.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
    return F.cross_entropy(up, target.long(), ignore_index=ignore)


def total_loss(outputs, target, aux_weight):
    if isinstance(outputs, tuple):
        main, aux = outputs
        return seg_loss(main, target) + aux_weight * seg_loss(aux, target)
    return seg_loss(outputs, target)
