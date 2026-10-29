"""timm model factory, the four freeze modes, parameter groups. Reuses the logic validated by bench2.py on Colab."""
import timm

N_CLASSES = 101
FAMILY = {  # timm name -> family (only used for channels_last)
    "resnet50.a1_in1k": "cnn",
    "deit_small_patch16_224.fb_in1k": "vit",
    "convnext_tiny.fb_in1k": "cnn",
    "swin_tiny_patch4_window7_224.ms_in1k": "vit",
}


def make_model(name, pretrained, drop_rate=0.0, drop_path_rate=0.0, num_classes=N_CLASSES):
    """drop_rate / drop_path_rate are set explicitly (timm Swin's default drop_path_rate may be non-zero)."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate,
                             drop_path_rate=drop_path_rate)


def apply_freeze(model, mode):
    """scratch/full -> everything trainable. head -> classifier only.
    partial -> last stage + head (ResNet layer4, ConvNeXt stage 4, Swin layer 4) or last half of the blocks + norm + head (DeiT)."""
    all_train = mode in ("scratch", "full")
    for p in model.parameters():
        p.requires_grad = all_train
    if all_train:
        return
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    if mode == "head":
        return
    if mode != "partial":
        raise ValueError(mode)
    if hasattr(model, "layer4"):
        mods = [model.layer4]
    elif hasattr(model, "blocks"):
        n = len(model.blocks)
        mods = list(model.blocks[n // 2:]) + [model.norm, model.fc_norm]
    elif hasattr(model, "stages"):
        mods = [model.stages[-1], model.head]
    elif hasattr(model, "layers"):
        mods = [model.layers[-1], model.norm, model.head]
    else:
        raise NotImplementedError(type(model).__name__)
    for m in mods:
        for p in m.parameters():
            p.requires_grad = True


def set_train_mode(model):
    """model.train(), except fully frozen modules stay in eval (BatchNorm statistics and dropout must not change while 'frozen')."""
    model.train()
    for m in model.modules():
        ps = list(m.parameters())
        if ps and not any(p.requires_grad for p in ps):
            m.eval()


def count_params(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, sum(p.numel() for p in model.parameters())


def param_groups(model, weight_decay):
    """AdamW: weight decay on trainable parameters with >= 2 dims only (not biases / norm parameters)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
