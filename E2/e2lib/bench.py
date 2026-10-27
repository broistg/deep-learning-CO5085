"""Smoke benchmark: how long will the E2 plan take on THIS machine?  python -m e2lib.bench
Times 8 model variants on random data (GPU only: forward + backward + AdamW step, fp16, batch 128), reports ms/step, estimated
epoch time, peak memory, parameters, tokens and MACs per image, then extrapolates the total time of the 30 planned runs.
Also checks that the data pipeline (GPU augmentation) keeps up. Needs no dataset download."""
import argparse
import time

import torch
import torch.nn.functional as F

from common.repro import env_info, write_json

from .data import GPUSet, fake_cifar
from .model import TinyTransformer
from .plan import all_cfgs
from .train import E2_DIR, Cfg

STEPS_PER_EPOCH, VAL_IMGS = 45000 // 128, 5000


def macs_per_image(model, device):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.randn(1, 3, 32, 32, device=device), return_attn=True)   # need_weights=True: no fused kernel, so every matmul is counted
    return fc.get_total_flops() / 2 / 1e6                       # MACs in millions (matmul/conv only)


def time_variant(tok, attn, device, steps=40, warm=10, bs=128):
    cfg = Cfg(tokenizer=tok, attn=attn)
    model = TinyTransformer(tok, attn, cfg.dim, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    x, y = torch.randn(bs, 3, 32, 32, device=device), torch.randint(0, 10, (bs,), device=device)
    mmacs = macs_per_image(model, device)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(warm + steps):
        if i == warm and device.type == "cuda":
            torch.cuda.synchronize()
        if i == warm:
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / steps * 1000
    return {"tokenizer": tok, "attn": attn, "ms_per_step": ms, "est_epoch_s": ms * STEPS_PER_EPOCH / 1000,
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad), "tokens": model.n_tokens, "mmacs": mmacs}


def main():
    argparse.ArgumentParser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = env_info()
    print(f"device: {env['gpu']}  torch {env['torch']}")
    rows = [time_variant(t, a, device) for t in ("patch4", "patch8", "row", "cnn") for a in ("scratch", "torch")]
    print(f"{'tokenizer':9s}{'attn':8s}{'tokens':>7s}{'params(M)':>10s}{'MMACs':>8s}{'ms/step':>9s}{'epoch(s)':>9s}{'mem(MB)':>9s}")
    for r in rows:
        print(f"{r['tokenizer']:9s}{r['attn']:8s}{r['tokens']:7d}{r['params'] / 1e6:10.3f}{r['mmacs']:8.0f}{r['ms_per_step']:9.1f}"
              f"{r['est_epoch_s']:9.1f}{(r['peak_mem_mb'] or 0):9.0f}")
    # data pipeline: one epoch of GPU augmentation without the model
    ds = GPUSet(*fake_cifar(n_train=45000)[:2], device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in ds.train_batches(128, 0, 0):
        pass
    if device.type == "cuda":
        torch.cuda.synchronize()
    aug_s = time.time() - t0
    by = {(r["tokenizer"], r["attn"]): r["est_epoch_s"] for r in rows}
    total_s, epochs = 0.0, Cfg().epochs
    for c in all_cfgs():
        total_s += epochs * (by[(c.tokenizer, c.attn)] + aug_s + 3)       # +3 s validation per epoch (rough)
    print(f"\naugmentation alone: {aug_s:.1f} s per epoch")
    print(f"ESTIMATED TOTAL for {len(all_cfgs())} runs x {epochs} epochs: {total_s / 3600:.1f} h  (tune with --epochs only after reading this)")
    write_json(E2_DIR / "results" / f"bench_{(env['gpu'] or 'cpu').replace(' ', '_')}.json", {"env": env, "rows": rows, "aug_s": aug_s,
                                                                                            "total_h": total_s / 3600})


if __name__ == "__main__":
    main()
