"""Architecture cost benchmark on ONE GPU in ONE session (what the report's cost tables and the Pareto figure use).

    python -m a1lib.cost_bench [--out A1/results] [--models resnet50,deit_s,convnext_t,swin_t]

Training (GPU only, no data loader): synthetic batch of 64 already on the GPU, AMP fp16, AdamW on the trainable parameters, channels_last for
CNNs; seconds per epoch = 70,700 training images / measured images per second. For each architecture and each of the four modes.
Inference: AMP fp16, torch.inference_mode, CUDA synchronised around every call; batch 1 (median and p95 latency, peak memory) and batch 128
(throughput = images / median time, peak memory); GMACs from fvcore at 224x224 (one multiply-add = one MAC).
Writes cost_gpu.json ("<model>|<mode>": seconds/epoch), cost_train.json (img/s, peak MB), cost_infer.json (+ the GPU and library versions)."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from a1lib.config import MODES
from a1lib.models import FAMILY, apply_freeze, count_params, make_model, param_groups
from a1lib.plan import N_TRAIN

A1 = Path(__file__).resolve().parents[1]
TIMM = {"resnet50": "resnet50.a1_in1k", "deit_s": "deit_small_patch16_224.fb_in1k", "convnext_t": "convnext_tiny.fb_in1k",
        "swin_t": "swin_tiny_patch4_window7_224.ms_in1k"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _sync():
    torch.cuda.synchronize()


def bench_train(name, mode, iters=40, warm=10, bs=64):
    model = make_model(TIMM[name], False)
    apply_freeze(model, mode)
    cl = FAMILY[TIMM[name]] == "cnn"
    model.cuda()
    if cl:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.AdamW(param_groups(model, 0.05), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(bs, 3, 224, 224, device="cuda")
    if cl:
        x = x.contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 101, (bs,), device="cuda")
    torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    for _ in range(warm):
        step()
    _sync()
    t = time.perf_counter()
    for _ in range(iters):
        step()
    _sync()
    ips = iters * bs / (time.perf_counter() - t)
    mem = torch.cuda.max_memory_allocated() / 2**20
    tr, tot = count_params(model)
    del model, opt
    torch.cuda.empty_cache()
    return {"img_per_s": ips, "s_per_epoch": N_TRAIN / ips, "peak_mem_mb": mem, "trainable_M": tr / 1e6, "total_M": tot / 1e6}


@torch.inference_mode()
def bench_infer(name):
    model = make_model(TIMM[name], False).cuda().eval()
    cl = FAMILY[TIMM[name]] == "cnn"
    if cl:
        model = model.to(memory_format=torch.channels_last)
    out = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    try:
        from fvcore.nn import FlopCountAnalysis
        fa = FlopCountAnalysis(model.float(), torch.randn(1, 3, 224, 224, device="cuda"))
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        out["gmacs"] = fa.total() / 1e9
    except Exception as e:  # noqa: BLE001
        out["gmacs"] = None
        log("fvcore failed:", repr(e)[:120])

    def run(bs, warm, iters):
        x = torch.randn(bs, 3, 224, 224, device="cuda")
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for i in range(warm + iters):
            _sync()
            t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                model(x)
            _sync()
            if i >= warm:
                ts.append(time.perf_counter() - t)
        return ts, torch.cuda.max_memory_allocated() / 2**20
    ts, m1 = run(1, 50, 300)
    ts = sorted(ts)
    out.update({"lat1_median_ms": 1e3 * statistics.median(ts), "lat1_p95_ms": 1e3 * ts[int(0.95 * len(ts)) - 1], "mem_b1_mb": m1})
    ts, m128 = run(128, 10, 40)
    out.update({"ips_b128": 128 / statistics.median(ts), "mem_b128_mb": m128})
    del model
    torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(A1 / "results"))
    ap.add_argument("--models", default="resnet50,deit_s,convnext_t,swin_t")
    a = ap.parse_args(argv)
    assert torch.cuda.is_available(), "needs a GPU"
    import timm
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "timm": timm.__version__, "date": time.strftime("%Y-%m-%d %H:%M")}
    gpu, train, infer = {}, {}, {"env": env}
    for name in a.models.split(","):
        for mode in MODES:
            r = bench_train(name, mode)
            train[f"{name}|{mode}"] = r
            gpu[f"{name}|{mode}"] = round(r["s_per_epoch"], 1)
            log(f"train {name:10s} {mode:8s} {r['img_per_s']:7.1f} img/s  {r['s_per_epoch']:6.1f} s/epoch  {r['peak_mem_mb']:6.0f} MB")
        infer[name] = bench_infer(name)
        log(f"infer {name:10s}", {k: round(v, 2) if isinstance(v, float) else v for k, v in infer[name].items()})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cost_gpu.json").write_text(json.dumps(gpu, indent=1), encoding="utf8")
    (out / "cost_train.json").write_text(json.dumps({"env": env, **train}, indent=1), encoding="utf8")
    (out / "cost_infer.json").write_text(json.dumps(infer, indent=1), encoding="utf8")
    log("wrote cost_gpu.json, cost_train.json, cost_infer.json")


if __name__ == "__main__":
    main()
