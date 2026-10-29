"""The experiment plan: which runs exist, who owns them, in what order. Pure Python (testable without torch).

    python -m a1lib.plan make --owners dat,ly      # writes A1/configs/*.yaml and A1/plan.json   (coordinator, once; commit both)
    python -m a1lib.plan show                      # hours per owner and stage
    python -m a1lib.plan choose-lr                 # after BOTH members finished the sweep runs: picks LRs, writes lr_choice.json
    python -m a1lib.plan extend --model M --group G --lr L   # edge rule: add one more LR level to the sweep
"""
import argparse
import json
from pathlib import Path

from common.repro import read_json, write_json

from a1lib.config import RunConfig, dump_config, lr_group, validate

N_TRAIN, N_VAL, VAL_S = 70_700, 5_050, 4.7  # images, images, validation seconds per epoch (val loader 1068 img/s)

PAIR1 = ["resnet50.a1_in1k", "deit_small_patch16_224.fb_in1k"]
PAIR2 = ["convnext_tiny.fb_in1k", "swin_tiny_patch4_window7_224.ms_in1k"]
SHORT = {"resnet50.a1_in1k": "resnet50", "deit_small_patch16_224.fb_in1k": "deit_s",
         "convnext_tiny.fb_in1k": "convnext_t", "swin_tiny_patch4_window7_224.ms_in1k": "swin_t"}
# measured end-to-end images/s on a Colab T4 (bench2.py): head / partial / full (scratch = full)
E2E_IPS = {"resnet50.a1_in1k": (338.0, 336.6, 281.7), "deit_small_patch16_224.fb_in1k": (334.3, 309.6, 275.2),
           "convnext_tiny.fb_in1k": (337.0, 319.9, 223.9), "swin_tiny_patch4_window7_224.ms_in1k": (308.4, 304.3, 170.9)}
LR_GRID = {"head": [3e-4, 1e-3, 3e-3], "finetune": [3e-5, 1e-4, 3e-4], "scratch": [1e-4, 3e-4, 1e-3]}
SWEEP_MODE = {"head": "head", "finetune": "full", "scratch": "scratch"}   # the sweep trains with these modes
PRETRAINED_EPOCHS, SCRATCH_EPOCHS, SWEEP_EPOCHS, SWEEP_FRACTION = 10, 40, 3, 0.2
FRACTIONS = (0.1, 0.25, 0.5)
STAGES = {"sweep": 0, "main": 1, "seed": 2, "ablation": 3, "stretch": 4}


def lr_tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def est_hours(model, mode, epochs, fraction=1.0):
    ips = E2E_IPS[model][{"head": 0, "partial": 1, "full": 2, "scratch": 2}[mode]]
    return epochs * (N_TRAIN * fraction / ips + VAL_S) / 3600


def _cfg(run_id, model, mode, epochs, kind, seed=0, fraction=1.0, lr="auto", notes=""):
    return validate(RunConfig(run_id=run_id, model=model, mode=mode, epochs=epochs, seed=seed, kind=kind, pretrained=(mode != "scratch"),
                              data_fraction=fraction, lr_peak=lr, notes=notes))


def sweep_run(model, group, lr):
    mode = SWEEP_MODE[group]
    rid = f"sw_{SHORT[model]}_{group}_lr{lr_tag(lr)}"
    return _cfg(rid, model, mode, SWEEP_EPOCHS, "sweep", 0, SWEEP_FRACTION, lr,
                notes=f"LR sweep for group '{group}' (3 epochs, 20% of train, seed 0, val only)")


def build_runs():
    """Every run of the protocol. Order inside a stage is the order they should be worked on."""
    runs = []
    for m in PAIR1 + PAIR2:  # sweeps for pair 2 are only needed if pair 2 is run; they are cheap, so created up front
        for g, grid in LR_GRID.items():
            for lr in grid:
                runs.append(sweep_run(m, g, lr))
    for m in PAIR1:  # core: 4 configs per model, seed 0
        for mode, ep in (("head", PRETRAINED_EPOCHS), ("partial", PRETRAINED_EPOCHS), ("full", PRETRAINED_EPOCHS), ("scratch", SCRATCH_EPOCHS)):
            runs.append(_cfg(f"m_{SHORT[m]}_{mode}_s0", m, mode, ep, "main"))
    for seed in (1, 2):  # 2 extra seeds on the 6 pretrained pair-1 configs
        for m in PAIR1:
            for mode in ("head", "partial", "full"):
                runs.append(_cfg(f"m_{SHORT[m]}_{mode}_s{seed}", m, mode, PRETRAINED_EPOCHS, "seed", seed))
    for fr in FRACTIONS:  # data-scale ablation (LR taken from the 100% run's group)
        for m in PAIR1:
            runs.append(_cfg(f"ab_{SHORT[m]}_scratch_f{int(fr * 100)}", m, "scratch", SCRATCH_EPOCHS, "ablation", 0, fr))
            runs.append(_cfg(f"ab_{SHORT[m]}_full_f{int(fr * 100)}", m, "full", PRETRAINED_EPOCHS, "ablation", 0, fr))
    for m in PAIR2:  # stretch 1: pair 2 pretrained
        for mode in ("head", "partial", "full"):
            runs.append(_cfg(f"x_{SHORT[m]}_{mode}_s0", m, mode, PRETRAINED_EPOCHS, "stretch"))
    runs.append(_cfg("x_deit_s_scratch_e100", PAIR1[1], "scratch", 100, "stretch",  # stretch 2: ViT given a long schedule
                     notes="asymmetric control: does the ViT gap close with 100 epochs?"))
    for m in PAIR2:  # stretch 3
        runs.append(_cfg(f"x_{SHORT[m]}_scratch_s0", m, "scratch", SCRATCH_EPOCHS, "stretch"))
    return runs


def assign(runs, owners):
    """Per stage, longest-first onto the owner with the least accumulated load (loads carry across stages)."""
    load = {o: 0.0 for o in owners}
    out = []
    for stage in sorted(set(STAGES[r.kind] for r in runs)):
        group = [r for r in runs if STAGES[r.kind] == stage]
        hours = {r.run_id: est_hours(r.model, r.mode, r.epochs, r.data_fraction) for r in group}
        for r in sorted(group, key=lambda r: (-hours[r.run_id], r.run_id)):
            o = min(owners, key=lambda o: (load[o], o))
            load[o] += hours[r.run_id]
            out.append({"run_id": r.run_id, "stage": stage, "kind": r.kind, "owner": o, "est_hours": round(hours[r.run_id], 3),
                        "config": f"configs/{r.run_id}.yaml", "needs_lr": r.lr_peak == "auto"})
    return sorted(out, key=lambda e: (e["stage"], e["run_id"]))


def summarize_plan(plan):
    tab = {}
    for e in plan["runs"]:
        tab.setdefault(e["owner"], {}).setdefault(e["kind"], 0.0)
        tab[e["owner"]][e["kind"]] += e["est_hours"]
    return tab


def choose_lr(rows, grid_original=None, accept_edge_after_extension=True):
    """rows: list of {"lr", "val_top1", "val_nll"} for ONE (model, group). Returns dict(lr, edge, extend_to|None, ...).
    Rule (protocol): highest val top-1 at the final sweep epoch; ties -> lower val NLL. If the best LR is at the edge of the grid,
    extend ONE step (x3 or /3) and rerun that level; after one extension the best is accepted even if it is at the new edge."""
    rows = sorted(rows, key=lambda r: r["lr"])
    best = sorted(rows, key=lambda r: (-r["val_top1"], r["val_nll"], r["lr"]))[0]
    lrs = [r["lr"] for r in rows]
    n_orig = len(grid_original) if grid_original else 3
    extended = len(rows) > n_orig
    at_low, at_high = best["lr"] == lrs[0], best["lr"] == lrs[-1]
    edge = at_low or at_high
    ext = None
    if edge and not (extended and accept_edge_after_extension):
        ext = best["lr"] / 3 if at_low else best["lr"] * 3
    return {"lr": best["lr"], "val_top1": best["val_top1"], "val_nll": best["val_nll"], "grid": lrs, "at_edge": edge,
            "extend_to": ext, "extended": extended}


def collect_sweep_rows(shared_dir):
    """{(model, group): [rows]} from the result.json files of finished sweep runs."""
    out = {}
    for p in sorted((Path(shared_dir) / "runs").glob("sw_*/result.json")):
        r = read_json(p)
        if not r or r.get("kind") != "sweep" or r.get("smoke"):
            continue
        cfg = r["config"]
        g = lr_group(cfg["mode"])
        v = r["final"]["val"]
        out.setdefault((cfg["model"], g), []).append({"lr": float(cfg["lr_peak"]), "val_top1": v["top1"], "val_nll": v["nll"],
                                                      "run_id": r["run_id"]})
    return out


def sweep_complete(plan, rows_by_key, model, group):
    ids = [e["run_id"] for e in plan["runs"] if e["kind"] == "sweep" and e["run_id"].startswith(f"sw_{SHORT[model]}_{group}_")]
    done = {r["run_id"] for r in rows_by_key.get((model, group), [])}
    return bool(ids) and set(ids) <= done


def load_plan(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _paths_args(ap):
    ap.add_argument("--a1-dir", default="A1", help="folder containing configs/ and plan.json (repo checkout)")


def main():
    from a1lib.paths import Paths
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    _paths_args(m)
    m.add_argument("--owners", required=True, help="comma list, e.g. dat,ly")
    s = sub.add_parser("show")
    _paths_args(s)
    c = sub.add_parser("choose-lr")
    _paths_args(c)
    c.add_argument("--models", default="pair1", choices=["pair1", "all"])
    e = sub.add_parser("extend")
    _paths_args(e)
    e.add_argument("--model", required=True)
    e.add_argument("--group", required=True, choices=list(LR_GRID))
    e.add_argument("--lr", required=True, type=float)
    a = ap.parse_args()
    a1 = Path(a.a1_dir)
    plan_path = a1 / "plan.json"

    if a.cmd == "make":
        owners = [o.strip() for o in a.owners.split(",") if o.strip()]
        runs = build_runs()
        for r in runs:
            dump_config(r, a1 / "configs" / f"{r.run_id}.yaml")
        plan = {"version": 1, "owners": owners, "grids": LR_GRID, "runs": assign(runs, owners)}
        write_json(plan_path, plan)
        print(f"wrote {len(runs)} configs and {plan_path}")
    plan = load_plan(plan_path)
    if a.cmd in ("make", "show"):
        tab = summarize_plan(plan)
        total = sum(sum(v.values()) for v in tab.values())
        print(f"estimated GPU-hours (T4, end-to-end): total {total:.1f}")
        for o, kinds in tab.items():
            print(f"  {o:10s} " + "  ".join(f"{k}={h:.1f}" for k, h in sorted(kinds.items(), key=lambda kv: STAGES[kv[0]])) + f"   | sum {sum(kinds.values()):.1f} h")
    elif a.cmd == "choose-lr":
        paths = Paths.from_env()
        rows = collect_sweep_rows(paths.shared)
        models = PAIR1 if a.models == "pair1" else PAIR1 + PAIR2
        chosen = read_json(paths.lr_choice, {}) or {}
        for model in models:
            for group, grid in LR_GRID.items():
                key = f"{model}|{group}"
                if not sweep_complete(plan, rows, model, group):
                    print(f"  {key:60s} sweep not finished yet")
                    continue
                res = choose_lr(rows[(model, group)], grid)
                tag = " (EDGE -> extend)" if res["extend_to"] else ""
                print(f"  {key:60s} best lr {res['lr']:g}  val top1 {100 * res['val_top1']:.2f}%  grid {[f'{x:g}' for x in res['grid']]}{tag}")
                if res["extend_to"]:
                    print(f"      next: python -m a1lib.plan extend --model {model} --group {group} --lr {res['extend_to']:.3g}")
                    chosen.pop(key, None)
                else:
                    chosen[key] = res
        write_json(paths.lr_choice, chosen)
        write_json(a1 / "results" / "lr_choice.json", chosen)
        print(f"wrote {paths.lr_choice} and {a1 / 'results' / 'lr_choice.json'} (commit the latter)")
    elif a.cmd == "extend":
        run = sweep_run(a.model, a.group, a.lr)
        dump_config(run, a1 / "configs" / f"{run.run_id}.yaml")
        owners = plan["owners"]
        load = {o: sum(x["est_hours"] for x in plan["runs"] if x["owner"] == o and x["stage"] == 0) for o in owners}
        o = min(owners, key=lambda o: (load[o], o))
        plan["runs"].append({"run_id": run.run_id, "stage": 0, "kind": "sweep", "owner": o,
                             "est_hours": round(est_hours(a.model, run.mode, run.epochs, run.data_fraction), 3),
                             "config": f"configs/{run.run_id}.yaml", "needs_lr": False})
        plan["runs"].sort(key=lambda e: (e["stage"], e["run_id"]))
        write_json(plan_path, plan)
        print(f"added {run.run_id} (owner {o}); commit configs/ and plan.json, then both members `git pull`")


if __name__ == "__main__":
    main()
