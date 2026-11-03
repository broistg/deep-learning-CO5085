"""The A2 experiment plan: which runs exist, who owns them, in what order. Pure Python (testable without torch).

    python -m a2lib.plan make --owners dat,ly      # writes A2/configs/*.yaml and A2/plan.json   (coordinator, once; commit both)
    python -m a2lib.plan show                      # hours per owner and stage
    python -m a2lib.plan choose-lr                 # after the sweep runs finished: picks LRs by dev mIoU, writes lr_choice_a2.json to the store
    python -m a2lib.plan extend --model M --init pretrained|scratch --lr L   # edge rule: add one more LR level to the sweep

Protocol (A2/README.md): LR sweep 3 epochs on 20% of train, seed 0, selection on the dev set (best dev mIoU, tie -> higher
pixel accuracy, edge -> extend one step); then 20 epochs (24,000 iterations) at batch 16 for every full run."""
import argparse
import json
from pathlib import Path

from common.repro import read_json, write_json

from a1lib.plan import choose_lr          # pure function: rows of {lr, val_top1, val_nll}

from a2lib.config import dump_config, make_config

MODELS = ["segformer_b0", "deeplabv3p_mnv2"]
SHORT = {"segformer_b0": "sf", "deeplabv3p_mnv2": "dl"}
INITS = ["pretrained", "scratch"]
LR_GRID = {"segformer_b0|pretrained": [3e-5, 6e-5, 1.2e-4], "segformer_b0|scratch": [2e-4, 6e-4, 1.8e-3],
           "deeplabv3p_mnv2|pretrained": [0.005, 0.01, 0.02], "deeplabv3p_mnv2|scratch": [0.01, 0.03, 0.1]}
EPOCHS, SWEEP_EPOCHS, SWEEP_FRACTION = 20, 3, 0.2
# measured on a Colab T4 (our benchmark): hours per full epoch (1,200 iterations at batch 16); dev evaluation adds ~0.01 h
H_PER_EPOCH = {"segformer_b0": 0.28, "deeplabv3p_mnv2": 0.42}
EVAL_H = 0.01
STAGE = {"sweep": 0, "pre0": 1, "scratch": 2, "seeds": 3}


def lr_tag(lr):
    return f"{lr:.1e}".replace(".0e", "e").replace("e-0", "e-").replace("e+0", "e+")


def est_hours(model, epochs, fraction=1.0):
    return epochs * (H_PER_EPOCH[model] * fraction + EVAL_H)


def sweep_run(model, init, lr, split_sha=""):
    rid = f"a2_sw_{SHORT[model]}_{init}_lr{lr_tag(lr)}"
    return make_config(rid, model, init == "pretrained", SWEEP_EPOCHS, "sweep", 0, SWEEP_FRACTION, lr, split_sha=split_sha,
                       notes=f"LR sweep ({init}): {SWEEP_EPOCHS} epochs on {int(SWEEP_FRACTION * 100)}% of train, seed 0, dev set only")


def main_run(model, init, seed, split_sha=""):
    rid = f"a2_m_{SHORT[model]}_{init}_s{seed}"
    kind = "main" if seed == 0 else "seed"
    return make_config(rid, model, init == "pretrained", EPOCHS, kind, seed, 1.0, "auto", split_sha=split_sha)


def build_runs(split_sha=""):
    """[(config, stage)] in the order they should be worked on."""
    runs = []
    for key, grid in LR_GRID.items():
        m, init = key.split("|")
        for lr in grid:
            runs.append((sweep_run(m, init, lr, split_sha), STAGE["sweep"]))
    for m in MODELS:
        runs.append((main_run(m, "pretrained", 0, split_sha), STAGE["pre0"]))
    for m in MODELS:
        runs.append((main_run(m, "scratch", 0, split_sha), STAGE["scratch"]))
    for s in (1, 2):
        for m in MODELS:
            runs.append((main_run(m, "pretrained", s, split_sha), STAGE["seeds"]))
    return runs


def entry(cfg, stage, owner):
    return {"run_id": cfg.run_id, "stage": stage, "kind": cfg.kind, "owner": owner,
            "est_hours": round(est_hours(cfg.model, cfg.epochs, cfg.data_fraction), 3), "config": f"configs/{cfg.run_id}.yaml",
            "needs_lr": cfg.lr_peak == "auto"}


def assign(runs, owners):
    """Per stage, longest-first onto the owner with the least accumulated load (loads carry across stages)."""
    load = {o: 0.0 for o in owners}
    out = []
    for stage in sorted({s for _, s in runs}):
        group = [(c, s) for c, s in runs if s == stage]
        for c, s in sorted(group, key=lambda cs: (-est_hours(cs[0].model, cs[0].epochs, cs[0].data_fraction), cs[0].run_id)):
            o = min(owners, key=lambda o: (load[o], o))
            load[o] += est_hours(c.model, c.epochs, c.data_fraction)
            out.append(entry(c, s, o))
    return sorted(out, key=lambda e: (e["stage"], e["run_id"]))


def summarize_plan(plan):
    tab = {}
    for e in plan["runs"]:
        tab.setdefault(e["owner"], {}).setdefault(e["kind"], 0.0)
        tab[e["owner"]][e["kind"]] += e["est_hours"]
    return tab


def collect_sweep_rows(shared_dir):
    """{(model, init): [rows]} from the result.json files of finished sweep runs (dev mIoU; tie-break = higher dev pixel accuracy)."""
    out = {}
    for p in sorted((Path(shared_dir) / "runs").glob("a2_sw_*/result.json")):
        r = read_json(p)
        if not r or r.get("kind") != "sweep" or r.get("smoke"):
            continue
        cfg, v = r["config"], r["final"]["dev"]
        out.setdefault((cfg["model"], "pretrained" if cfg["pretrained"] else "scratch"), []).append(
            {"lr": float(cfg["lr_peak"]), "val_top1": v["mIoU"], "val_nll": -v["pixel_acc"], "run_id": r["run_id"]})
    return out


def sweep_complete(plan, rows_by_key, model, init):
    ids = [e["run_id"] for e in plan["runs"] if e["kind"] == "sweep" and e["run_id"].startswith(f"a2_sw_{SHORT[model]}_{init}_")]
    done = {r["run_id"] for r in rows_by_key.get((model, init), [])}
    return bool(ids) and set(ids) <= done


def load_plan(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def split_sha():
    from a2lib.data import load_split
    return load_split(Path(__file__).resolve().parents[1] / "split.json")["sha256_16"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make", "show", "choose-lr", "extend"):
        p = sub.add_parser(name)
        p.add_argument("--a2-dir", default="A2", help="folder containing configs/ and plan.json (repo checkout)")
        if name == "make":
            p.add_argument("--owners", required=True, help="comma list, e.g. dat,ly")
        if name == "extend":
            p.add_argument("--model", required=True, choices=MODELS)
            p.add_argument("--init", required=True, choices=INITS)
            p.add_argument("--lr", required=True, type=float)
    a = ap.parse_args()
    d = Path(a.a2_dir)
    plan_path = d / "plan.json"
    if a.cmd == "make":
        owners = [o.strip() for o in a.owners.split(",") if o.strip()]
        runs = build_runs(split_sha())
        for c, _ in runs:
            dump_config(c, d / "configs" / f"{c.run_id}.yaml")
        write_json(plan_path, {"version": 1, "owners": owners, "grids": LR_GRID, "runs": assign(runs, owners)})
        print(f"wrote {len(runs)} configs and {plan_path}")
    plan = load_plan(plan_path)
    if a.cmd in ("make", "show"):
        tab = summarize_plan(plan)
        print(f"estimated GPU-hours (Colab-T4 timings): total {sum(sum(v.values()) for v in tab.values()):.1f}")
        for o, kinds in tab.items():
            print(f"  {o:10s} " + "  ".join(f"{k}={h:.1f}" for k, h in sorted(kinds.items())) + f"   | sum {sum(kinds.values()):.1f} h")
    elif a.cmd == "choose-lr":
        from a2lib.paths import Paths
        paths = Paths.from_env()
        paths.store.refresh()
        rows = collect_sweep_rows(paths.shared)
        chosen = paths.store.read_json("lr_choice_a2.json", {}, fresh=True) or {}
        for key, grid in LR_GRID.items():
            m, init = key.split("|")
            if not sweep_complete(plan, rows, m, init):
                print(f"  {key:34s} sweep not finished yet")
                continue
            res = choose_lr(rows[(m, init)], grid)
            tag = " (EDGE -> extend)" if res["extend_to"] else ""
            print(f"  {key:34s} best lr {res['lr']:g}  dev mIoU {100 * res['val_top1']:.2f}%  grid {[f'{x:g}' for x in res['grid']]}{tag}")
            if res["extend_to"]:
                print(f"      next: python -m a2lib.plan extend --model {m} --init {init} --lr {res['extend_to']:.3g}")
                chosen.pop(key, None)
            else:
                chosen[key] = res
        paths.store.put({"lr_choice_a2.json": json.dumps(chosen, indent=1)}, "lr_choice_a2")
        write_json(d / "results" / "lr_choice_a2.json", chosen)
        print(f"wrote lr_choice_a2.json to the store {paths.store.describe()} and {d / 'results' / 'lr_choice_a2.json'} (commit the latter)")
    elif a.cmd == "extend":
        run = sweep_run(a.model, a.init, a.lr, split_sha())
        dump_config(run, d / "configs" / f"{run.run_id}.yaml")
        owners = plan["owners"]
        load = {o: sum(x["est_hours"] for x in plan["runs"] if x["owner"] == o and x["stage"] == 0) for o in owners}
        o = min(owners, key=lambda o: (load[o], o))
        plan["runs"].append(entry(run, 0, o))
        plan["runs"].sort(key=lambda e: (e["stage"], e["run_id"]))
        write_json(plan_path, plan)
        print(f"added {run.run_id} (owner {o}); commit configs/ and plan.json, then both members `git pull`")


if __name__ == "__main__":
    main()
