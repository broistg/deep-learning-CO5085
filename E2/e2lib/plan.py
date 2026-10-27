"""All E2 runs: 4 tokenizers x 2 attention implementations x 3 seeds (24) + positional-encoding ablation (scratch, patch4 and row, 3 seeds: 6).
    python -m e2lib.plan            # run everything not finished yet, seed 0 first
    python -m e2lib.plan --list"""
import argparse

from .train import Cfg, E2_DIR, run

TOKENIZERS = ("patch4", "patch8", "row", "cnn")


def all_cfgs():
    out = []
    for seed in (0, 1, 2):
        for tok in TOKENIZERS:
            for attn in ("scratch", "torch"):
                out.append(Cfg(tokenizer=tok, attn=attn, seed=seed))
        for tok in ("patch4", "row"):
            out.append(Cfg(tokenizer=tok, attn="scratch", use_pos=False, seed=seed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="run_id prefix")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    for cfg in all_cfgs():
        if a.only and not cfg.run_id.startswith(a.only):
            continue
        done = (E2_DIR / "results" / f"{cfg.run_id}.json").exists()
        if a.list:
            print(("done " if done else "todo ") + cfg.run_id)
        else:
            run(cfg, data_root=a.data_root)


if __name__ == "__main__":
    main()
