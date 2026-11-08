"""Run or list the 4 recurrent E3 configurations across seeds 0, 1, and 2.

From the repository root: PYTHONPATH=.:E3 python -m e3lib.plan --list
"""

import argparse

from .train import ARTIFACTS_DIR, RESULTS_DIR, Cfg, run


def all_cfgs(hidden_size=128):
    configs = []
    for seed in (0, 1, 2):
        for model in ("lstm", "gru"):
            for tokenizer in ("row", "patch4"):
                configs.append(Cfg(model=model, tokenizer=tokenizer, seed=seed,
                                   hidden_size=hidden_size))
    return configs


def status(cfg, results_dir=RESULTS_DIR, artifacts_dir=ARTIFACTS_DIR):
    result = results_dir / f"{cfg.run_id}.json"
    checkpoint = artifacts_dir / cfg.run_id / "ckpt" / "last.pt"
    if result.exists():
        return "result"  # run() verifies the result and artifacts before skipping
    if checkpoint.exists():
        return "checkpoint"
    if (artifacts_dir / cfg.run_id).exists():
        return "incomplete"
    return "todo"


def main(argv=None):
    parser = argparse.ArgumentParser(description="List or run the 12 E3 recurrent experiments")
    parser.add_argument("--list", action="store_true", help="show files without loading CIFAR-10")
    parser.add_argument("--only", help="run ID prefix, for example gru_patch4")
    args = parser.parse_args(argv)
    for cfg in all_cfgs():
        if args.only and not cfg.run_id.startswith(args.only):
            continue
        if args.list:
            print(f"{status(cfg):10s} {cfg.run_id}")
        else:
            run(cfg)


if __name__ == "__main__":
    main()
