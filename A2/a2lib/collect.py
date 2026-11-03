"""Copy the small result files of finished A2 runs from the store into the repo (commit them), fetch the confusion matrices, and
optionally download the final weights for the Release.

    python -m a2lib.collect results [--a2-dir A2]      # store runs/*/result.json (+ confusion_*.npy) -> A2/results/runs/<run_id>/
    python -m a2lib.collect weights --out release/    # model_final.pt of every full run -> release/a2_weights/<run_id>.pt + SHA256SUMS.txt
"""
import argparse
from pathlib import Path

from common.repro import read_json, sha256_file, write_json

from a2lib.paths import Paths


def collect_results(paths, a2_dir):
    paths.store.refresh()
    out = Path(a2_dir) / "results" / "runs"
    n = 0
    for p in sorted((paths.shared / "runs").glob("a2_*/result.json")):
        r = read_json(p)
        if not r or r.get("smoke") or r.get("fake"):
            continue
        d = out / p.parent.name
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "result.json", r)
        for name in ("confusion_dev.npy", "confusion_test.npy"):
            rel = f"runs/{p.parent.name}/{name}"
            if not (d / name).exists() and any(f == rel for f, _ in paths.store.list_files(f"runs/{p.parent.name}/")):
                paths.store.get(rel, d / name)
        n += 1
    print(f"copied {n} runs to {out} (commit A2/results/)")


def collect_weights(paths, out_dir):
    out = Path(out_dir) / "a2_weights"
    out.mkdir(parents=True, exist_ok=True)
    sums = []
    for rel, size in paths.store.list_files("runs/"):
        if rel.endswith("/model_final.pt") and "/a2_" in "/" + rel:
            rid = rel.split("/")[1]
            dst = out / f"{rid}.pt"
            if not dst.exists():
                paths.store.get(rel, dst)
            sums.append(f"{sha256_file(dst)}  a2_weights/{dst.name}")
    (Path(out_dir) / "SHA256SUMS_a2.txt").write_text("\n".join(sums) + "\n")
    print(f"{len(sums)} weight files in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["results", "weights"])
    ap.add_argument("--a2-dir", default="A2")
    ap.add_argument("--out", default="release")
    a = ap.parse_args()
    paths = Paths.from_env()
    collect_results(paths, a.a2_dir) if a.cmd == "results" else collect_weights(paths, a.out)


if __name__ == "__main__":
    main()
