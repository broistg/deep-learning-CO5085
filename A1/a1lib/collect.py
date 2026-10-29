"""Copy the small result files of finished runs from the shared folder into the repo, and build the Release bundle.

    python -m a1lib.collect results           # shared/runs/*/result.json -> A1/results/<run_id>.json   (commit these)
    python -m a1lib.collect bundle --out release/   # zip of logits + final weights + SHA256SUMS.txt     (upload to a GitHub Release)
"""
import argparse
import shutil
import zipfile
from pathlib import Path

from common.repro import read_json, sha256_file, write_json
from a1lib.paths import Paths


def collect_results(paths, a1_dir):
    out = Path(a1_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((paths.shared / "runs").glob("*/result.json")):
        r = read_json(p)
        if not r or r.get("smoke"):
            continue
        write_json(out / f"{p.parent.name}.json", r)
        n += 1
    print(f"copied {n} result files to {out}")


def bundle(paths, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sums = []
    for kind, pat in (("logits", "logits_*.pt"), ("weights", "model_final.pt")):
        zp = out / f"a1_{kind}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for p in sorted((paths.shared / "runs").glob(f"*/{pat}")):
                if p.parent.name.startswith("sw_"):
                    continue
                z.write(p, f"{p.parent.name}/{p.name}")
        sums.append(f"{sha256_file(zp)}  {zp.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["results", "bundle"])
    ap.add_argument("--a1-dir", default="A1")
    ap.add_argument("--out", default="release")
    a = ap.parse_args()
    paths = Paths.from_env()
    collect_results(paths, a.a1_dir) if a.cmd == "results" else bundle(paths, a.out)


if __name__ == "__main__":
    main()
