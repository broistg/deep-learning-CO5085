"""Copy the small result files of finished runs from the store into the repo, and build the Release bundle.

    python -m a1lib.collect results           # store runs/*/result.json -> A1/results/<run_id>.json   (commit these)
    python -m a1lib.collect bundle --out release/   # zip of logits + final weights + SHA256SUMS.txt     (store -> local zips)
    python -m a1lib.collect artifacts --bundle release/ --out A1/artifacts
        # unpack those zips into A1/artifacts/<run_id>/ and convert the logits to compact .npz (fp16 + labels; needs torch once).
        # Then `python common/pack_release.py --only a1` makes the Release assets (a1_logits.zip, a1_weights_*.zip) and the manifest.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

from common.repro import read_json, sha256_file, write_json
from a1lib.paths import Paths


def collect_results(paths, a1_dir):
    paths.store.refresh()
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
    tmp = out / "_dl"
    files = [(rel, s) for rel, s in paths.store.list_files("runs/") if rel.endswith(("/logits_val.pt", "/logits_test.pt", "/model_final.pt"))]
    for kind, pat in (("logits", "/logits_"), ("weights", "/model_final.pt")):
        zp = out / f"a1_{kind}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
            for rel, _ in files:
                rid, name = rel.split("/")[1], rel.split("/")[2]
                if rid.startswith("sw_") or pat not in "/" + name:
                    continue
                z.write(paths.store.get(rel, tmp / rid / name), f"{rid}/{name}")
                (tmp / rid / name).unlink()
        sums.append(f"{sha256_file(zp)}  {zp.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))


def artifacts(bundle_dir, out_dir):
    """Unpack release/a1_logits.zip and a1_weights.zip into out_dir/<run_id>/; logits_*.pt -> logits_*.npz (float16 logits + int labels)."""
    import io

    import numpy as np
    import torch
    out = Path(out_dir)
    n_l = n_w = 0
    with zipfile.ZipFile(Path(bundle_dir) / "a1_logits.zip") as z:
        for name in z.namelist():
            rid, fn = name.split("/")
            d = torch.load(io.BytesIO(z.read(name)), map_location="cpu", weights_only=False)
            dst = out / rid / fn.replace(".pt", ".npz")
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, logits=d["logits"].numpy().astype(np.float16), labels=d["labels"].numpy().astype(np.int64))
            n_l += 1
    wz = Path(bundle_dir) / "a1_weights.zip"
    if wz.exists():
        with zipfile.ZipFile(wz) as z:
            for name in z.namelist():
                rid, fn = name.split("/")
                (out / rid).mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(out / rid / fn, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n_w += 1
    print(f"{n_l} logits files -> .npz, {n_w} weight files unpacked into {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["results", "bundle", "artifacts"])
    ap.add_argument("--bundle", default="release", help="artifacts: folder with a1_logits.zip / a1_weights.zip")
    ap.add_argument("--a1-dir", default="A1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "artifacts":
        return artifacts(a.bundle, a.out or str(Path(a.a1_dir) / "artifacts"))
    paths = Paths.from_env()
    collect_results(paths, a.a1_dir) if a.cmd == "results" else bundle(paths, a.out or "release")


if __name__ == "__main__":
    main()
