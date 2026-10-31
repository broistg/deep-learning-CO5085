"""Pack the weights/logits into GitHub Release assets and write common/release_manifest.json (SHA-256 + size).

    python common/pack_release.py            # packs E1/artifacts and E2/artifacts that exist, updates the manifest, prints the gh command
    python common/pack_release.py --tag weights-v1

What goes in (kept small so that free Colab can download it in a minute):
  E1: E1/artifacts/<run>/model_final.pt, model_best_val_loss.pt   -> e1_artifacts.zip
  E2: E2/artifacts/<run>/logits_test.pt, model_final.pt           -> e2_artifacts.zip
  A1: A1/artifacts/<run>/logits_{val,test}.npz (all runs)         -> a1_logits.zip
      A1/artifacts/<run>/model_final.pt, in three zips (each below GitHub's 2 GiB per-asset limit):
        core (m_*_s0), seeds (m_*_s1, m_*_s2), extra (ablation ab_* and stretch x_*)  -> a1_weights_core.zip, a1_weights_seeds.zip, a1_weights_extra.zip
  `--only a1` packs just the A1 assets (leaves the E1/E2 entries of the manifest untouched).
Entries are stored with repo-relative paths, so `common.notebook_utils.fetch_release` can extract them straight into the repo.
Upload is manual (see README, "Release weights"); the manifest is committed, the zips are not."""
import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.notebook_utils import MANIFEST, sha256_file  # noqa: E402

import re  # noqa: E402

ASSETS = {   # name: (folder, files, optional predicate on the run folder name)
    "e1_artifacts.zip": ("E1/artifacts", ("model_final.pt", "model_best_val_loss.pt")),
    "e2_artifacts.zip": ("E2/artifacts", ("logits_test.pt", "model_final.pt")),
    "a1_logits.zip": ("A1/artifacts", ("logits_val.npz", "logits_test.npz"), lambda n: not n.startswith("sw_")),
    "a1_weights_core.zip": ("A1/artifacts", ("model_final.pt",), lambda n: re.fullmatch(r"m_.*_s0", n) is not None),
    "a1_weights_seeds.zip": ("A1/artifacts", ("model_final.pt",), lambda n: re.fullmatch(r"m_.*_s[12]", n) is not None),
    "a1_weights_extra.zip": ("A1/artifacts", ("model_final.pt",), lambda n: n.startswith(("ab_", "x_"))),
}
OUT = ROOT / "release_assets"   # git-ignored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--repo", default=None, help="owner/name (default: keep the manifest's)")
    ap.add_argument("--only", default=None, help="prefix of the asset names to pack, e.g. a1")
    a = ap.parse_args()
    m = json.loads(MANIFEST.read_text(encoding="utf8")) if MANIFEST.exists() else {"repo": "broistg/deep-learning-CO5085", "tag": "weights-v1", "assets": {}}
    m["tag"] = a.tag or m["tag"]
    m["repo"] = a.repo or m["repo"]
    OUT.mkdir(exist_ok=True)
    packed = []
    for name, spec in ASSETS.items():
        if a.only and not name.startswith(a.only):
            continue
        folder, files = spec[0], spec[1]
        keep = spec[2] if len(spec) > 2 else (lambda n: True)
        base = ROOT / folder
        members = sorted(p for f in files for p in base.glob(f"*/{f}") if keep(p.parent.name))
        if not members:
            print(f"{name}: nothing in {folder}, skipped")
            continue
        zp = OUT / name
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED if files == ("model_final.pt",) else zipfile.ZIP_DEFLATED, compresslevel=6) as z:   # weights do not compress
            for p in members:
                z.write(p, p.relative_to(ROOT).as_posix())
        m["assets"][name] = {"sha256": sha256_file(zp), "bytes": zp.stat().st_size, "files": len(members)}
        packed.append(zp)
        print(f"{name}: {len(members)} files, {zp.stat().st_size / 1e6:.1f} MB, sha256 {m['assets'][name]['sha256'][:16]}...")
    MANIFEST.write_text(json.dumps(m, indent=1) + "\n", encoding="utf8")
    if packed:
        print(f"\nmanifest written: {MANIFEST.relative_to(ROOT)}  (commit it)\nthen create the Release and upload:\n"
              f"  gh release create {m['tag']} {' '.join(str(p.relative_to(ROOT)) for p in packed)} "
              f"--repo {m['repo']} --title \"Trained weights ({m['tag']})\" --notes \"Weights and logits of the E1, E2 and A1 runs; verified by SHA-256 in common/release_manifest.json\"")


if __name__ == "__main__":
    main()
