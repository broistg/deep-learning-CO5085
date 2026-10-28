"""Pack the weights/logits into GitHub Release assets and write common/release_manifest.json (SHA-256 + size).

    python common/pack_release.py            # packs E1/artifacts and E2/artifacts that exist, updates the manifest, prints the gh command
    python common/pack_release.py --tag weights-v1

What goes in (kept small so that free Colab can download it in a minute):
  E1: E1/artifacts/<run>/model_final.pt, model_best_val_loss.pt   -> e1_artifacts.zip
  E2: E2/artifacts/<run>/logits_test.pt, model_final.pt           -> e2_artifacts.zip
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

ASSETS = {
    "e1_artifacts.zip": ("E1/artifacts", ("model_final.pt", "model_best_val_loss.pt")),
    "e2_artifacts.zip": ("E2/artifacts", ("logits_test.pt", "model_final.pt")),
}
OUT = ROOT / "release_assets"   # git-ignored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--repo", default=None, help="owner/name (default: keep the manifest's)")
    a = ap.parse_args()
    m = json.loads(MANIFEST.read_text(encoding="utf8")) if MANIFEST.exists() else {"repo": "broistg/deep-learning-CO5085", "tag": "weights-v1", "assets": {}}
    m["tag"] = a.tag or m["tag"]
    m["repo"] = a.repo or m["repo"]
    OUT.mkdir(exist_ok=True)
    packed = []
    for name, (folder, files) in ASSETS.items():
        base = ROOT / folder
        members = sorted(p for f in files for p in base.glob(f"*/{f}"))
        if not members:
            print(f"{name}: nothing in {folder}, skipped")
            continue
        zp = OUT / name
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in members:
                z.write(p, p.relative_to(ROOT).as_posix())
        m["assets"][name] = {"sha256": sha256_file(zp), "bytes": zp.stat().st_size, "files": len(members)}
        packed.append(zp)
        print(f"{name}: {len(members)} files, {zp.stat().st_size / 1e6:.1f} MB, sha256 {m['assets'][name]['sha256'][:16]}...")
    MANIFEST.write_text(json.dumps(m, indent=1) + "\n", encoding="utf8")
    if packed:
        print(f"\nmanifest written: {MANIFEST.relative_to(ROOT)}  (commit it)\nthen create the Release and upload:\n"
              f"  gh release create {m['tag']} {' '.join(str(p.relative_to(ROOT)) for p in packed)} "
              f"--repo {m['repo']} --title \"Trained weights ({m['tag']})\" --notes \"Weights and test logits of the E1/E2 runs; verified by SHA-256 in common/release_manifest.json\"")


if __name__ == "__main__":
    main()
