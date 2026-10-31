"""Helpers for the replication notebooks (E1/notebooks/E1.ipynb, E2/notebooks/E2.ipynb). No torch import: the load path runs on a CPU runtime.

MODE = "load"   regenerate every table and figure from the committed results and compare them with the committed copies (minutes)
MODE = "rerun"  retrain everything into a fresh results/ folder (the published one is kept in results_published/) and compare

The notebooks stay thin: all logic is here or in e1lib / e2lib."""
import hashlib
import json
import math
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "common" / "release_manifest.json"


# ------------------------------------------------------------------ setup ------------------------------------------------------------
def banner(mode, exp, hours_hint=None):
    if mode not in ("load", "rerun"):
        raise ValueError('MODE must be "load" or "rerun"')
    if mode == "load":
        print(f"{exp}: MODE=load. Regenerates tables/figures from the committed results and checks them against the report. "
              "A free CPU runtime is enough; takes a few minutes.")
    else:
        print(f"{exp}: MODE=rerun. Retrains every model from scratch (needs a GPU runtime).")
        if hours_hint:
            print(f"      {hours_hint}")


def sh(cmd, cwd=ROOT, env_extra=None):
    """Run a command, stream its output, raise on failure (no silent errors in a replication run)."""
    import os
    env = dict(os.environ, **(env_extra or {}))
    print("$", " ".join(map(str, cmd)))
    r = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env)
    if r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")


# ------------------------------------------------------------------ release weights --------------------------------------------------
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_manifest(path=MANIFEST):
    return json.loads(Path(path).read_text(encoding="utf8"))


def fetch_release(asset, dest=None, manifest=None, force=False):
    """Download one Release asset (a zip whose entries are paths relative to the repo root, e.g. E2/artifacts/<run>/model_final.pt),
    verify its SHA-256 against common/release_manifest.json, extract into the repo. Needed only for weights/logits (confusion matrix,
    error gallery, inference checks); the tables and curves do not need it."""
    m = manifest or load_manifest()
    if asset not in m["assets"]:
        raise KeyError(f"{asset} is not in the manifest: run `python common/pack_release.py` first")
    info = m["assets"][asset]
    if not info.get("sha256"):
        raise RuntimeError(f"{asset}: manifest has no sha256 yet (the Release has not been created; see README 'Release weights')")
    dest = Path(dest or ROOT)
    marker = dest / ".release" / f"{asset}.{info['sha256'][:12]}.ok"
    if marker.exists() and not force:
        print(f"{asset}: already extracted")
        return
    url = f"https://github.com/{m['repo']}/releases/download/{m['tag']}/{asset}"
    zpath = dest / ".release" / asset
    zpath.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} ({info.get('bytes', 0) / 1e6:.0f} MB)")
    urllib.request.urlretrieve(url, zpath)
    got = sha256_file(zpath)
    if got != info["sha256"]:
        zpath.unlink()
        raise RuntimeError(f"{asset}: SHA-256 mismatch (got {got[:16]}..., manifest {info['sha256'][:16]}...); file removed")
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():   # refuse path traversal
            if n.startswith(("/", "..")) or ".." in Path(n).parts:
                raise RuntimeError(f"unsafe path in archive: {n}")
        z.extractall(dest)
    marker.write_text("ok\n")
    print(f"{asset}: verified and extracted to {dest}")


# ------------------------------------------------------------------ comparing outputs -----------------------------------------------
def _walk_numbers(a, b, path, tol, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}/{k}", "missing on one side", None))
            else:
                _walk_numbers(a[k], b[k], f"{path}/{k}", tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} vs {len(b)}", None))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _walk_numbers(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(a - b)
        if d > tol * max(1.0, abs(b)):
            out.append((path, f"{a} vs {b}", d))
    elif a != b:
        out.append((path, f"{a!r} vs {b!r}", None))


def compare_outputs(new_dir, ref_dir, tex=True, summary=True, tol=1e-9, optional=()):
    """Regenerated vs committed: the `tab_*.tex` files must be byte-identical and summary.json numerically equal (tol, relative).
    Prints one line per file and raises AssertionError if anything differs, so a broken replication cannot pass silently.
    optional: names of tables / summary keys that exist only when the logits were downloaded (e.g. A1: tab_calibration.tex, "errors");
    they are skipped, with a message, when the regenerated side lacks them."""
    new_dir, ref_dir = Path(new_dir), Path(ref_dir)
    bad = []
    if tex:
        for ref in sorted(ref_dir.glob("tab_*.tex")):
            new = new_dir / ref.name
            if not new.exists() and ref.name in optional:
                print(f"skip {ref.name} (needs the logits from the Release)")
                continue
            ok = new.exists() and new.read_text(encoding="utf8") == ref.read_text(encoding="utf8")
            print(f"{'OK  ' if ok else 'DIFF'} {ref.name}")
            if not ok:
                bad.append(ref.name)
    if summary and (ref_dir / "summary.json").exists():
        diffs = []
        _walk_numbers(json.loads((new_dir / "summary.json").read_text(encoding="utf8")),
                      json.loads((ref_dir / "summary.json").read_text(encoding="utf8")), "", tol, diffs)
        # `errors` comes from the confusion/error figures and exists only when the weights/logits were downloaded
        new_keys = json.loads((new_dir / "summary.json").read_text(encoding="utf8"))
        for key in ("errors",) + tuple(k for k in optional if not k.endswith(".tex")):
            if key not in new_keys:
                diffs = [d for d in diffs if not d[0].startswith(f"/{key}")]
                print(f"(summary.json: '{key}' skipped, the logits were not downloaded)")
        print(f"{'OK  ' if not diffs else 'DIFF'} summary.json" + ("" if not diffs else f"  ({len(diffs)} differences, first: {diffs[0]})"))
        if diffs:
            bad.append("summary.json")
    if bad:
        raise AssertionError(f"regenerated outputs differ from the committed ones: {bad}")
    print("All regenerated tables and numbers equal the committed (= reported) ones.")


def compare_runs(new_dir, ref_dir, metric=("final", "test_acc"), pattern="*.json", atol_points=1.0):
    """Rerun vs published, run by run. A retrain on another GPU is not bit-identical (cuDNN, AMP), so this reports the difference in
    accuracy points and flags the runs beyond `atol_points`; the headline check is the mean per configuration (see `compare_means`)."""
    new_dir, ref_dir = Path(new_dir), Path(ref_dir)
    rows = []
    for ref in sorted(ref_dir.glob(pattern)):
        if ref.name.endswith("_preds.json") or ref.name in ("summary.json",) or ref.name.startswith(("bench_", "tab_")):
            continue
        new = new_dir / ref.name
        r = json.loads(ref.read_text(encoding="utf8"))
        if "final" not in r:
            continue
        a = r
        try:
            for k in metric:
                a = a[k]
        except (KeyError, TypeError):          # e.g. A1 sweep runs have no test metrics
            continue
        if not new.exists():
            rows.append((ref.stem, 100 * a, math.nan, math.nan))
            continue
        b = json.loads(new.read_text(encoding="utf8"))
        try:
            for k in metric:
                b = b[k]
        except (KeyError, TypeError):
            rows.append((ref.stem, 100 * a, math.nan, math.nan))
            continue
        rows.append((ref.stem, 100 * a, 100 * b, 100 * (b - a)))
    print(f"{'run':32s} {'published':>10s} {'rerun':>10s} {'diff (pts)':>11s}")
    for n, a, b, d in rows:
        flag = "  <-- check" if not (abs(d) <= atol_points) else ""
        print(f"{n:32s} {a:10.2f} {b:10.2f} {d:+11.2f}{flag}")
    done = [r for r in rows if not math.isnan(r[3])]
    if done:
        print(f"\n{len(done)}/{len(rows)} runs compared, max |diff| {max(abs(r[3]) for r in done):.2f} pts, "
              f"mean diff {sum(r[3] for r in done) / len(done):+.2f} pts")
    return rows


def compare_means(new_dir, ref_dir, metric=("final", "test_acc"), only_rerun=False):
    """Per configuration (run id without the seed): mean test accuracy of the rerun vs the published mean +- sample std over seeds.
    A rerun reproduces the result when the difference is within about one published std (3 seeds, so only a rough yardstick)."""
    import re
    from statistics import mean, stdev

    def groups(d):
        g = {}
        for p in sorted(Path(d).glob("*.json")):
            if p.name == "summary.json" or p.name.startswith(("bench_", "tab_")):
                continue
            r = json.loads(p.read_text(encoding="utf8"))
            if "final" not in r:
                continue
            v = r
            try:
                for k in metric:
                    v = v[k]
            except (KeyError, TypeError):      # e.g. A1 sweep runs have no test metrics
                continue
            g.setdefault(re.sub(r"_s\d+(?=$|_)", "", p.stem), []).append(100 * v)
        return g
    new, ref = groups(new_dir), groups(ref_dir)
    print(f"{'configuration':30s} {'published':>16s} {'rerun':>8s} {'diff':>7s}")
    for k in sorted(ref):
        a = ref[k]
        sd = stdev(a) if len(a) > 1 else float("nan")
        if k not in new:
            if not only_rerun:
                print(f"{k:30s} {mean(a):7.2f} +- {sd:5.2f}   (not rerun)")
            continue
        d = mean(new[k]) - mean(a)
        flag = "" if abs(d) <= max(sd, 0.5) else "  <-- outside 1 std"
        print(f"{k:30s} {mean(a):7.2f} +- {sd:5.2f} {mean(new[k]):8.2f} {d:+7.2f}{flag}")


# ------------------------------------------------------------------ rerun bookkeeping -----------------------------------------------
def stash_published(exp):
    """Rerun mode: move <exp>/results to <exp>/results_published and leave an empty results/ for the new runs (the training code skips
    runs whose JSON exists, so a rerun needs an empty folder). Safe to call twice."""
    res, pub = ROOT / exp / "results", ROOT / exp / "results_published"
    if pub.exists():
        print(f"{pub.relative_to(ROOT)} already exists; continuing (finished runs in {res.relative_to(ROOT)} are skipped = resume)")
        res.mkdir(exist_ok=True)
        return pub
    if not res.exists() or not any(res.iterdir()):
        raise RuntimeError(f"{res} is empty: nothing published to compare with (did the clone get the results?)")
    shutil.move(str(res), str(pub))
    res.mkdir()
    return pub


def restore_published(exp):
    """Put the published results back (drops the rerun results; they are kept as results_rerun/)."""
    res, pub, rr = (ROOT / exp / n for n in ("results", "results_published", "results_rerun"))
    if not pub.exists():
        print("nothing to restore")
        return
    if rr.exists():
        shutil.rmtree(rr)
    shutil.move(str(res), str(rr))
    shutil.move(str(pub), str(res))
    print(f"published results restored; the rerun is in {rr.relative_to(ROOT)}")


def estimate_hours(results_dir, epochs_scale=1.0):
    """Wall-clock estimate of a rerun from the published runs: sum of (median s/epoch x epochs). Runs on a slower GPU take longer;
    a free Colab T4 is usually within about 1-2x of a laptop RTX 3050."""
    tot, n = 0.0, 0
    for p in Path(results_dir).glob("*.json"):
        if p.name == "summary.json" or p.name.startswith(("bench_", "tab_")):
            continue
        r = json.loads(p.read_text(encoding="utf8"))
        if "train_s_per_epoch_median" in r and "config" in r:
            tot += r["train_s_per_epoch_median"] * r["config"].get("epochs", 30) * epochs_scale
            n += 1
    return tot / 3600, n


def show_pngs(folder, pattern="*.png", width=720):
    from IPython.display import Image, display
    for f in sorted(Path(folder).glob(pattern)):
        print(f.name)
        display(Image(str(f), width=width))


if __name__ == "__main__":
    print(sys.argv[0], "is a library; see the notebooks")
