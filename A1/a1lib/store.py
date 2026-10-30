"""The shared store: where checkpoints, status, locks and results live so that several Kaggle sessions (and two accounts) can work
on the same run list. Two backends behind one interface:

    LocalStore(path)       a folder (tests, a laptop, or a mounted drive)
    HFStore("org/name")    a private Hugging Face repo; every write is ONE atomic commit

Choose with the environment variable A1_STORE:  "hf:<org>/<repo>"  or a folder path.

Layout inside the store (relative paths):
    lr_choice.json                       learning rates chosen from the sweep
    runs/<run_id>/config.yaml  status.json  run.lock.json  result.json  ckpt_meta.json
    runs/<run_id>/last.pt  logits_val.pt  logits_test.pt  model_final.pt          (large files, never mirrored)
    stats/<worker>.json                  bytes each worker has pushed (for `stats` / the squash schedule)
    maintenance.json  maintenance.log    written only by `python -m a1lib.store squash`

Design rules
  * Small files (json/yaml/log) are mirrored to a local folder by `refresh()`; readers use the mirror (`store.mirror`).
  * `put()` is atomic and can be conditional (`parent=` the head you read): this is what makes lock acquisition safe.
  * Every Hugging Face push adds to the repo's LFS history, which counts against the storage quota. Large files are therefore pushed
    rarely (A1_PUSH_EVERY_MIN, default 45) and `squash` reclaims the history.
  * This module imports no torch, so `status`, `list` and the tests run anywhere.
"""
import argparse
import contextlib
import json
import os
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path

META_SUFFIXES = (".json", ".yaml", ".jsonl", ".log")
STALE_S = float(os.environ.get("A1_LOCK_STALE_S", 2100))        # a lock whose heartbeat is older than this is dead
HEARTBEAT_S = float(os.environ.get("A1_HEARTBEAT_S", 600))      # also the interval at which status.json is pushed
PUSH_EVERY_S = float(os.environ.get("A1_PUSH_EVERY_MIN", 45)) * 60
WARN_BYTES = 40e9                                                # warn when this much has been pushed since the last squash


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class StoreConflict(RuntimeError):
    """A conditional put lost the race (someone committed after `parent`)."""


def _size(v):
    if v is None:
        return 0
    if isinstance(v, (bytes, bytearray)):
        return len(v)
    if isinstance(v, str):
        return len(v.encode("utf8"))
    return Path(v).stat().st_size


def _as_bytes(v):
    if isinstance(v, str):
        return v.encode("utf8")
    return bytes(v)


def is_meta(rel):
    return rel.endswith(META_SUFFIXES)


class Store:
    """Interface + shared behaviour. Subclasses implement _put/_head/_get/_list/_fresh/squash_history/refresh."""
    kind = "abstract"

    def __init__(self):
        self.pushed_bytes = 0
        self.pushed_commits = 0
        self._count_lock = threading.Lock()

    # -- to implement ------------------------------------------------------------------------------------------
    def refresh(self):
        raise NotImplementedError

    def head(self):
        raise NotImplementedError

    def _put(self, files, message, parent):
        raise NotImplementedError

    def get(self, rel, dest):
        raise NotImplementedError

    def list_files(self, prefix=""):
        """[(relative path, size)] of every file in the store under `prefix`."""
        raise NotImplementedError

    def read_bytes_fresh(self, rel, revision=None):
        """Latest content of one small file (bypasses the mirror), or None if it does not exist."""
        raise NotImplementedError

    def squash_history(self):
        raise NotImplementedError

    # -- shared ------------------------------------------------------------------------------------------------
    def put(self, files, message="update", parent=None):
        """files: {relative path: Path | bytes | str | None}; None deletes. One atomic commit. parent: head the caller read."""
        files = {k: v for k, v in files.items()}
        n = sum(_size(v) for v in files.values())
        self._put(files, message, parent)
        with self._count_lock:
            self.pushed_bytes += n
            self.pushed_commits += 1

    def read_json(self, rel, default=None, fresh=False, revision=None):
        try:
            if fresh:
                b = self.read_bytes_fresh(rel, revision)
            else:
                p = self.mirror / rel
                b = p.read_bytes() if p.exists() else None
            return default if b is None else json.loads(b.decode("utf8"))
        except (ValueError, OSError):
            return default

    def exists(self, rel):
        return any(r == rel for r, _ in self.list_files(rel))

    def describe(self):
        return f"{self.kind}:{getattr(self, 'repo_id', None) or getattr(self, 'root', '')}"


# ----------------------------------------------------------------------------------------------------------------
class LocalStore(Store):
    kind = "local"

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.mirror = self.root
        self._meta = self.root / ".store"
        self._meta.mkdir(exist_ok=True)

    @contextlib.contextmanager
    def _flock(self):
        try:
            import fcntl
        except ImportError:                                   # Windows: crude mkdir lock
            d = self._meta / "lock.d"
            t0 = time.time()
            while True:
                try:
                    d.mkdir()
                    break
                except FileExistsError:
                    if time.time() - t0 > 30:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                d.rmdir()
            return
        with open(self._meta / "lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def refresh(self):
        return None

    def head(self):
        p = self._meta / "head"
        return p.read_text().strip() if p.exists() else "0"

    def _put(self, files, message, parent):
        with self._flock():
            if parent is not None and parent != self.head():
                raise StoreConflict(f"head moved: {parent} -> {self.head()}")
            for rel, v in files.items():
                dst = self.root / rel
                if v is None:
                    dst.unlink(missing_ok=True)
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + f".tmp{uuid.uuid4().hex[:6]}")
                if isinstance(v, (bytes, bytearray, str)):
                    tmp.write_bytes(_as_bytes(v))
                else:
                    shutil.copyfile(v, tmp)
                os.replace(tmp, dst)
            (self._meta / "head").write_text(uuid.uuid4().hex[:12])

    def get(self, rel, dest):
        src = self.root / rel
        if not src.exists():
            raise FileNotFoundError(rel)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        return dest

    def list_files(self, prefix=""):
        out = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and ".store" not in p.relative_to(self.root).parts and ".tmp" not in p.name:
                rel = p.relative_to(self.root).as_posix()
                if rel.startswith(prefix):
                    out.append((rel, p.stat().st_size))
        return out

    def read_bytes_fresh(self, rel, revision=None):
        p = self.root / rel
        return p.read_bytes() if p.exists() else None

    def squash_history(self):
        return None                                           # a folder has no history


# ----------------------------------------------------------------------------------------------------------------
def hf_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001
        return None


class HFStore(Store):
    kind = "hf"

    def __init__(self, repo_id, mirror_dir, token=None, create=False):
        super().__init__()
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.token = token or hf_token()
        if not self.token:
            raise RuntimeError("no Hugging Face token: add a Kaggle secret named HF_TOKEN (write access) or set HF_TOKEN")
        self.api = HfApi(token=self.token)
        self.mirror = Path(mirror_dir)
        self.mirror.mkdir(parents=True, exist_ok=True)
        if create:
            self.api.create_repo(repo_id, private=True, exist_ok=True)

    # every network call goes through here: retry 429/5xx/timeouts, never retry a lost race or a missing file
    def _retry(self, fn, tries=5):
        from huggingface_hub.utils import HfHubHTTPError
        for i in range(tries):
            try:
                return fn()
            except HfHubHTTPError as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code in (412, 409):
                    raise StoreConflict(str(e)[:200]) from e
                if code in (401, 403, 404) or i == tries - 1:
                    raise
                wait = min(60, 3 * 2 ** i)
                log(f"  ! Hugging Face HTTP {code}; retry in {wait}s ({i + 1}/{tries})")
                time.sleep(wait)
            except (OSError, ConnectionError, TimeoutError) as e:
                if i == tries - 1:
                    raise
                wait = min(60, 3 * 2 ** i)
                log(f"  ! network error {type(e).__name__}; retry in {wait}s ({i + 1}/{tries})")
                time.sleep(wait)

    def head(self):
        return self._retry(lambda: self.api.repo_info(self.repo_id).sha)

    def list_files(self, prefix=""):
        def go():
            kw = {"path_in_repo": prefix.rstrip("/")} if prefix and "/" in prefix else {}
            out = []
            for x in self.api.list_repo_tree(self.repo_id, recursive=True, **kw):
                if hasattr(x, "size") and not str(type(x).__name__).endswith("Folder"):
                    if x.path.startswith(prefix):
                        out.append((x.path, int(x.size or 0)))
            return sorted(out)
        return self._retry(go)

    def refresh(self):
        """Mirror every small file (json/yaml/log) into self.mirror and drop mirrored files that no longer exist remotely."""
        from huggingface_hub import snapshot_download

        def go():
            snapshot_download(self.repo_id, token=self.token, local_dir=str(self.mirror),
                              allow_patterns=[f"*{s}" for s in META_SUFFIXES])
        self._retry(go)
        remote = {r for r, _ in self.list_files()}
        for p in list(self.mirror.rglob("*")):
            if p.is_file() and ".cache" not in p.relative_to(self.mirror).parts and is_meta(p.name):
                if p.relative_to(self.mirror).as_posix() not in remote:
                    p.unlink(missing_ok=True)

    def read_bytes_fresh(self, rel, revision=None):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        def go():
            tmp = self.mirror / ".fresh" / uuid.uuid4().hex[:8]
            try:
                p = hf_hub_download(self.repo_id, rel, token=self.token, local_dir=str(tmp), revision=revision, force_download=True)
                return Path(p).read_bytes()
            except EntryNotFoundError:
                return None
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        try:
            return self._retry(go)
        except Exception as e:  # noqa: BLE001   (404 on a missing path surfaces as an HTTP error in some versions)
            if getattr(getattr(e, "response", None), "status_code", None) == 404:
                return None
            raise

    def _put(self, files, message, parent):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        ops = []
        for rel, v in files.items():
            if v is None:
                ops.append(CommitOperationDelete(path_in_repo=rel))
            elif isinstance(v, (bytes, bytearray, str)):
                ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=_as_bytes(v)))
            else:
                ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(v)))
        # a delete of a file that is already gone makes the commit fail; filter those out using the listing
        if any(isinstance(o, CommitOperationDelete) for o in ops):
            have = {r for r, _ in self.list_files()}
            ops = [o for o in ops if not isinstance(o, CommitOperationDelete) or o.path_in_repo in have]
        if not ops:
            return
        kw = {"parent_commit": parent} if parent else {}
        self._retry(lambda: self.api.create_commit(self.repo_id, ops, commit_message=message[:200], **kw))
        for rel, v in files.items():                           # keep our own writes visible in the mirror
            if is_meta(rel):
                p = self.mirror / rel
                if v is None:
                    p.unlink(missing_ok=True)
                elif isinstance(v, (bytes, bytearray, str)):
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(_as_bytes(v))
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(v, p)

    def get(self, rel, dest):
        from huggingface_hub import hf_hub_download
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".hfdl_{uuid.uuid4().hex[:8]}"

        def go():
            return hf_hub_download(self.repo_id, rel, token=self.token, local_dir=str(tmp), force_download=True)
        try:
            p = self._retry(go)
            os.replace(p, dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return dest

    def squash_history(self):
        self._retry(lambda: self.api.super_squash_history(repo_id=self.repo_id, branch="main"))


def open_store(spec=None, local_dir=None, create=False):
    """spec: 'hf:org/repo' or a folder. Default: environment variable A1_STORE."""
    spec = spec or os.environ.get("A1_STORE")
    if not spec:
        raise RuntimeError("A1_STORE is not set. Use 'hf:<org>/<repo>' (Kaggle) or a folder path (tests).")
    if spec.startswith("hf:"):
        mirror = Path(local_dir or os.environ.get("A1_LOCAL", "/tmp/a1_local")) / "mirror"
        return HFStore(spec[3:], mirror, create=create)
    return LocalStore(spec)


# ----------------------------------------------------------------------------------------------------------------
# worker identity and the per-run lock
# ----------------------------------------------------------------------------------------------------------------
def worker_id(owner, gpu=None):
    """Unique per process family: <owner>-<session>-<gpu>. A1_SESSION names the Kaggle session (e.g. s1, s2); two workers of the
    same owner on the same machine differ by GPU, two sessions differ by A1_SESSION."""
    sess = os.environ.get("A1_SESSION") or socket.gethostname()
    g = os.environ.get("CUDA_VISIBLE_DEVICES", "") if gpu is None else str(gpu)
    g = (g.split(",")[0] if g else "x") or "x"
    return f"{owner}-{sess}-g{g}"


class RunLock:
    """One lock file per run. A live lock (heartbeat younger than STALE_S) held by ANOTHER worker blocks acquisition, even when that
    worker has the same owner: two sessions of the same person must not train the same run. Acquisition is conditional on the
    store head (compare-and-swap), so two workers starting at the same moment cannot both win."""

    def __init__(self, store, run_id, owner, worker, stale_s=STALE_S):
        self.store, self.rid, self.owner, self.worker, self.stale_s = store, run_id, owner, worker, stale_s
        self.rel = f"runs/{run_id}/run.lock.json"
        self.lost = False

    def payload(self):
        return json.dumps({"owner": self.owner, "worker": self.worker, "host": socket.gethostname(), "heartbeat": time.time()})

    def acquire(self, force=False, tries=8):
        for i in range(tries):
            head = self.store.head()
            info = self.store.read_json(self.rel, {}, fresh=True, revision=head if self.store.kind == "hf" else None) or {}
            age = time.time() - info.get("heartbeat", 0)
            if info and info.get("worker") != self.worker and age < self.stale_s and not force:
                raise RuntimeError(f"run {self.rid} is locked by worker '{info.get('worker')}' (owner {info.get('owner')}, "
                                   f"heartbeat {age:.0f}s ago)")
            try:
                self.store.put({self.rel: self.payload()}, f"lock {self.rid} {self.worker}", parent=head)
                self.lost = False
                return
            except StoreConflict:
                time.sleep(1 + i)                             # someone committed in between: re-read and decide again
        raise RuntimeError(f"could not acquire the lock of {self.rid} (store is busy); try again in a minute")

    def still_mine(self):
        info = self.store.read_json(self.rel, {}, fresh=True) or {}
        return info.get("worker") == self.worker

    def release(self):
        try:
            if self.still_mine():
                self.store.put({self.rel: None}, f"unlock {self.rid} {self.worker}")
        except Exception as e:  # noqa: BLE001
            log(f"  ! could not release the lock of {self.rid} ({e!r}); it expires by itself after {self.stale_s / 60:.0f} min")


def lock_is_live(info, stale_s=STALE_S, now=None):
    return bool(info) and (now or time.time()) - info.get("heartbeat", 0) < stale_s


# ----------------------------------------------------------------------------------------------------------------
# what a worker writes for one run
# ----------------------------------------------------------------------------------------------------------------
class RunSync:
    """Pushes one run's state to the store: heartbeat+status (small, every HEARTBEAT_S), checkpoint (large, every PUSH_EVERY_S and at
    clean exit), final artefacts. Local files stay the working copy; the store is the hand-over point between sessions."""

    def __init__(self, store, run_id, owner, worker, local_dir, push_every_s=None):
        self.store, self.rid, self.owner, self.worker = store, run_id, owner, worker
        self.local = Path(local_dir)
        self.rel = f"runs/{run_id}"
        self.lock = RunLock(store, run_id, owner, worker)
        self.push_every_s = PUSH_EVERY_S if push_every_s is None else push_every_s
        self.last_ckpt_push = time.time()
        self.last_pushed_epoch = -1
        self._mutex = threading.Lock()

    def _stats(self):
        return {f"stats/{self.worker}.json": json.dumps({"worker": self.worker, "bytes_pushed": self.store.pushed_bytes,
                                                         "commits": self.store.pushed_commits, "updated": time.time()})}

    def beat(self):
        """Lock heartbeat + local status.json in ONE commit. Called by the heartbeat thread and on state changes."""
        with self._mutex:
            if not self.lock.still_mine():
                self.lock.lost = True
                raise RuntimeError(f"lock of {self.rid} was taken over by another worker (this one looked dead for >{STALE_S / 60:.0f} min)")
            files = {self.lock.rel: self.lock.payload(), **self._stats()}
            st = self.local / "status.json"
            if st.exists():
                files[f"{self.rel}/status.json"] = st
            self.store.put(files, f"beat {self.rid} {self.worker}")

    def should_push_ckpt(self):
        return time.time() - self.last_ckpt_push >= self.push_every_s

    def push_ckpt(self, epoch):
        """last.pt + ckpt_meta.json in one commit (the checkpoint was verified when it was written locally)."""
        with self._mutex:
            src = self.local / "ckpt" / "last.pt"
            meta = json.dumps({"epoch": epoch, "saved_at": time.time(), "worker": self.worker})
            self.store.put({f"{self.rel}/last.pt": src, f"{self.rel}/ckpt_meta.json": meta, **self._stats()},
                           f"ckpt {self.rid} epoch {epoch}")
            self.last_ckpt_push = time.time()
            self.last_pushed_epoch = epoch

    def remote_epoch(self):
        m = self.store.read_json(f"{self.rel}/ckpt_meta.json", None, fresh=True)
        return None if not m else int(m["epoch"])

    def pull_ckpt(self, local_epoch=0):
        """Download the store's last.pt when it is ahead of the local one. Returns the epoch pulled, or None."""
        ep = self.remote_epoch()
        if ep is None or ep <= local_epoch:
            return None
        dest = self.local / "ckpt" / "last.pt"
        self.store.get(f"{self.rel}/last.pt", dest)
        return ep

    def push_final(self, files, result_json):
        """Final artefacts + result.json in one commit; result.json is what marks the run as finished."""
        with self._mutex:
            allf = {f"{self.rel}/{name}": p for name, p in files.items()}
            allf[f"{self.rel}/result.json"] = result_json
            allf.update(self._stats())
            self.store.put(allf, f"result {self.rid}")


# ----------------------------------------------------------------------------------------------------------------
# maintenance: stats and the guarded squash
# ----------------------------------------------------------------------------------------------------------------
def _fmt(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def live_workers(store, stale_s=STALE_S):
    store.refresh()
    out = []
    for p in sorted((store.mirror / "runs").glob("*/run.lock.json")) if (store.mirror / "runs").exists() else []:
        info = store.read_json(p.relative_to(store.mirror).as_posix(), {}) or {}
        if lock_is_live(info, stale_s):
            out.append((p.parent.name, info.get("worker"), time.time() - info.get("heartbeat", 0)))
    return out


def stats(store):
    store.refresh()
    files = store.list_files()
    now_bytes = sum(s for _, s in files)
    per = {}
    for p in sorted((store.mirror / "stats").glob("*.json")) if (store.mirror / "stats").exists() else []:
        d = store.read_json(p.relative_to(store.mirror).as_posix(), {}) or {}
        per[d.get("worker", p.stem)] = float(d.get("bytes_pushed", 0))
    base = (store.read_json("maintenance.json", {}) or {}).get("baseline", {})
    since = sum(max(0.0, v - base.get(w, 0.0)) for w, v in per.items())
    return {"files": len(files), "current_bytes": now_bytes, "pushed_since_squash": since, "per_worker": per,
            "last_squash": (store.read_json("maintenance.json", {}) or {}).get("last_squash"), "warn": since >= WARN_BYTES}


def print_stats(store):
    s = stats(store)
    print(f"store {store.describe()}: {s['files']} files, {_fmt(s['current_bytes'])} now")
    print(f"pushed since the last squash (upper bound of the LFS history that counts against the quota): {_fmt(s['pushed_since_squash'])}")
    if s["last_squash"]:
        print("last squash:", time.strftime("%Y-%m-%d %H:%M", time.localtime(s["last_squash"])))
    if s["warn"]:
        print(f"!! more than {WARN_BYTES / 1e9:.0f} GB pushed since the last squash: run `python -m a1lib.store squash` "
              "when no run is active (e.g. between sessions)")
    return s


def squash(store, actor, force=False, stale_s=STALE_S):
    """Squash the Hugging Face history to reclaim LFS storage. Destructive and irreversible, hence guarded:
    refuses while any worker has a live lock (unless force), records file lists before/after, verifies nothing changed, logs it."""
    live = live_workers(store, stale_s)
    if live and not force:
        lines = "\n".join(f"  {rid}: {w} (heartbeat {age / 60:.0f} min ago)" for rid, w, age in live)
        raise RuntimeError(f"refusing to squash: {len(live)} run(s) have a live lock (heartbeat < {stale_s / 60:.0f} min):\n{lines}\n"
                           "Wait until the sessions have stopped, or pass --force if you are sure those workers are dead.")
    before = store.list_files()
    log(f"squash: {len(before)} files, {_fmt(sum(s for _, s in before))} before")
    s0 = stats(store)
    store.squash_history()
    after = store.list_files()
    ok = dict(before) == dict(after)
    entry = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} squash by {actor}: files {len(before)} -> {len(after)}, "
             f"bytes {sum(s for _, s in before)} -> {sum(s for _, s in after)}, pushed_since_last {int(s0['pushed_since_squash'])}, "
             f"live_locks {len(live)}{' (FORCED)' if force and live else ''}, verify {'OK' if ok else 'MISMATCH'}")
    old_log = store.read_bytes_fresh("maintenance.log") or b""
    store.put({"maintenance.log": old_log + (entry + "\n").encode("utf8"),
               "maintenance.json": json.dumps({"last_squash": time.time(), "by": actor, "baseline": s0["per_worker"]})},
              "maintenance: squash record")
    if not ok:
        missing = sorted(set(dict(before)) - set(dict(after)))[:10]
        raise RuntimeError(f"squash verification FAILED: file lists differ (missing after: {missing}). See maintenance.log.")
    print(entry)
    return entry


def main():
    ap = argparse.ArgumentParser(description="python -m a1lib.store {init|ls|stats|squash}   (store from A1_STORE)")
    ap.add_argument("cmd", choices=["init", "ls", "stats", "squash"])
    ap.add_argument("--actor", default=os.environ.get("A1_OWNER", "unknown"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    store = open_store(create=(a.cmd == "init"))
    if a.cmd == "init":
        print("store ready:", store.describe())
    elif a.cmd == "ls":
        for rel, size in store.list_files():
            print(f"{size:>14,d}  {rel}")
    elif a.cmd == "stats":
        print_stats(store)
    else:
        squash(store, a.actor, a.force)


if __name__ == "__main__":
    main()
