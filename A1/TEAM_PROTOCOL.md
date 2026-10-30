# A1 team protocol: running the full experiment on two Kaggle accounts

Goal: finish every run of the locked plan (`plan.json`, 77 configs: 36 LR-sweep, 8 core, 12 seed, 12 ablation, 9 stretch; about 25 GPU-hours committed, about 20 more with the stretch runs) with two people, free Kaggle GPUs, no lost work, and numbers that are identical in kind no matter who trained them.

Measured on Kaggle (2 x T4, 4 vCPU): ResNet-50 is about 20 % slower than on a Colab T4, DeiT-S about the same; the input pipeline tops out near 740 img/s per machine; one independent worker per GPU scales 100 %. Quota: 30 GPU-hours per account per week, counted in wall-clock session time (both GPUs together cost the same as one).

## 1. Accounts and roles
- Each member uses **only their own** Kaggle account (phone-verified) and Hugging Face login. Never share an account, never open extra accounts for more quota (the course FAQ forbids circumventing limits).
- **Coordinator** (default: dat): owns `plan.json`, runs `plan choose-lr` / `extend`, the store maintenance (section 6) and the GitHub Release. **Everyone** runs their own workers.
- Owners in the plan are `dat` and `ly`. Runs are pre-assigned to balance estimated GPU time; run only your own unless you agree otherwise in chat (section 8).

## 2. The store (private Hugging Face repo) and the local disk
The shared state lives in ONE private repo inside a free Hugging Face organization that both members belong to (e.g. `dl4dl/a1-store`), 100 GB private storage. Both members need a **write** token, kept as a Kaggle secret named `HF_TOKEN` (Notebook > Add-ons > Secrets). Never print or commit it.

```
lr_choice.json                       chosen peak LR per (model, group); written only by `plan choose-lr`
maintenance.json, maintenance.log    written only by `store squash`
stats/<worker>.json                  bytes each worker pushed (drives the squash reminder)
runs/<run_id>/
  config.yaml                        copy of the config used (hash-checked on every start)
  status.json                        live state, pushed with the heartbeat (every 15 min)
  run.lock.json                      worker id + heartbeat: nobody trains the same run twice
  last.pt, ckpt_meta.json            newest pushed checkpoint (every 45 min and at every clean exit) and its epoch
  result.json                        final metrics + history + timing + env   (its existence = the run is DONE)
  logits_val.pt, logits_test.pt, model_final.pt   (not for sweep runs)
```
Every write is one atomic commit, and Hugging Face allows only **128 commits per hour per repo** (all workers of both accounts together). A run therefore costs two commits (start: lock + config; end: result + status + lock removal) plus one heartbeat every 15 minutes; each process keeps to a budget of 24 commits per hour (`A1_MAX_COMMITS_PER_H`; heartbeats are dropped first, real writes wait) and waits out a 429 instead of failing. lock acquisition is conditional on the repo head, so two workers starting at the same moment cannot both win. Nobody edits or deletes files in the store by hand.

Working data stay on the session's local disk (`/kaggle/temp/a1_local`: the dataset copy and the per-epoch checkpoint), never in `/kaggle/working` (20 GB output cap, saved on commit). A session that dies loses at most the epochs since the last push (45 min by default; `A1_PUSH_EVERY_MIN`).

## 3. One-time setup
1. Coordinator: create the Hugging Face organization and the private repo; both members join and create write tokens. Check: `python -m a1lib.store init`.
2. Coordinator: upload the Food-101 cache (`food101_256`, 101 000 images + `split.json`, sha256[:16] must equal `75ed81bfe380c089`) as a **private Kaggle dataset**; share it with the teammate (Kaggle: dataset > Settings > Sharing > add collaborator). Each member adds it to the worker notebook with *Add Input*. Do not upload the raw dataset anywhere public.
3. Coordinator: `python -m a1lib.plan make --owners dat,ly`; commit `A1/configs/` and `A1/plan.json`; `python -m a1lib.plan show`.
4. Both: copy `A1/notebooks/kaggle_worker.ipynb` into your Kaggle account, set `OWNER`, add the dataset, add the `HF_TOKEN` secret, enable GPU T4 x2 and Internet, run the first cells, run the **smoke test** cell once (2 epochs x 6 batches, writes nothing to the store). It must print DONE.

## 4. Order of work
1. **Sweep** (stage 0, about 1.8 GPU-h): both run workers. When all `sw_*` runs are done (`worker status`), the coordinator runs `python -m a1lib.plan choose-lr` (needs `A1_STORE` and `HF_TOKEN`). If it prints an EDGE message, run the printed `plan extend ...` command, commit, both `git pull`, rerun workers, `choose-lr` again. (Rule: best val top-1 at epoch 3, tie goes to lower val NLL; extend once by x3 or /3 if the best is at the edge of the grid.)
2. **Core** (8), **seeds** (12), **ablation** (12): workers take them in plan order; runs whose LR is not chosen yet are skipped, not failed.
3. **Stretch** only after everything above is done and only if time remains.
4. Post-hoc analyses (calibration, ensemble, robustness, Grad-CAM, errors) run on the saved logits and weights.

## 5. Daily routine, workers and sessions
- A **session** is one running Kaggle notebook (max 12 h, stop it at about 11 h with `--max-hours 11`). In it, `python -m a1lib.worker launch --owner X --gpus 2` copies the dataset once (about 5 min), then starts **one worker per GPU**; each worker has the id `<owner>-<session>-g<gpu>` and works through the run list independently (no DataParallel/DDP: it would change batch size, data order and BatchNorm statistics).
- **More sessions at the same time** (same account or both accounts) are allowed and safe: all workers share the one run list through the store locks. They finish sooner in wall-clock but burn quota proportionally faster, and Kaggle limits concurrent GPU sessions (about 2 per account; check your account). Each new session pays the 5 minute data copy, so prefer a few long sessions over many short ones.
- Check progress with `python -m a1lib.worker status` (last cell of the notebook, or a second notebook on CPU). A status older than about 30 minutes while you expect progress means something is wrong.
- When a session ends (time limit, quota, crash), start a new one with the same cells. Workers resume from the store's checkpoint.

## 6. Store maintenance (Hugging Face history)
Hugging Face keeps every pushed version of a large file (LFS history) and counts it against the 100 GB. Keep the number of large pushes modest (45 min interval) and reclaim history with a squash:
- `python -m a1lib.store stats` shows files, current size and the bytes pushed since the last squash (an upper bound of the history). It prints a warning above 40 GB. Any member may run it; the notebook prints it at the end of every session.
- `python -m a1lib.store squash --actor dat` (**coordinator only**) rewrites the history to one commit. It is irreversible, therefore: it **refuses** while any run has a live lock (heartbeat < 45 min; `--force` only when you know those workers are dead), lists files before and after, verifies that the file list and sizes are identical, and appends a line to `maintenance.log`. Do it between sessions (nobody running), after announcing it in chat, when `stats` warns or at the latest weekly. After a squash the quota may take up to about 36 h to refresh.
- Never run it in the middle of someone's session. If it reports MISMATCH, stop, tell the team, check `maintenance.log`.

## 7. Saving results
- Source of truth: `runs/*/result.json` in the store (small, complete).
- Coordinator (or anyone) after each batch: `python -m a1lib.collect results` copies them to `A1/results/<run_id>.json`; commit and push those (small). Never commit checkpoints, logits, weights or the dataset.
- At the end the coordinator runs `python -m a1lib.collect bundle --out release/` and uploads `a1_logits.zip`, `a1_weights.zip`, `SHA256SUMS.txt` to a GitHub Release (split if over the size limit). The report and the website cite the Release and the checksums.
- Report numbers come from `result.json` only: final epoch, no best-checkpoint selection, test set used once per run at the end.

## 8. Ownership, takeover and forced actions
- `worker run --owner X --steal` also takes **unfinished runs of others that have no live lock**. Use after agreeing in chat.
- A live run (heartbeat < 45 min) is never taken, not even by another worker of the same owner. `--force` (ignore a live lock or a config mismatch) is only for a crashed session whose lock is still fresh, after the other person agrees.
- Never change a `config.yaml` or a run_id after the run started; a new setting means a new run_id.

## 9. Reproducibility and quality gates
- Augmentation is seeded **per sample** from (run seed, epoch, sample index): the same images get the same crops/RandAugment/flip whatever `num_workers`, session or GPU. Mixup/CutMix runs in the main process and its RNG is saved in the checkpoint, so a resumed run continues the same stream. Same hardware reproduces bit-for-bit; different GPU types differ slightly (cuDNN, fp16), so do not mix hardware inside one comparison without saying so.
- Allowed to rerun: crash, disconnect, store error, a bug that is fixed in the code (say so in chat and in the commit message).
- Not allowed: rerunning because a number is lower than expected or to pick a better seed. All results of a planned run are reported.
- A run that hits NaN is marked `failed`. Report it with the message from `status.json`; do not silently restart with a lower LR. The team decides, and the decision goes into the report.
- Every `result.json` records config hash, split hash, library versions, GPU, sessions (restarts) and the worker. Everyone pulls the same commit; versions are pinned in `requirements/a1.txt`.

## 10. Timeline
Confirm the deadlines with the course page before relying on this table.
| Goal | When |
|---|---|
| setup, smoke test, sweep, `choose-lr` | first 2 sessions |
| core + seeds (about 8.7 GPU-h per person) | next |
| ablation, analyses on saved logits | after that |
| cutoff to start new runs; stretch only if everything above is complete | about 4 days before the deadline |
| write-up, website, Release | last 3 days |

## 11. Local machine (RTX 3050 6 GB)
Only for development and smoke tests (`--smoke --fake` with `A1_STORE=<some folder>` needs no dataset and no Hugging Face). Never for reported numbers.

## 12. AI-use disclosure
Claude was used to write and review this code and protocol. Each member must be able to explain every file; the README declares the use. Run the smoke test and read `train.py` once before you start.
