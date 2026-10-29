# A1 team protocol: running the full experiment on two Colab accounts

Goal: finish every run of the locked plan (`plan.json`, 77 configs: 36 LR-sweep, 8 core, 12 seed, 12 ablation, 9 stretch; about 25 GPU-hours committed, about 20 more if the stretch runs are done) with two people, free Colab, no lost work, and numbers that are identical in kind no matter who trained them.

## 1. Accounts and roles
- Each member uses **only their own** Google account and their own Colab session. Do not share an account or open several accounts to get more GPU time (the course FAQ forbids circumventing limits).
- **Coordinator** (one person, default: dat): owns `plan.json`, runs `plan choose-lr`/`extend`, makes the GitHub Release. **Everyone** runs their own worker.
- Owners in the plan are `dat` and `ly`. Runs are pre-assigned to balance estimated GPU time; you only run your own unless you agree otherwise in chat (section 8).

## 2. Shared folder (Google Drive): `a1_shared/`
Create once by the coordinator, share with the teammate as **Editor**. The teammate opens it and uses *Add shortcut to Drive* so it appears as `My Drive/a1_shared` (the notebook checks this exact path).

```
a1_shared/
  cache/food101_256.tar          dataset pre-resized to 256 px, contains split.json (sha256[:16] must equal 75ed81bfe380c089); + manifest.json
  lr_choice.json                 chosen peak LR per (model, group); written only by `plan choose-lr`
  runs/<run_id>/
    config.yaml                  copy of the config used (hash-checked on every start)
    status.json                  live state; written only by the run's current owner
    run.lock.json                heartbeat, so nobody trains the same run twice
    ckpt/last.pt, last_good.pt   resumable checkpoints (atomic, CRC-verified)
    result.json                  final metrics + history + timing + env   (its existence = the run is DONE)
    logits_val.pt, logits_test.pt, model_final.pt   (not for sweep runs)
```
Single-writer rules: a run directory is written by one person at a time (the lock enforces it). `lr_choice.json` and `plan.json` are written by the coordinator only. Nobody edits or deletes files in `runs/` by hand.

## 3. One-time setup (about 30 minutes, once)
1. Coordinator: create the repo, push this code, fill `REPO_URL` in `A1/notebooks/worker.ipynb`.
2. Coordinator, on Colab or any machine with the official `food-101` folder: `python -m a1lib.data make-split --food-root <food-101> --out split.json` (must print `OK`), then `python -m a1lib.data build-cache --food-root <food-101> --shared-dir <a1_shared>` which resizes to 256 px, writes `cache/food101_256.tar` and `manifest.json` into `a1_shared/` and verifies them. Do **not** upload the raw dataset anywhere public.
3. Coordinator: `python -m a1lib.plan make --owners dat,ly`; commit `A1/configs/` and `A1/plan.json`; `python -m a1lib.plan show` to check hours per person.
4. Both: open `A1/notebooks/worker.ipynb` in Colab (GPU: T4), set `OWNER`, run cells 1 to 5.
5. Both: **smoke test** once: `python -m a1lib.train --config A1/configs/m_resnet50_head_s0.yaml --smoke` (2 epochs x 6 batches, writes only to local disk). It must finish and print DONE. Report failures in chat before starting real runs.

## 4. Order of work
1. **Sweep** (stage 0, about 1.8 GPU-h): both run `worker run`. When both have finished their sweeps (`worker status` shows all `sw_*` done), the coordinator runs `python -m a1lib.plan choose-lr`. If it prints an EDGE message, run the printed `plan extend ...` command, commit, both `git pull`, both rerun the worker, then `choose-lr` again. (Rule: best val top-1 at epoch 3, tie goes to lower val NLL; extend once by x3 or /3 if the best is at the edge of the grid.)
2. **Core** (8 runs), **seeds** (12), **ablation** (12): the worker takes them in plan order; runs whose LR is not chosen yet are skipped, not failed.
3. **Stretch** only after everything above is done and only if time remains (cutoff for starting any new run: 05/11).
4. Post-hoc analyses (calibration, ensemble, robustness, Grad-CAM, errors) run on the saved logits and weights; they need no GPU time on the critical path.

## 5. Daily routine
1. Open the notebook, run cells 1 to 6 (about 3 minutes until training starts). Cell 6 is the only long one.
2. Check `worker status` from time to time (cell 7, or a second tab). Epochs take 3 to 11 minutes, so a status older than about 30 minutes while you expect progress means something is wrong.
3. When a session ends, rerun cells 1 to 6. Nothing else is needed.

## 6. Preventing and surviving Colab disconnects
Colab free gives at most about 12 h per session, disconnects idle sessions, and has dynamic usage limits (GPU may be refused for a while). We design for interruption instead of fighting it.
- Keep the Colab tab open and in the foreground when possible, keep the laptop awake and plugged in (disable sleep), stable network. Training itself is activity; cell output scrolling is not required.
- **Do not** use keep-alive hacks (auto-clicking scripts, fake activity) and **do not** use extra accounts. They violate the usage rules and risk losing the account; the resume system makes them unnecessary.
- What protects the work: a checkpoint is written locally after every epoch (atomic + verified, previous good one kept) and copied to Drive; the heartbeat thread keeps the lock fresh; on restart the worker resumes from the newest valid checkpoint (local or Drive) with the same data order and RNG state. Worst case lost: one epoch.
- If Drive is slow or full, the push fails up to 3 times then the run stops with an error rather than silently training without a backup. Keep at least 15 GB free on the account that owns `a1_shared` (checkpoints are about 100 to 300 MB each; `last.pt`, `last_good.pt` per run; finished runs can have `ckpt/` removed by the coordinator after `collect`).
- GPU quota exhausted ("cannot connect to GPU"): stop trying for the day, tell the teammate in chat; they may `--steal` your unfinished runs (section 8). Do not switch the runtime to CPU.
- A session killed while a run was `running` shows `STALE?` in `worker status` after 20 minutes without heartbeat. The owner just reruns the worker; anyone else needs agreement first.

## 7. Saving results
- Source of truth: `a1_shared/runs/*/result.json` (small, complete).
- Coordinator (or anyone) after each batch of runs: `python -m a1lib.collect results` copies them to `A1/results/<run_id>.json`; commit and push those files (they are small). Never commit checkpoints, logits, weights or the dataset.
- At the end the coordinator runs `python -m a1lib.collect bundle --out release/` and uploads `a1_logits.zip`, `a1_weights.zip` and `SHA256SUMS.txt` to a GitHub Release (check the file size limits; split if needed). The report and the website cite the Release and the checksums.
- Report numbers come from `result.json` only: final epoch, no best-checkpoint selection, test set used once per run at the end.

## 8. Ownership, takeover and forced actions
- `worker run --owner X --steal` also takes **unfinished runs of others that have no live lock**. Use after agreeing in chat ("I take your remaining ablation runs").
- A run that is live (heartbeat < 20 min) is never taken. `--force` (ignore a live lock or a config mismatch) is only for a crashed session whose lock is still fresh and needs agreement from the other person first.
- Never change a `config.yaml` or a run_id after the run started; a new setting means a new run_id.

## 9. Quality gates (what counts as a rerun)
- Allowed to rerun: crash, disconnect, Drive error, wrong GPU, a bug that is fixed in the code (then say so in chat and in the commit message).
- Not allowed: rerunning because a number is lower than expected or to pick a better seed. All results of a planned run are reported.
- A run that hits NaN is marked `failed` by the trainer. Report it in chat with the message from `status.json`; do not silently restart with a lower LR. The team decides, and the decision goes into the report.
- Every `result.json` records config hash, split hash, library versions, GPU, sessions (restarts). Mixed environments (e.g. different timm version) are not allowed: everyone pulls the same commit; pinned in `requirements/a1.txt`.

## 10. Timeline
| Date | Goal |
|---|---|
| now to 26/10 | A1.1 submitted (plan and report skeleton); setup and smoke test done |
| 24 to 27/10 | sweep, `choose-lr` |
| 27/10 to 03/11 | core + seeds (about 8.7 GPU-h each person) |
| 03 to 06/11 | ablation, analyses on saved logits |
| 05/11 | cutoff to start new runs; stretch only if everything above is complete |
| 06 to 09/11 | write-up, website, Release; A1.2 due 09/11 |
Note A2.1 (02/11) and A2.2 (16/11) overlap: do not let A2 work stop the worker; GPU runs need only the notebook open.

## 11. Local machine (RTX 3050 6 GB)
Only for development and smoke tests (`--smoke --fake` needs no dataset). Never for reported numbers: other hardware changes timing and numerics.

## 12. AI-use disclosure
Claude was used to write and review this code and protocol. Each member must be able to explain every file; the README declares the use. Run the smoke test and read `train.py` once before you start.
