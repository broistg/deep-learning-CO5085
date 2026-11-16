# E1 baseline comparison

Run `venv/bin/python E3/compare_checkpoints.py` from the repository root after
the six E1 MLP/MiniVGG and twelve E3 LSTM/GRU final checkpoints are available.
The script evaluates all 18 checkpoints on the same CIFAR-10 test set and writes
`checkpoint_results.json` here. It reads the checkpoint files and shared CIFAR-10
data directly; E1/E3 result JSON files are not required. It does not retrain any
model or change the published results in `E1/results/` and `E3/results/`.
