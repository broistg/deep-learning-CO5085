"""Per-sample augmentation seeds. Pure Python (no torch), so it is unit-tested without a GPU.

Why: with the DataLoader's default seeding, each worker process owns one RNG stream and batches go to workers round-robin, so
WHICH random crop an image gets depends on num_workers. Here the seed of every sample is a function of (run seed, epoch, sample
index) only. Any num_workers (and any machine) gives the same augmented image for the same sample and epoch.
"""

_M = (1 << 64) - 1


def _mix(x):
    """splitmix64 finaliser: a cheap, well-mixed 64-bit hash."""
    x = (x + 0x9E3779B97F4A7C15) & _M
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M
    return x ^ (x >> 31)


def sample_seed(run_seed, epoch, index):
    """32-bit seed for the augmentation of sample `index` in `epoch` of a run with seed `run_seed`."""
    h = _mix(_mix(_mix(int(run_seed)) ^ int(epoch)) ^ int(index))
    return h & 0xFFFFFFFF
