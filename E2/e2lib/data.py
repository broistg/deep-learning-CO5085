"""CIFAR-10 held entirely on the GPU as uint8 (45k x 3 x 32 x 32 = 138 MB); augmentation on the GPU too (random crop pad 4 + flip).
Randomness for an epoch comes from a CPU generator seeded by (seed, epoch), so a resumed run sees the same data order."""
import numpy as np
import torch
import torch.nn.functional as F

from common.cifar import CLASSES, DATA_ROOT, SPLIT_PATH, channel_stats, fake_cifar, get_splits, load_cifar  # noqa: F401  (re-exported)

# Fallback only (benchmark with synthetic data). Real runs compute the statistics from the 45,000 TRAIN images: common.cifar.channel_stats.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class GPUSet:
    def __init__(self, x_nhwc, y, device, mean=MEAN, std=STD):
        self.x = torch.from_numpy(x_nhwc).to(device).permute(0, 3, 1, 2).contiguous()      # uint8 (N, 3, 32, 32)
        self.y = torch.from_numpy(np.asarray(y)).long().to(device)
        self.n = len(self.y)
        self.device = device
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def normalize(self, xb):
        return (xb.float() / 255 - self.mean) / self.std

    def eval_batches(self, bs):
        for i in range(0, self.n, bs):
            yield self.normalize(self.x[i:i + bs]), self.y[i:i + bs]

    def train_batches(self, bs, seed, epoch, max_batches=None):
        g = torch.Generator().manual_seed(seed * 100003 + epoch)
        perm = torch.randperm(self.n, generator=g)
        oy, ox = torch.randint(0, 9, (self.n,), generator=g), torch.randint(0, 9, (self.n,), generator=g)
        flip = torch.rand(self.n, generator=g) < 0.5
        ar = torch.arange(32, device=self.device)
        nb = self.n // bs                                              # drop the last partial batch
        for b in range(nb if max_batches is None else min(nb, max_batches)):
            idx = perm[b * bs:(b + 1) * bs]
            xb = F.pad(self.normalize(self.x[idx.to(self.device)]), (4, 4, 4, 4))     # zero pad = mean colour after normalisation
            rows = oy[idx].to(self.device)[:, None] + ar
            cols = ox[idx].to(self.device)[:, None] + ar
            cols = torch.where(flip[idx].to(self.device)[:, None], cols.flip(1), cols)
            bi = torch.arange(len(idx), device=self.device)[:, None, None]
            xb = xb[bi, :, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)    # random 32x32 window (+ mirror)
            yield xb.contiguous(), self.y[idx.to(self.device)]
