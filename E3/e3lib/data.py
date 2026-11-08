
"""CIFAR-10 data pipeline for E3."""

import numpy as np
import torch
import torch.nn.functional as F

from common.cifar import (
    DATA_ROOT,
    SPLIT_PATH,
    channel_stats,
    get_splits,
)


class ImageBatches:
    """Store CIFAR-10 images as uint8 and prepare batches."""

    def __init__(self, x_nhwc, y, device, mean, std):
        self.device = torch.device(device)

        # [N, H, W, C] -> [N, C, H, W]
        self.x = (
            torch.from_numpy(np.ascontiguousarray(x_nhwc))
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(self.device)
        )

        self.y = torch.as_tensor(
            np.asarray(y),
            dtype=torch.long,
            device=self.device,
        )

        self.n = len(self.y)

        self.mean = torch.tensor(
            mean, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            std, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

        if self.x.shape != (self.n, 3, 32, 32):
            raise ValueError("Expected CIFAR-10 images [N, 3, 32, 32]")

        if self.x.dtype != torch.uint8:
            raise ValueError("Images must be uint8")

    def normalize(self, x):
        """Convert uint8 images to normalized float32 tensors."""
        return (x.float() / 255.0 - self.mean) / self.std

    def eval_batches(self, batch_size):
        """Deterministic batches without augmentation."""
        for start in range(0, self.n, batch_size):
            end = start + batch_size

            xb = self.normalize(self.x[start:end])
            yb = self.y[start:end]

            yield xb, yb

    def train_batches(
        self,
        batch_size,
        seed,
        epoch,
        max_batches=None,
    ):
        """Shuffled batches with random crop and horizontal flip."""

        # Independent randomness for each seed and epoch
        generator = torch.Generator().manual_seed(
            seed * 100003 + epoch
        )

        perm = torch.randperm(self.n, generator=generator)

        # Crop offsets for an image padded by 4 on each side
        oy = torch.randint(
            0, 9, (self.n,), generator=generator
        )
        ox = torch.randint(
            0, 9, (self.n,), generator=generator
        )

        flips = (
            torch.rand(self.n, generator=generator) < 0.5
        )

        positions = torch.arange(32, device=self.device)

        n_batches = self.n // batch_size

        if max_batches is not None:
            n_batches = min(n_batches, max_batches)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            idx_cpu = perm[start:end]
            idx = idx_cpu.to(self.device)

            xb = self.normalize(self.x[idx])

            # zero padding after normalization = mean
            xb = F.pad(xb, (4, 4, 4, 4))

            rows = (
                oy[idx_cpu].to(self.device)[:, None]
                + positions
            )
            cols = (
                ox[idx_cpu].to(self.device)[:, None]
                + positions
            )

            cols = torch.where(
                flips[idx_cpu].to(self.device)[:, None],
                cols.flip(1),
                cols,
            )

            bi = torch.arange(
                len(idx), device=self.device
            )[:, None, None]

            # Random crop and horizontal flip
            xb = xb[
                bi,
                :,
                rows[:, :, None],
                cols[:, None, :],
            ].permute(0, 3, 1, 2).contiguous()

            yield xb, self.y[idx]



def make_datasets(device, data_root=None, fake=False):
    """Build train/validation/test datasets from shared CIFAR-10."""

    splits, split_hash = get_splits(
        data_root or DATA_ROOT,
        SPLIT_PATH,
        fake=fake,
    )

    mean, std = channel_stats(splits["train"][0])

    train = ImageBatches(
        *splits["train"], device, mean=mean, std=std
    )
    val = ImageBatches(
        *splits["val"], device, mean=mean, std=std
    )
    test = ImageBatches(
        *splits["test"], device, mean=mean, std=std
    )

    metadata = {
        "split_sha256_16": split_hash,
        "channel_mean": mean,
        "channel_std": std,
    }

    return train, val, test, metadata
