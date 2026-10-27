"""E1 data: three separate Dataset objects over the frozen split (common/cifar.py). Only the TRAIN one augments.
Pipeline per image: uint8 -> float [0, 1] -> Normalize(train mean/std) -> [train only] RandomCrop(32, padding=4) + RandomHorizontalFlip.
Cropping AFTER normalisation means the padding (zeros) is the mean colour, exactly as in E2."""
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from common.cifar import CLASSES


class CifarArrays(Dataset):
    """Also exposes `.data` (uint8 NHWC), `.targets` (tensor) and `.classes`, so the EDA plots of the notebook work on it."""

    def __init__(self, x_nhwc, y, mean, std, train=False):
        self.data = x_nhwc
        self.targets = torch.from_numpy(np.asarray(y)).long()
        self.classes = CLASSES
        self.x = torch.from_numpy(x_nhwc).permute(0, 3, 1, 2).contiguous()          # uint8 (N, 3, 32, 32)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.aug = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]) if train else None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = (self.x[i].float() / 255 - self.mean) / self.std
        if self.aug is not None:
            img = self.aug(img)
        return img, int(self.targets[i])


def make_loaders(splits, mean, std, batch_size=64, workers=2, seed=0, augment=True):
    """-> train_loader, val_loader, test_loader. The shuffle order and the augmentation are seeded by `seed`."""
    d = {k: CifarArrays(*splits[k], mean, std, train=(k == "train" and augment)) for k in ("train", "val", "test")}
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(d["train"], batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True, generator=g)
    val = DataLoader(d["val"], batch_size=batch_size, shuffle=False, num_workers=workers)
    test = DataLoader(d["test"], batch_size=batch_size, shuffle=False, num_workers=workers)
    return train, val, test, d
