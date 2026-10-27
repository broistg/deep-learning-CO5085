"""Four ways to turn a 3x32x32 image into a sequence of tokens (B, N, D). Every tokenizer ends in dimension D."""
import torch.nn as nn


class PatchTokenizer(nn.Module):
    """Non-overlapping P x P patches, flattened and linearly projected (= Conv2d with kernel = stride = P)."""

    def __init__(self, dim, patch, img=32):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.n = (img // patch) ** 2
        self.grid = (img // patch, img // patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)           # (B, D, h, w) -> (B, h*w, D)


class RowTokenizer(nn.Module):
    """Each image row (3 channels x 32 pixels = 96 numbers) is one token."""

    def __init__(self, dim, img=32):
        super().__init__()
        self.proj = nn.Linear(3 * img, dim)
        self.n, self.grid = img, (img, 1)

    def forward(self, x):                                         # (B, 3, H, W)
        B, C, H, W = x.shape
        return self.proj(x.permute(0, 2, 1, 3).reshape(B, H, C * W))   # (B, H, C*W) -> (B, H, D)


class CNNStemTokenizer(nn.Module):
    """Small conv stem (3 conv layers, two of them stride 2): every position of the 8x8 feature map is a token."""

    def __init__(self, dim, img=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, dim, 3, 2, 1))
        self.n, self.grid = (img // 4) ** 2, (img // 4, img // 4)

    def forward(self, x):
        return self.stem(x).flatten(2).transpose(1, 2)


def make_tokenizer(name, dim, img=32):
    if name == "patch4":
        return PatchTokenizer(dim, 4, img)
    if name == "patch8":
        return PatchTokenizer(dim, 8, img)
    if name == "row":
        return RowTokenizer(dim, img)
    if name == "cnn":
        return CNNStemTokenizer(dim, img)
    raise ValueError(f"unknown tokenizer {name}")


TOKENIZERS = ("patch4", "patch8", "row", "cnn")
