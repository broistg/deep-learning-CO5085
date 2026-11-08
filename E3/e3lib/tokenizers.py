"""
Implement 2 tokenizers
"""

from torch import nn


class PatchTokenizer(nn.Module):
    """Non-overlapping P x P patches, flattened and linearly projected (= Conv2d with kernel = stride = P)."""

    def __init__(self, dim, patch_size, img_size=32):
        super().__init__()

        if patch_size <= 0 or img_size % patch_size != 0:
            raise ValueError(
                "patch_size must be positive and divide img_size"
            )
        
        self.proj = nn.Conv2d(
            in_channels=3, 
            out_channels=dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        self.image_size = img_size
        self.n_steps = (img_size // patch_size) ** 2
        self.grid = (img_size // patch_size, img_size // patch_size)

    def forward(self, x):
        if (x.ndim != 4 or 
            x.shape[1] != 3 or 
            x.shape[2] != self.image_size or 
            x.shape[3] != self.image_size
        ):
            raise ValueError(
                f"Expected input shape [B, 3, {self.image_size}, {self.image_size}],"
                f" but got {x.shape}"
            )

        # [B, 3, 32, 32] -> [B, 64, 8, 8] -> [B, 64, 64] -> [B, 64, 64]
        #                                    [Batch, Features, Sequence] -> [Batch, Sequence, Features]
        return self.proj(x).flatten(2).transpose(1, 2)     


class RowTokenizer(nn.Module):
    """
    Convert a 32x32 RGB image into a sequence of row tokens.
    Each image row (3 channels x 32 pixels = 96 numbers) is one token.
    """

    def __init__(self, dim, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Linear(3 * img_size, dim)
        self.n, self.n_steps, self.grid = img_size, img_size, (img_size, 1)

    def forward(self, x): 
        # x.shape: [B, 3, H, W]
        if (x.ndim != 4 or 
            x.shape[1] != 3 or 
            x.shape[2] != self.img_size or 
            x.shape[3] != self.img_size
        ):
            raise ValueError(
                f"Expected input shape [B, 3, {self.img_size}, {self.img_size}],"
                f" but got {x.shape}"
            )
        
        B, C, H, W = x.shape
        #  [B, C, H, W] -> [B, H, C, W] -> [B, H, C*W] -> [B, H, D]
        return self.proj(x.permute(0, 2, 1, 3).reshape(B, H, C * W))   # (B, H, C*W) -> (B, H, D)


def make_tokenizer(name, dim, img_size=32):
    if name == "patch4":
        return PatchTokenizer(
            dim=dim, 
            patch_size=4, 
            img_size=img_size)
    
    if name == "patch8":
        return PatchTokenizer(
            dim=dim, 
            patch_size=8, 
            img_size=img_size
            )
    if name == "row":
        return RowTokenizer(
            dim=dim, 
            img_size=img_size)
    raise ValueError(f"unknown tokenizer {name}") 
