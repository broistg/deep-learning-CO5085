"""Image corruptions for the robustness test (numpy + PIL only; no torch). Input and output: uint8 HxWx3 arrays (the 224x224 centre crop).

Six corruption types x three severities (1 = mild, 3 = strong). Every call is deterministic: the randomness comes from
sample_seed(BASE_SEED, type*10 + severity, image index), so the corrupted test subset is identical on every machine
(and is never distributed: it is regenerated from the committed seed).
"""
import io

import numpy as np
from PIL import Image, ImageFilter

from a1lib.seeding import sample_seed

BASE_SEED = 20261101
TYPES = ["gaussian_noise", "gaussian_blur", "jpeg", "contrast", "occlusion", "patch_shuffle"]
LEVELS = {
    "gaussian_noise": [0.04, 0.08, 0.16],        # sigma, fraction of the 0-255 range
    "gaussian_blur": [1.0, 2.0, 4.0],            # radius in pixels
    "jpeg": [30, 15, 5],                         # quality
    "contrast": [0.5, 0.3, 0.15],                # factor around the mean
    "occlusion": [0.10, 0.25, 0.40],             # fraction of the image area covered by one grey square
    "patch_shuffle": [2, 4, 8],                  # grid size: the image is cut into g x g patches that are permuted
}


def rng_for(ctype, severity, index):
    return np.random.default_rng(sample_seed(BASE_SEED, TYPES.index(ctype) * 10 + severity, index))


def corrupt(img, ctype, severity, index):
    """img: uint8 HxWx3. severity in {1, 2, 3}. Returns a new uint8 array of the same shape."""
    p = LEVELS[ctype][severity - 1]
    rng = rng_for(ctype, severity, index)
    h, w, _ = img.shape
    if ctype == "gaussian_noise":
        out = img.astype(np.float32) + rng.normal(0, p * 255.0, img.shape).astype(np.float32)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if ctype == "gaussian_blur":
        return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=p)))
    if ctype == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=int(p))
        buf.seek(0)
        with Image.open(buf) as im:
            return np.asarray(im.convert("RGB")).copy()
    if ctype == "contrast":
        m = img.astype(np.float32).mean()
        return np.clip(np.rint((img.astype(np.float32) - m) * p + m), 0, 255).astype(np.uint8)
    if ctype == "occlusion":
        side = int(round(np.sqrt(p * h * w)))
        y0, x0 = int(rng.integers(0, h - side + 1)), int(rng.integers(0, w - side + 1))
        out = img.copy()
        out[y0:y0 + side, x0:x0 + side] = 127
        return out
    if ctype == "patch_shuffle":
        g = int(p)
        ph, pw = h // g, w // g
        tiles = [img[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] for i in range(g) for j in range(g)]
        order = rng.permutation(len(tiles))
        out = img.copy()
        for k, t in enumerate(order):
            i, j = divmod(k, g)
            out[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = tiles[t]
        return out
    raise ValueError(ctype)
