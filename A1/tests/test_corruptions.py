import unittest

import numpy as np

from a1lib.corruptions import LEVELS, TYPES, corrupt


class Corruptions(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)

    def test_shape_dtype_determinism(self):
        for t in TYPES:
            for s in (1, 2, 3):
                a, b = corrupt(self.img, t, s, 7), corrupt(self.img, t, s, 7)
                self.assertEqual(a.shape, self.img.shape)
                self.assertEqual(a.dtype, np.uint8)
                self.assertTrue(np.array_equal(a, b), (t, s))
        self.assertFalse(np.array_equal(corrupt(self.img, "occlusion", 2, 1), corrupt(self.img, "occlusion", 2, 2)))   # depends on the index

    def test_severity_is_monotone(self):
        base = self.img.astype(np.float32)
        smooth = np.clip(np.add.outer(np.arange(224), np.arange(224)) / 446 * 255, 0, 255).astype(np.uint8)[..., None].repeat(3, 2)
        for t in TYPES:
            ref = smooth if t in ("gaussian_blur", "jpeg") else self.img
            d = [np.abs(corrupt(ref, t, s, 3).astype(np.float32) - ref.astype(np.float32)).mean() for s in (1, 2, 3)]
            self.assertLess(d[0], d[2], (t, d))

    def test_inputs_not_modified_and_occlusion_area(self):
        copy = self.img.copy()
        out = corrupt(self.img, "occlusion", 3, 0)
        self.assertTrue(np.array_equal(self.img, copy))
        frac = (out != self.img).any(axis=2).mean()
        self.assertLess(frac, LEVELS["occlusion"][2] + 0.02)
        self.assertGreater(frac, 0.3)

    def test_patch_shuffle_keeps_pixels(self):
        out = corrupt(self.img, "patch_shuffle", 2, 5)
        self.assertEqual(sorted(out.ravel().tolist()), sorted(self.img.ravel().tolist()))


if __name__ == "__main__":
    unittest.main()
