"""PYTHONPATH=. python -m unittest discover -s common/tests -v     (numpy only, no torch, no download)"""
import unittest

import numpy as np

from common.cifar import CLASSES, channel_stats, fake_cifar, get_splits


class Cifar(unittest.TestCase):
    def test_channel_stats_matches_numpy(self):
        x = np.random.default_rng(1).integers(0, 256, (300, 32, 32, 3), dtype=np.uint8)
        mean, std = channel_stats(x, chunk=64)
        ref = x.reshape(-1, 3).astype(np.float64) / 255
        self.assertTrue(np.allclose(mean, ref.mean(0), atol=1e-12))
        self.assertTrue(np.allclose(std, ref.std(0), atol=1e-9))

    def test_channel_stats_constant_image(self):
        x = np.full((4, 32, 32, 3), 255, dtype=np.uint8)
        mean, std = channel_stats(x)
        self.assertEqual(mean, (1.0, 1.0, 1.0))
        self.assertTrue(all(s < 1e-6 for s in std))

    def test_fake_splits_are_disjoint_and_labelled(self):
        s, h = get_splits(fake=True)
        self.assertEqual(h, "fake")
        self.assertEqual(len(s["train"][1]) + len(s["val"][1]), len(fake_cifar()[1]))
        self.assertEqual(s["train"][0].shape[1:], (32, 32, 3))
        self.assertEqual(np.bincount(s["val"][1], minlength=10).tolist(), [20] * 10)

    def test_ten_classes(self):
        self.assertEqual(len(CLASSES), 10)


if __name__ == "__main__":
    unittest.main()


class Fingerprint(unittest.TestCase):
    def test_fingerprint_detects_one_changed_pixel_and_is_stable(self):
        from common.cifar import fingerprint
        r = np.random.default_rng(0)
        a = (r.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8), r.integers(0, 10, 5), r.integers(0, 256, (2, 32, 32, 3), dtype=np.uint8), r.integers(0, 10, 2))
        f = fingerprint(*a)
        self.assertEqual(f, fingerprint(*[x.copy() for x in a]))
        b = [x.copy() for x in a]
        b[0][0, 0, 0, 0] ^= 1
        self.assertNotEqual(f, fingerprint(*b))
