
import unittest

import torch
from e3lib.tokenizers import PatchTokenizer, RowTokenizer


class TestRowTokenizer(unittest.TestCase):

    def test_output_shape(self):
        """Check that row tokenization returns the expected batch, sequence, and feature dimensions."""
        tokenizer = RowTokenizer(dim=64)
        x = torch.randn(2, 3, 32, 32)

        tokens = tokenizer(x)

        self.assertEqual(tokens.shape, (2, 32, 64))

    def test_row_order(self):
        """Check that each output token contains the pixels from the corresponding input row."""
        # Identity projection lets us inspect raw row values
        tokenizer = RowTokenizer(dim=96)

        with torch.no_grad():
            tokenizer.proj.weight.copy_(torch.eye(96))
            tokenizer.proj.bias.zero_()

        x = torch.arange(
            3 * 32 * 32, dtype=torch.float32
        ).reshape(1, 3, 32, 32)

        tokens = tokenizer(x)

        for row in (0, 15, 31):
            expected = x[0, :, row, :].reshape(-1)

            self.assertTrue(
                torch.equal(tokens[0, row], expected)
            )

    def test_backward(self):
        """Check that backpropagation produces finite gradients for the row projection."""
        tokenizer = RowTokenizer(dim=64)
        x = torch.randn(2, 3, 32, 32)

        loss = tokenizer(x).square().mean()
        loss.backward()

        grad = tokenizer.proj.weight.grad

        assert grad is not None
        self.assertTrue(torch.isfinite(grad).all())

class TestPatchTokenizer(unittest.TestCase):

    def test_output_shape(self):
        """Check that patch tokenization returns the expected dimensions and token count."""
        tokenizer = PatchTokenizer(dim=64, patch_size=4)
        x = torch.randn(2, 3, 32, 32)

        tokens = tokenizer(x)

        self.assertEqual(tokens.shape, (2, 64, 64))
        self.assertEqual(tokenizer.n_steps, 64)

    def test_patch_order(self):
        """Check that every output token matches its input patch in row-major order."""
        # 48 features = 3 channels * 4 * 4
        tokenizer = PatchTokenizer(dim=48, patch_size=4)

        # Make the projection preserve all raw patch values.
        with torch.no_grad():
            tokenizer.proj.weight.copy_(
                torch.eye(48).reshape(48, 3, 4, 4)
            )
            assert tokenizer.proj.bias is not None
            tokenizer.proj.bias.zero_()

        x = torch.arange(
            3 * 32 * 32, dtype=torch.float32
        ).reshape(1, 3, 32, 32)

        tokens = tokenizer(x)

        # Verify every patch in row-major order.
        for row in range(8):
            for col in range(8):
                index = row * 8 + col

                expected = x[
                    0,
                    :,
                    row * 4:(row + 1) * 4,
                    col * 4:(col + 1) * 4
                ].reshape(-1)

                self.assertTrue(
                    torch.equal(tokens[0, index], expected)
                )

    def test_backward(self):
        """Check that backpropagation produces finite gradients for the patch projection."""
        tokenizer = PatchTokenizer(dim=64, patch_size=4)
        x = torch.randn(2, 3, 32, 32)

        loss = tokenizer(x).square().mean()
        loss.backward()

        grad = tokenizer.proj.weight.grad

        self.assertIsNotNone(grad)
        assert grad is not None
        self.assertTrue(torch.isfinite(grad).all())
        
if __name__ == "__main__":
    unittest.main()
