
"""Unit tests for E3 recurrent classifiers."""

import unittest

import torch
from e3lib.models import GRUClassifier, LSTMClassifier
from torch import nn


class TestLSTMClassifier(unittest.TestCase):

    def test_output_shape(self):
        x = torch.randn(2, 3, 32, 32)

        for tokenizer in ("row", "patch4"):
            with self.subTest(tokenizer=tokenizer):
                model = LSTMClassifier(
                    tokenizer=tokenizer
                )

                logits = model(x)

                self.assertEqual(logits.shape, (2, 10))

    def test_parameter_count(self):
        expected = {
            "row": 40138,
            "patch4": 37066,
        }

        for tokenizer, count in expected.items():
            with self.subTest(tokenizer=tokenizer):
                model = LSTMClassifier(
                    tokenizer=tokenizer
                )

                actual = sum(
                    p.numel()
                    for p in model.parameters()
                    if p.requires_grad
                )

                self.assertEqual(actual, count)

    def test_backward(self):
        model = LSTMClassifier(tokenizer="row")

        x = torch.randn(2, 3, 32, 32)
        # x= x.to(next(model.parameters()).device)

        labels = torch.tensor([1, 7])

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)

        loss.backward()

        self.assertTrue(torch.isfinite(loss))

        self.assertTrue(torch.isfinite(logits).all())

        for name, param in model.named_parameters():
            with self.subTest(parameter=name):
                grad = param.grad
                if grad is None:
                    self.fail(f"Gradient for {name} is None")
                self.assertTrue(
                    torch.isfinite(grad).all()
                )

    def test_independent_forward_calls(self):
        model = LSTMClassifier(tokenizer="row")
        model.eval()

        x = torch.randn(2, 3, 32, 32)
        x = x.to(next(model.parameters()).device)

        other = torch.randn(2, 3, 32, 32)
        other = other.to(next(model.parameters()).device)

        with torch.no_grad():
            first = model(x)
            model(other)
            second = model(x)

        self.assertTrue(
            torch.allclose(first, second)
        )

    def test_invalid_tokenizer(self):
        with self.assertRaises(ValueError):
            LSTMClassifier(tokenizer="unknown")


class TestGRUClassifier(unittest.TestCase):

    def test_output_shape(self):
        x = torch.randn(2, 3, 32, 32)

        for tokenizer in ("row", "patch4"):
            with self.subTest(tokenizer=tokenizer):
                model = GRUClassifier(
                    tokenizer=tokenizer
                )

                logits = model(x)

                self.assertEqual(logits.shape, (2, 10))

    def test_parameter_count(self):
        expected = {
            "row": 31818,
            "patch4": 28746,
        }

        for tokenizer, count in expected.items():
            with self.subTest(tokenizer=tokenizer):
                model = GRUClassifier(
                    tokenizer=tokenizer
                )

                actual = sum(
                    p.numel()
                    for p in model.parameters()
                    if p.requires_grad
                )

                self.assertEqual(actual, count)

    def test_backward(self):
        model = GRUClassifier(tokenizer="row")

        x = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([1, 7])

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)

        loss.backward()

        self.assertTrue(torch.isfinite(loss))

        


        for name, param in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(param.grad)
                self.assertTrue(
                    torch.isfinite(param.grad).all()
                )

    def test_independent_forward_calls(self):
        model = GRUClassifier(tokenizer="row")
        model.eval()

        x = torch.randn(2, 3, 32, 32)
        other = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            first = model(x)
            model(other)
            second = model(x)

        self.assertTrue(
            torch.allclose(first, second)
        )

    def test_invalid_tokenizer(self):
        with self.assertRaises(ValueError):
            GRUClassifier(tokenizer="unknown")


if __name__ == "__main__":
    unittest.main()
