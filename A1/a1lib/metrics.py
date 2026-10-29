"""Metrics from logits (NumPy only): top-k, NLL, ECE, Brier, macro-F1."""
import numpy as np


def softmax(logits):
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def topk_acc(logits, y, k=1):
    top = np.argpartition(-logits, kth=min(k, logits.shape[1] - 1), axis=1)[:, :k] if k < logits.shape[1] else None
    if top is None:
        return 1.0
    return float((top == y[:, None]).any(axis=1).mean())


def nll(logits, y):
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def brier(logits, y):
    p = softmax(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def ece(logits, y, n_bins=15):
    p = softmax(logits)
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def macro_f1(pred, y, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def summarize(logits, y, n_classes):
    logits = np.asarray(logits, dtype=np.float32)
    y = np.asarray(y)
    pred = logits.argmax(axis=1)
    return {"top1": float((pred == y).mean()), "top5": topk_acc(logits, y, 5), "nll": nll(logits, y),
            "ece": ece(logits, y), "brier": brier(logits, y), "macro_f1": macro_f1(pred, y, n_classes), "n": int(len(y))}
