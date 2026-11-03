"""Segmentation metrics from a confusion matrix (numpy only). Rows = ground truth, columns = prediction, ignored pixels are skipped."""
import numpy as np


def update_confusion(conf, pred, label, num_classes, ignore=255):
    """conf: int64 [C, C] accumulated in place. pred, label: integer arrays of the same shape."""
    pred, label = np.asarray(pred).ravel(), np.asarray(label).ravel()
    ok = (label != ignore) & (label < num_classes)
    idx = label[ok].astype(np.int64) * num_classes + pred[ok].astype(np.int64)
    conf += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf


def summarize(conf):
    """mIoU / mean Dice over the classes that occur in the ground truth or the prediction (NaN classes are counted and reported)."""
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diag(conf)
    gt, pr = conf.sum(1), conf.sum(0)
    union = gt + pr - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        dice = np.where(gt + pr > 0, 2 * tp / (gt + pr), np.nan)
        acc_c = np.where(gt > 0, tp / gt, np.nan)
    total = conf.sum()
    return {"mIoU": float(np.nanmean(iou)), "mDice": float(np.nanmean(dice)), "mAcc": float(np.nanmean(acc_c)),
            "pixel_acc": float(tp.sum() / total) if total else float("nan"), "n_nan_classes": int(np.isnan(iou).sum()),
            "n_pixels": int(total), "iou": [None if np.isnan(v) else float(v) for v in iou]}
