"""
Segmentation Metrics Utilities
===============================
Dice, IoU, per-class segmentation metrics for the UMOD pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute Dice score for binary masks.

    Parameters
    ----------
    pred : np.ndarray, binary
    target : np.ndarray, binary
    smooth : float

    Returns
    -------
    float : Dice coefficient
    """
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute IoU (Jaccard) score for binary masks."""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score_seg(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute pixel-wise precision for binary masks."""
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + smooth) / (tp + fp + smooth)


def recall_score_seg(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute pixel-wise recall for binary masks."""
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + smooth) / (tp + fn + smooth)


def per_class_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                      num_classes: int = 8) -> Dict[int, Dict[str, float]]:
    """
    Compute per-class Dice, IoU, Precision, Recall on full-resolution masks.

    Parameters
    ----------
    pred_mask : np.ndarray (H, W), values 0..num_classes-1
    gt_mask : np.ndarray (H, W), values 0..num_classes-1
    num_classes : int

    Returns
    -------
    dict : {class_id: {'dice': float, 'iou': float, 'precision': float, 'recall': float}}
    """
    results = {}
    for cls_id in range(num_classes):
        pred_binary = (pred_mask == cls_id).astype(np.float32)
        gt_binary = (gt_mask == cls_id).astype(np.float32)

        # Skip if class not present in either
        if pred_binary.sum() == 0 and gt_binary.sum() == 0:
            results[cls_id] = {'dice': 1.0, 'iou': 1.0, 'precision': 1.0, 'recall': 1.0}
        else:
            results[cls_id] = {
                'dice': dice_score(pred_binary, gt_binary),
                'iou': iou_score(pred_binary, gt_binary),
                'precision': precision_score_seg(pred_binary, gt_binary),
                'recall': recall_score_seg(pred_binary, gt_binary),
            }

    return results


def mean_dice(pred_mask: np.ndarray, gt_mask: np.ndarray,
              num_classes: int = 8, exclude_background: bool = True) -> float:
    """
    Compute mean Dice across classes (optionally excluding background).

    Parameters
    ----------
    pred_mask, gt_mask : np.ndarray (H, W)
    num_classes : int
    exclude_background : bool

    Returns
    -------
    float : mean Dice
    """
    metrics = per_class_metrics(pred_mask, gt_mask, num_classes)
    start_cls = 1 if exclude_background else 0
    dices = [metrics[c]['dice'] for c in range(start_cls, num_classes)]
    return float(np.mean(dices))


def binary_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute binary (foreground vs background) Dice score.
    For direct comparison with paper baseline (Dice=0.877).
    """
    pred_fg = (pred_mask > 0).astype(np.float32)
    gt_fg = (gt_mask > 0).astype(np.float32)
    return dice_score(pred_fg, gt_fg)


def pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute overall pixel accuracy."""
    return float((pred_mask == gt_mask).sum() / pred_mask.size)


def confusion_matrix_seg(pred_mask: np.ndarray, gt_mask: np.ndarray,
                         num_classes: int = 8) -> np.ndarray:
    """
    Compute pixel-level confusion matrix.

    Returns
    -------
    np.ndarray of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for cls_true in range(num_classes):
        for cls_pred in range(num_classes):
            cm[cls_true, cls_pred] = ((gt_mask == cls_true) & (pred_mask == cls_pred)).sum()
    return cm


def aggregate_metrics(per_image_metrics: List[Dict[int, Dict[str, float]]],
                      num_classes: int = 8) -> Dict:
    """
    Aggregate per-image, per-class metrics into summary statistics.

    Parameters
    ----------
    per_image_metrics : list of per_class_metrics() outputs
    num_classes : int

    Returns
    -------
    dict : {class_id: {metric: {'mean': float, 'std': float}}}
    """
    from collections import defaultdict

    class_metrics = defaultdict(lambda: defaultdict(list))

    for image_metrics in per_image_metrics:
        for cls_id, metrics in image_metrics.items():
            for metric_name, value in metrics.items():
                class_metrics[cls_id][metric_name].append(value)

    summary = {}
    for cls_id in range(num_classes):
        summary[cls_id] = {}
        for metric_name in ['dice', 'iou', 'precision', 'recall']:
            values = class_metrics[cls_id][metric_name]
            if values:
                summary[cls_id][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
            else:
                summary[cls_id][metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
                }

    return summary
