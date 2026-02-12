"""
Loss Functions for Segmentation and Classification
====================================================
CompoundLoss (Dice + Focal) for segmentation
MultiFocalLoss for multi-class classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.

    Computes per-class Dice loss and returns weighted average.
    """

    def __init__(self, num_classes: int = 8, smooth: float = 1e-6,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C, H, W) float
        targets : (B, H, W) long
        """
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # Sum over batch, height, width
        intersection = (probs * targets_onehot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_onehot.sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.class_weights is not None:
            weights = self.class_weights.to(dice.device)
            dice_loss = 1.0 - (dice * weights).sum() / weights.sum()
        else:
            dice_loss = 1.0 - dice.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # per-class weights, shape (C,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, C, H, W) float
        targets : (B, H, W) long
        """
        alpha = self.alpha.to(logits.device) if self.alpha is not None else None
        ce_loss = F.cross_entropy(logits, targets, weight=alpha,
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CompoundLoss(nn.Module):
    """
    Compound loss: weighted sum of Dice + Focal loss.

    Used for segmentation training (both U-Net and SegFormer).
    """

    def __init__(self, num_classes: int = 8,
                 dice_weight: float = 0.5,
                 focal_weight: float = 0.5,
                 focal_gamma: float = 2.0,
                 class_weights: Optional[np.ndarray] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        cw_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

        self.dice_loss = DiceLoss(num_classes=num_classes, class_weights=cw_tensor)
        self.focal_loss = FocalLoss(alpha=cw_tensor, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d_loss = self.dice_loss(logits, targets)
        f_loss = self.focal_loss(logits, targets)
        return self.dice_weight * d_loss + self.focal_weight * f_loss


class MultiFocalLoss(nn.Module):
    """
    Multi-class Focal Loss for severity classification (3 classes).

    Extension of binary FocalLoss from src/03_ft_transformer.py to multi-class.
    """

    def __init__(self, num_classes: int = 3, gamma: float = 2.0,
                 class_weights: Optional[np.ndarray] = None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

        if class_weights is not None:
            self.register_buffer('weight', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, num_classes) float
        targets : (B,) long
        """
        weight = self.weight.to(logits.device) if self.weight is not None else None
        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
