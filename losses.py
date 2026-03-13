import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """
    Computes Dice Loss for multiclass segmentation.
    Expects raw logits (un-softmaxed) and integer class labels.
    Handles ignore_index=255 safely before one-hot encoding.
    """
    def __init__(self, num_classes=10, smooth=1e-5, ignore_index=255):
        super(MulticlassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Mask out ignore index pixels before anything else
        # Replace 255 with 0 temporarily so one_hot doesn't crash
        valid_mask = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode: (B, H, W) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets_clean, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Zero out ignored pixels in both probs and targets
        valid_mask_expanded = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_expanded
        targets_one_hot = targets_one_hot * valid_mask_expanded

        dice_loss = 0.0
        for i in range(self.num_classes):
            prob_c = probs[:, i, :, :]
            target_c = targets_one_hot[:, i, :, :]

            intersection = torch.sum(prob_c * target_c)
            union = torch.sum(prob_c) + torch.sum(target_c)

            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice_score)

        return dice_loss / self.num_classes


class SegmentationLoss(nn.Module):
    """
    Combined Weighted CrossEntropy + Dice Loss.
    ce_weight and dice_weight control the balance.
    ignore_index=255 is handled in both losses.
    """
    def __init__(self, class_weights, num_classes=10,
                 ce_weight=0.6, dice_weight=0.4, ignore_index=255):
        super(SegmentationLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        # ignore_index=255 tells CrossEntropyLoss to skip those pixels
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )
        self.dice = MulticlassDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index
        )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)