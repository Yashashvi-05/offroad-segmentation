import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

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
        valid_mask = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0

        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets_clean, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

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
    Combined Weighted Focal + Dice Loss.
    """
    def __init__(self, class_weights, num_classes=10, ignore_index=255):
        super(SegmentationLoss, self).__init__()
        self.focal = FocalLoss(weight=class_weights, gamma=2.0, ignore_index=ignore_index)
        self.dice = MulticlassDiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits, targets):
        return 0.6 * self.focal(logits, targets) + 0.4 * self.dice(logits, targets)