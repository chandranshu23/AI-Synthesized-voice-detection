import torch
import torch.nn as nn

# Binary Cross Entropy Loss with Logits for stable training.
# This loss function automatically applies a sigmoid to the outputs.
class BinaryClassificationLoss(nn.Module):
    def __init__(self, pos_weight=None):
        """
        pos_weight: a weight of positive examples. Useful if you have class imbalance.
        """
        super(BinaryClassificationLoss, self).__init__()
        # If pos_weight is provided, pass it as a tensor.
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        """
        logits: raw model outputs (not passed through sigmoid)
        targets: binary labels (0 or 1)
        """
        loss = self.criterion(logits, targets.float())
        return loss

# AverageMeter class remains useful for logging training progress.
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


