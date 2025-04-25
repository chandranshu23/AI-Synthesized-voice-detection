import torch
import cupy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ClassificationMetrics:
    def __init__(self, threshold=0.5):
        """
        threshold: Threshold to convert probabilities into binary predictions.
        """
        self.threshold = threshold
        self.reset()

    def update(self, logits, targets):
        """
        Accumulate predictions and targets.
        Args:
            logits: Raw model outputs (before sigmoid), a tensor of shape (batch_size, 1)
            targets: Ground truth labels, a tensor of shape (batch_size, 1) with values 0 or 1.
        """
        # Apply sigmoid to convert logits to probabilities.
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()

        # Convert probabilities to binary predictions using the threshold.
        preds = (probs >= self.threshold).astype(int)
        
        self.all_targets.extend(targets.tolist())
        self.all_preds.extend(preds.tolist())
        self.all_probs.extend(probs.tolist())

    def compute(self):
        """
        Compute and return the classification metrics.
        Returns:
            accuracy, precision, recall, f1, roc_auc
        """
        accuracy = accuracy_score(self.all_targets, self.all_preds)
        precision = precision_score(self.all_targets, self.all_preds, zero_division=0)
        recall = recall_score(self.all_targets, self.all_preds, zero_division=0)
        f1 = f1_score(self.all_targets, self.all_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(self.all_targets, self.all_probs)
        except ValueError:
            roc_auc = None
            print("Warning: ROC-AUC score is undefined (only one class present).")
        
        return accuracy, precision, recall, f1, roc_auc

    def log_metrics(self):
        """
        Compute and log the metrics.
        Returns:
            A formatted string of the metrics.
        """
        accuracy, precision, recall, f1, roc_auc = self.compute()
        log_str = (
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1 Score: {f1:.4f}\n" +
            (f"ROC-AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC: Undefined")
        )
        print(log_str)
        return log_str
    
    def reset(self):
        """
        Reset the stored predictions and targets.
        """
        self.all_targets = []
        self.all_preds = []
        self.all_probs = []
