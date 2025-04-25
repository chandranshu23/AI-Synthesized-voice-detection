import os
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from torch.cuda.amp import autocast  # Mixed Precision Inference

# Import argument parser, dataset, loss, and model.
from parseArgsTest import parse_args
from utils import AudioSpectrogramDataset, weights_init_xavier, AverageMeter
from loss import BinaryClassificationLoss
from metric import ClassificationMetrics
from model import ResNet101v2

# Enable CuDNN Benchmarking for optimized performance
torch.backends.cudnn.benchmark = True

class Tester:
    def __init__(self, args):
        self.args = args
        
        # Define dataset path and test labels file.
        dataset_dir = os.path.join(args.root, args.dataset)
        test_label_file = os.path.join(dataset_dir, "test.txt")
        
        # Define normalization for spectrogram images.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Load the test dataset.
        self.test_dataset = AudioSpectrogramDataset(dataset_dir, test_label_file, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.test_batch_size, 
                                      shuffle=False, num_workers=4, pin_memory=True)
        
        # Load classification model.
        self.model = ResNet101v2(num_classes=1).cuda()
        self.model.apply(weights_init_xavier)
        
        # Load checkpoint.
        checkpoint_path = os.path.join("result", args.model_dir)
        checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # Initialize evaluation loss function and metrics.
        self.criterion = BinaryClassificationLoss()
        self.metric = ClassificationMetrics(threshold=args.threshold if hasattr(args, 'threshold') else 0.5)
        self.loss_meter = AverageMeter()

        print("Model loaded. Running evaluation...")

    def evaluate(self):
        """Runs inference on the test dataset and computes performance metrics."""
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.cuda()
                labels = labels.cuda().float().unsqueeze(1)

                with autocast():  # Enable AMP for inference
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                self.loss_meter.update(loss.item(), images.size(0))
                self.metric.update(logits, labels)

        # Compute final classification metrics.
        accuracy, precision, recall, f1, roc_auc = self.metric.compute()
        print(f"Test Loss: {self.loss_meter.avg:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Save test results to a .mat file.
        result_dir = os.path.join("result", self.args.st_model + "_test_results.mat")
        scio.savemat(result_dir, {
            'loss': self.loss_meter.avg,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        })
        print(f"Results saved to {result_dir}")

        # Plot ROC curve.
        self.plot_roc_curve()

    def plot_roc_curve(self):
        """Plots and saves the ROC curve."""
        labels, probs = self.metric.all_targets, self.metric.all_probs
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        roc_save_path = os.path.join("result", self.args.st_model + "_roc_curve.png")
        plt.savefig(roc_save_path, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {roc_save_path}")


def main():
    args = parse_args()
    tester = Tester(args)
    tester.evaluate()

if __name__ == "__main__":
    main()