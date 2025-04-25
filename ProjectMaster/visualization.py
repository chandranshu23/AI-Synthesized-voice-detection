import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from parseArgsTest import parse_args            # updated argument parser for your project
from utils import AudioSpectrogramDataset  # your test dataset loader for spectrograms
from model import ResNet101v2  # your classification model
from utils import weights_init_xavier    # any necessary utilities

class Visualizer:
    def __init__(self, args):
        self.args = args
        # Directory where your dataset is stored
        self.dataset_dir = os.path.join(args.root, args.dataset)
        
        # Directory for saving visualizations
        self.vis_dir = os.path.join(self.dataset_dir, 'visualization_results', f"{args.st_model}_results")
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Define the transforms – adjust normalization as needed for your spectrograms.
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])
        
        # Create the test dataset and loader.
        # Expected to return (image, label, image_id) if possible.
        label_file = os.path.join(self.dataset_dir, "test_labels.txt")
        self.testset = AudioSpectrogramDataset(self.dataset_dir, label_file, transform=input_transform, suffix=args.suffix)
        self.test_loader = DataLoader(dataset=self.testset, batch_size=args.test_batch_size,
                                      num_workers=args.workers, drop_last=False)
        
        # Load your classification model.
        self.model = ResNet101v2(num_classes=1)
        self.model = self.model.cuda()
        self.model.apply(weights_init_xavier)
        
        # Load checkpoint – adjust checkpoint path as needed.
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.model_dir))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # Lists for accumulating predictions and ground truth for ROC curve plotting.
        self.all_labels = []
        self.all_probs = []
    
    def visualize_predictions(self):
        # Process test data and save per-image visualization.
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                # Expect batch to be (images, labels, image_ids). If image_ids are not provided,
                # we create a default naming.
                if len(batch) == 3:
                    images, labels, image_ids = batch
                else:
                    images, labels = batch
                    image_ids = [f"img_{i*self.args.test_batch_size + j}" for j in range(images.size(0))]
                
                images = images.cuda()
                outputs = self.model(images)  # Shape: (B, 1) logits
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs >= self.args.threshold).astype(int)
                labels_np = labels.cpu().numpy().flatten()
                
                # Store for ROC curve plotting.
                self.all_labels.extend(labels_np.tolist())
                self.all_probs.extend(probs.tolist())
                
                # Visualize each image in the batch.
                for j in range(len(images)):
                    img_tensor = images[j].cpu()
                    # Remove channel dimension if single-channel and convert to numpy array.
                    img_np = img_tensor.squeeze().numpy()
                    
                    fig, ax = plt.subplots()
                    ax.imshow(img_np, cmap='gray')
                    title_str = (f"ID: {image_ids[j]}, Pred: {preds[j]}, "
                                 f"Prob: {probs[j]:.2f}, GT: {labels_np[j]}")
                    ax.set_title(title_str)
                    ax.axis('off')
                    
                    save_path = os.path.join(self.vis_dir, f"{image_ids[j]}_viz.png")
                    fig.savefig(save_path, bbox_inches='tight')
                    plt.close(fig)
        
        # After visualizing all images, plot the ROC curve.
        self.plot_roc_curve()
    
    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.all_labels, self.all_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        roc_save_path = os.path.join(self.vis_dir, "roc_curve.png")
        plt.savefig(roc_save_path, bbox_inches='tight')
        plt.close()

def main(args):
    vis = Visualizer(args)
    vis.visualize_predictions()

if __name__ == "__main__":
    args = parse_args()
    main(args)
