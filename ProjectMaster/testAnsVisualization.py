import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# Import your argument parser and utilities from previously uploaded files.
from parseArgsTest import parse_args
from utils import AudioSpectrogramDataset, weights_init_xavier
from model import ResNet101v2  # Using your classification model from model.py

def plot_roc_curve(labels, probs, save_path='roc_curve.png'):
    fpr, tpr, _ = roc_curve(labels, probs)
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
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f'ROC curve saved to {save_path}')

def visualize_prediction(image_tensor, prob, label, save_path):
    # Convert the image tensor to a numpy array for visualization.
    # If your spectrogram images are single-channel, squeeze the channel dimension.
    image_np = image_tensor.squeeze().cpu().numpy()
    
    plt.figure()
    plt.imshow(image_np, cmap='gray')
    plt.title(f'Pred: {prob:.2f} | GT: {label}')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define directory paths.
    # We assume your test images and label file are in args.root/args.dataset.
    test_dir = os.path.join(args.root, args.dataset)
    # Assume a label file (e.g., "test_labels.txt") exists in that directory.
    label_file = os.path.join(test_dir, "test_labels.txt")
    
    # Define transforms – adjust normalization if needed for your spectrograms.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create the test dataset and dataloader using your AudioSpectrogramDataset.
    test_dataset = AudioSpectrogramDataset(test_dir, label_file, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    
    # Load your classification model.
    # For binary classification, we set num_classes=1.
    model = ResNet101v2(num_classes=1)
    model = model.to(device)
    model.apply(weights_init_xavier)
    
    # Load the checkpoint (ensure args.checkpoint_dir and args.model_dir are set in your parser).
    checkpoint_path = os.path.join(args.checkpoint_dir, args.model_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    all_labels = []
    all_probs = []
    
    # Create a directory for saving per-image visualizations.
    vis_dir = os.path.join(test_dir, "visualization_results")
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Running Inference")):
            images = images.to(device)
            outputs = model(images)  # Expected shape: (B, 1)
            # Convert logits to probabilities using sigmoid.
            probs = torch.sigmoid(outputs).squeeze()
            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_probs.extend(probs_np.tolist())
            all_labels.extend(labels_np.tolist())
            
            # Save visualization for each image in the batch.
            for j in range(images.size(0)):
                vis_path = os.path.join(vis_dir, f"image_{i * args.test_batch_size + j}.png")
                visualize_prediction(images[j], probs_np[j], labels_np[j], vis_path)
    
    # Plot overall ROC curve using the accumulated predictions.
    roc_save_path = os.path.join(vis_dir, "roc_curve.png")
    plot_roc_curve(all_labels, all_probs, save_path=roc_save_path)

if __name__ == '__main__':
    main()
