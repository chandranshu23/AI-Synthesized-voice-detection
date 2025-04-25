import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision Training

# Import argument parser, dataset, loss, and utilities.
from parsArgsTrain import parse_args
from utils import AudioSpectrogramDataset, weights_init_xavier
from loss import BinaryClassificationLoss, AverageMeter
from metric import ClassificationMetrics
from loadParamData import loadDataset, loadParams  
from model import ResNet101v2

# Enable CuDNN Benchmarking for optimized performance
torch.backends.cudnn.benchmark = True

def main():
    args = parse_args()
    
    # Load checkpoint only once in the main process.
    checkpoint_path = ""
    model = None
    start_epoch = args.start_epoch
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model = ResNet101v2(num_classes=1).cuda()
        model.load_state_dict(checkpoint['state_dict'])
        # Optionally, update start_epoch based on checkpoint if available:
        # start_epoch = checkpoint.get('epoch', args.start_epoch) + 1
    else:
        print("No checkpoint found, training from scratch.")
        start_epoch = 0

    trainer = Trainer(args, model=model, start_epoch=start_epoch)
    trainer.run()

class Trainer(object):
    def __init__(self, args, model=None, start_epoch=0):
        self.args = args
        self.start_epoch = start_epoch  
        # Use torch.cuda.amp.GradScaler() with no device argument.
        self.scaler = torch.amp.GradScaler('cuda')
        self.loss_meter = AverageMeter()
        # Set dataset directory
        self.dataset_dir = os.path.join(args.root, args.dataset)
        train_label_file = os.path.join(self.dataset_dir, "train_labels.txt")
        test_label_file  = os.path.join(self.dataset_dir, "test_labels.txt")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Create datasets and loaders
        self.train_dataset, self.test_dataset, self.eval_dataset = loadDataset(args.root, args.dataset)
        self.train_loader = DataLoader(
            AudioSpectrogramDataset(self.dataset_dir, os.path.join(self.dataset_dir, "train.txt"), transform=transform), 
            batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader  = DataLoader(
            AudioSpectrogramDataset(self.dataset_dir, os.path.join(self.dataset_dir, "test.txt"), transform=transform), 
            batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.eval_loader  = DataLoader(
            AudioSpectrogramDataset(self.dataset_dir, os.path.join(self.dataset_dir, "eval.txt"), transform=transform), 
            batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)  # Eval Set
                
        # Initialize model: use the provided model if available.
        if model is not None:
            self.model = model
        else:
            self.model = ResNet101v2(num_classes=1).cuda()
            self.model.apply(weights_init_xavier)
        print("Model Initialized")
        
        # Loss function
        self.criterion = BinaryClassificationLoss(pos_weight=args.pos_weight if hasattr(args, 'pos_weight') else None)
        
        # Classification metrics
        self.metric = ClassificationMetrics(threshold=args.threshold if hasattr(args, 'threshold') else 0.5)
        
        # Initialize optimizer with the model parameters.
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        
        # Choose scheduler based on command-line argument.
        if args.scheduler == "OneCycleLR":
            self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                      total_steps=args.epochs * len(self.train_loader))
        elif args.scheduler == "ReduceLROnPlateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        else:
            self.scheduler = None
        
        self.best_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training")

        for images, labels in train_bar:
            images = images.cuda()
            labels = labels.cuda().float().unsqueeze(1)
            
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):  # Mixed Precision Training
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * images.size(0)
            train_bar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
        
        return running_loss / len(self.train_dataset)

    def evaluate(self, epoch, mode="test"):
        self.model.eval()
    
        loader = self.eval_loader if mode == "eval" else self.test_loader
        desc = f"Epoch {epoch} {mode.capitalize()} Evaluation"

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=desc):
                images, labels = images.cuda(), labels.cuda().float().unsqueeze(1)
                with torch.amp.autocast(device_type='cuda'):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                self.metric.update(logits, labels)

        eval_loss = self.loss_meter.avg
        metrics_log = self.metric.log_metrics()
    
        return eval_loss, metrics_log

    def save_checkpoint(self, epoch, eval_loss):
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.best_epoch = epoch
            save_path = os.path.join("result", f"{self.args.model}_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),  # Save optimizer state
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,  # Save scheduler state
                'loss': eval_loss,
            }, save_path)
            print(f"Checkpoint saved at epoch {epoch} with loss {eval_loss:.4f}")

    def run(self):
        # Use self.start_epoch to resume training.
        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss = self.train_epoch(epoch)
            eval_loss, eval_metrics_log = self.evaluate(epoch, mode="eval")

            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
            print(eval_metrics_log)

            # Save Eval Loss to log file
            eval_log_file = os.path.join("result", self.args.save_dir, "eval_log.txt")
            with open(eval_log_file, "a") as f:
                f.write(f"Epoch {epoch}: Eval Loss: {eval_loss:.4f}\n")
                f.write(eval_metrics_log + "\n")

            if self.scheduler is not None:
                # For ReduceLROnPlateau, step with eval_loss
                if self.args.scheduler == "ReduceLROnPlateau":
                    self.scheduler.step(eval_loss)
                else:
                    self.scheduler.step()

            self.save_checkpoint(epoch, eval_loss)

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()} is available.")
    main()
