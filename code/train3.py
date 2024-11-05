import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
import gc
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class AnimeDataset(Dataset):
    def __init__(self, image_dir, json_dir, split_file, tag_mapping_file, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        
        with open(split_file, 'r') as f:
            self.image_ids = json.load(f)
        
        with open(tag_mapping_file, 'r') as f:
            self.tag_mapping = json.load(f)
        
        self.num_classes = len(self.tag_mapping)
        self.reverse_mapping = {v: k for k, v in self.tag_mapping.items()}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        json_path = os.path.join(self.json_dir, f"{img_id}.json")
        
        with open(json_path, 'r') as f:
            img_data = json.load(f)
        
        subfolder = f"{int(img_id) % 1000:04d}"
        img_filename = os.path.basename(img_data['file_path'])
        img_path = os.path.join(self.image_dir, subfolder, img_filename)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        tags = torch.zeros(self.num_classes)
        for tag in img_data['tags']:
            if tag in self.tag_mapping:
                tags[self.tag_mapping[tag]] = 1
        
        return image, tags

def create_run_directory():
    """Create a directory for the current training run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"training_run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    return run_dir

def save_metrics(metrics_dict, save_path):
    """Save metrics to CSV"""
    df = pd.DataFrame([metrics_dict])
    mode = 'a' if os.path.exists(save_path) else 'w'
    header = not os.path.exists(save_path)
    df.to_csv(save_path, mode=mode, header=header, index=False)

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, metrics, run_dir):
    """Save checkpoint with metrics"""
    checkpoint_path = os.path.join(run_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, run_dir, epoch):
    """Evaluate model and compute metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    for inputs, labels in tqdm(dataloader, desc='Evaluating'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item() * inputs.size(0)
        preds = (outputs > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Clear cache
        del outputs, preds
        torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    metrics = {
        'loss': total_loss / len(dataloader.dataset),
        'f1': f1_score(all_labels, all_preds, average='micro'),
        'precision': precision_score(all_labels, all_preds, average='micro'),
        'recall': recall_score(all_labels, all_preds, average='micro')
    }
    
    # Save metrics
    metrics_path = os.path.join(run_dir, "metrics", "metrics.csv")
    save_metrics(metrics, metrics_path)
    
    return metrics

# Only showing the modified train_model function, the rest remains the same

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=10, device='cuda', run_dir=None):
    if run_dir is None:
        run_dir = create_run_directory()
    
    scaler = GradScaler()
    best_f1 = 0.0
    
    # Initialize empty lists for plotting
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Training - Loss: 0.0000')
        
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Update progress bar with current loss
            current_loss = loss.item()
            pbar.set_description(f'Training - Loss: {current_loss:.4f}')
            
            # Clear memory
            del outputs, loss
            torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, device, run_dir, epoch)
        val_losses.append(val_metrics['loss'])
        
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, scaler, val_metrics, run_dir)
        
        # Update learning rate based on validation F1 score
        scheduler.step(val_metrics['f1'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')
        
        # Print metrics
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val F1: {val_metrics["f1"]:.4f}')
        print(f'Val Precision: {val_metrics["precision"]:.4f}')
        print(f'Val Recall: {val_metrics["recall"]:.4f}')
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'training_history.png'))
    plt.close()
    
    return model, run_dir

def main():
    # Paths and hyperparameters
    image_dir = 'danbooru-images/danbooru-images/'
    json_dir = 'processed_data/'
    tag_mapping_file = 'processed_data/tag_mapping.json'
    train_split_file = 'processed_data/train_index.json'
    val_split_file = 'processed_data/val_index.json'

    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Get the weights and transforms from the updated ResNet50 weights
    weights = ResNet50_Weights.DEFAULT
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        weights.transforms(),  # Use the transforms from the weights
    ])

    # Datasets and DataLoaders
    train_dataset = AnimeDataset(image_dir, json_dir, train_split_file, tag_mapping_file, transform=transform)
    val_dataset = AnimeDataset(image_dir, json_dir, val_split_file, tag_mapping_file, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True, persistent_workers=True)

    # Model with updated initialization
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.tag_mapping))
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                   factor=0.1, patience=3)

    # Create run directory
    run_dir = create_run_directory()

    # Train the model
    trained_model, run_dir = train_model(model, train_loader, val_loader, criterion, 
                                       optimizer, scheduler, num_epochs=num_epochs, 
                                       device=device, run_dir=run_dir)

    print(f"Training complete! Results saved in {run_dir}")

if __name__ == "__main__":
    main()