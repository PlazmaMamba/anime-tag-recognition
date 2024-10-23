import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # Add these imports for AMP

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

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        json_path = os.path.join(self.json_dir, f"{img_id}.json")
        
        with open(json_path, 'r') as f:
            img_data = json.load(f)
        
        # Update the image path construction
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

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_f1, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),  # Save scaler state
        'best_f1': best_f1
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, scheduler, scaler, filename="checkpoint.pth"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])  # Load scaler state
        best_f1 = checkpoint['best_f1']
        print(f"Checkpoint loaded: {filename}")
        return start_epoch, best_f1
    else:
        return 0, 0.0

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda', checkpoint_interval=1, resume=False):
    scaler = GradScaler()  # Create GradScaler for AMP
    
    if resume:
        start_epoch, best_f1 = load_checkpoint(model, optimizer, scheduler, scaler)
    else:
        start_epoch, best_f1 = 0, 0.0
    
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            pbar = tqdm(dataloader, desc=f'{phase.capitalize()} - Loss: {0:.4f}')
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with autocast():  # Enable autocasting for mixed precision
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            scaler.scale(loss).backward()  # Scale the loss
                            scaler.step(optimizer)  # Perform scaled optimizer step
                            scaler.update()  # Update the scale for next iteration
                
                running_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_description(f'{phase.capitalize()} - Loss: {loss.item():.4f}')
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='micro')
            epoch_precision = precision_score(all_labels, all_preds, average='micro')
            epoch_recall = recall_score(all_labels, all_preds, average='micro')
            
            print(f'{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f}')
            
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
        
        # Save checkpoint at specified intervals
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_f1)
        
        # Check for user input to pause training
        user_input = input("Press 'q' to stop training, or any other key to continue: ")
        if user_input.lower() == 'q':
            print("Training paused. You can resume later using the checkpoint.")
            save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_f1)
            return model
    
    print(f'Best val F1: {best_f1:4f}')
    return model

def main():
    # Paths
    image_dir = 'danbooru-images/danbooru-images/'
    json_dir = 'processed_data/'
    tag_mapping_file = 'processed_data/tag_mapping.json'
    train_split_file = 'processed_data/train_index.json'
    val_split_file = 'processed_data/val_index.json'

    # Hyperparameters
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    train_dataset = AnimeDataset(image_dir, json_dir, train_split_file, tag_mapping_file, transform=transform)
    val_dataset = AnimeDataset(image_dir, json_dir, val_split_file, tag_mapping_file, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.tag_mapping))
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Add resume option
    resume = input("Do you want to resume from a checkpoint? (y/n): ").lower() == 'y'

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                                num_epochs=num_epochs, device=device, checkpoint_interval=1, resume=resume)

    print("Training complete!")

if __name__ == "__main__":
    main()