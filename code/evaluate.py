import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import os
from PIL import Image
from tqdm import tqdm

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

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='micro')
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')
    
    return f1, precision, recall

def main():
    # Paths
    image_dir = 'danbooru-images/danbooru-images/'
    json_dir = 'processed_data/'
    tag_mapping_file = 'processed_data/tag_mapping.json'
    val_split_file = 'processed_data/val_index.json'
    test_split_file = 'processed_data/test_index.json'
    model_path = 'best_model.pth'  # or 'checkpoint.pth'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tag mapping
    with open(tag_mapping_file, 'r') as f:
        tag_mapping = json.load(f)

    # Transformations (same as training but without augmentation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load validation and test datasets
    val_dataset = AnimeDataset(image_dir, json_dir, val_split_file, tag_mapping_file, transform=transform)
    test_dataset = AnimeDataset(image_dir, json_dir, test_split_file, tag_mapping_file, transform=transform)

    batch_size = 32
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(tag_mapping))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_f1, val_precision, val_recall = evaluate_model(model, val_loader, device)
    print(f"Validation Results:")
    print(f"F1 Score: {val_f1:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_f1, test_precision, test_recall = evaluate_model(model, test_loader, device)
    print(f"Test Results:")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")

if __name__ == "__main__":
    main()