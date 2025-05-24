import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class SevenSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders named 0-9.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Collect all image paths and their respective labels
        for label in range(10):  # 0 to 9
            label_dir = os.path.join(root_dir, str(label))
            if not os.path.isdir(label_dir):
                continue
                
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_seven_segment_classifier(
    data_dir,
    val_dir=None,
    val_split=0.2,
    batch_size=256,
    num_epochs=3,
    learning_rate=0.001,
    model_save_path='checkpoints/seven_seg_classification_best.pth'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading and preparation code remains the same
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SevenSegmentDataset(data_dir, transform=transform)
    
    if val_dir:
        train_dataset = dataset
        val_dataset = SevenSegmentDataset(val_dir, transform=transform)
    else:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Add tqdm progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Training")
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar with current loss
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = running_corrects.double() / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        # Add tqdm progress bar for validation
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar with current loss
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_val_loss = running_loss / len(val_dataset)
        epoch_val_acc = running_corrects.double() / len(val_dataset)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        
        print(f'Train Loss: {epoch_train_loss:.4f} Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.4f}')
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved improved model to {model_save_path}')
    
    # Plotting code remains the same
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('training_history.png')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train seven-segment classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--val_dir', type=str, default=None, help='Path to validation set')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    train_seven_segment_classifier(
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )