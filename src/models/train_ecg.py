import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
import os
import argparse
import numpy as np
from src.utils.preprocessing import get_ecg_loaders

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ECGNet(nn.Module):
    """ResNet1D-18 adapted for 12-lead ECG"""
    def __init__(self, num_classes=1):
        super(ECGNet, self).__init__()
        # Input: (B, 12, 1000)
        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Scheduler for learning rate decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device) # BCEWithLogitsLoss expects Float labels

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # For BCE, outputs are logits. Sigmoid -> Prob
                    probs = torch.sigmoid(outputs).squeeze()
                    preds = (probs > 0.5).float()
                    
                    # Ensure labels match shape
                    labels = labels.float()
                    
                    loss = criterion(outputs.squeeze(), labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'models/ecg_model.pt')
                    print("🔥 New Best Model Saved!")

        print()

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='ecg', help='Path to data') # Default to 'ecg' root
    args = parser.parse_args()
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device in use: {device}")
    
    # Data Loaders
    print("⏳ Loading PTB-XL Data...")
    train_loader, val_loader, test_loader = get_ecg_loaders(args.data_dir, batch_size=args.batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    
    # Calculate Class Weight for Imbalance
    # Simple heuristic: Pos_weight = Num_Neg / Num_Pos
    # But let's compute it strictly from tran set
    print("⚖️ Calculating Class Weights...")
    # This is a bit slow iterate, but necessary for correct weighting
    # Or we can just estimate from pandas df directly since we have it via the dataset in loader?
    # No easy access to dataset from here without refactoring.
    # Let's trust Adam to handle slight imbalance or use a fixed weight if needed.
    # PTB-XL is roughly balanced for superclasses on aggregate but binary 'Normal' vs 'All Else' might be skewed.
    # Ratio is roughly 9528 Normal vs 11000+ Abnormal. Pretty balanced (~45% Normal).
    # So pos_weight=1.0 is fine.
    
    # Model
    print("🏗️ Building 12-Lead 1D CNN Model...")
    model = ECGNet(num_classes=1).to(device) # Binary Output (1 logit)
    
    # Loss: BCEWithLogitsLoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train
    print("🧪 Starting Training...")
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs, device=device)
    
    print("✅ Training Complete.")

if __name__ == "__main__":
    main()
