"""
train.py
Fine-tunes the 6-channel ResNet50 model using standard RGB disaster imagery.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from src.model import MultimodalDisasterModel

# Define Hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation pipeline for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom Dataset to convert 3-channel RGB to 6-channel Multimodal format
class RGBSatelliteDataset(Dataset):
    def __init__(self, data_dir):
        # ImageFolder automatically infers the 5 classes from your subfolder names
        self.dataset = datasets.ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rgb_tensor, label = self.dataset[idx]
        
        # Duplicate the 3 RGB channels to create a 6-channel tensor
        # Channels 0,1,2 = Real RGB | Channels 3,4,5 = Simulated Radar/NIR
        six_channel_tensor = torch.cat((rgb_tensor, rgb_tensor), dim=0)
        
        return six_channel_tensor, label

def train_model():
    print(f"Training on device: {DEVICE}")
    
    # Path to your training data
    train_dir = "data/train"
    if not os.path.exists(train_dir):
        print(f"Error: Directory '{train_dir}' not found.")
        return

    # Instantiate the Dataset and DataLoader
    train_dataset = RGBSatelliteDataset(data_dir=train_dir)
    
    if len(train_dataset) == 0:
        print("Error: No images found. Ensure your .jpg/.png files are inside data/train/<class_name>/ folders.")
        return
        
    print(f"Found {len(train_dataset)} training images across {len(train_dataset.dataset.classes)} classes.")
    print(f"Classes: {train_dataset.dataset.classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate Model
    model = MultimodalDisasterModel(num_channels=6, num_classes=5).to(DEVICE)
    
    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    # Save Checkpoint
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/disaster_model.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Model weights saved to {checkpoint_path}")

if __name__ == "__main__":
    train_model()