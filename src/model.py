"""
src/model.py
ECE Track: Vision AI model modified for 6-channel multimodal ingestion.
"""
import os
import urllib.request
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from preprocess import generate_multimodal_tensor

class MultimodalDisasterModel(nn.Module):
    def __init__(self, num_channels=6, num_classes=5):
        """
        Adapts a standard ResNet50 backbone to accept multi-sensor satellite data.
        """
        super().__init__()
        
        # Load a pretrained ResNet
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify first convolutional layer to accept 6 channels (RGB + NIR + VV + VH)
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=num_channels, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=old_conv.bias
        )
        
        # Initialize the new channel weights
        with torch.no_grad():
            self.model.conv1.weight[:, :3, :, :] = old_conv.weight              
            self.model.conv1.weight[:, 3:6, :, :] = old_conv.weight[:, :3, :, :]
            
        # Modify the final output head for 5 disaster types (FL, WF, TC, DR, EQ)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

def run_vision_inference(stacked_tensor: torch.Tensor) -> float:
    print("Initializing Multimodal ResNet50...")
    
    # 1. Setup paths and check if the model is already downloaded
    model = MultimodalDisasterModel(num_channels=6, num_classes=5)
    weights_path = "models/disaster_model.pth"
    
    # 2. Automatically download the model if it is missing
    if not os.path.exists(weights_path):
        print("Downloading model weights from GitHub...")
        os.makedirs("models", exist_ok=True)
        
        # PASTE YOUR LINK INSIDE THESE QUOTES:
        weights_url = "https://github.com/Krish582004/multimodal-disaster-detection/releases/download/v1.0/disaster_model.pth" 
        
        urllib.request.urlretrieve(weights_url, weights_path)
        print("Download complete.")

    # 3. Load the weights and run the prediction
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    
    batched_tensor = stacked_tensor.unsqueeze(0)
    with torch.no_grad():
        raw_logits = model(batched_tensor)
        probabilities = torch.softmax(raw_logits, dim=1)
        max_confidence = torch.max(probabilities).item()
        
    return round(max_confidence, 3)

if __name__ == "__main__":
    print("--- ECE AI INFERENCE TEST ---")
    sample_bbox = [88.20, 22.45, 88.50, 22.70]
    input_tensor = generate_multimodal_tensor(sample_bbox)
    confidence_score = run_vision_inference(input_tensor)
    
    print(f"\n✅ Inference Complete!")
    print(f"Visual AI Confidence Score: {confidence_score * 100:.1f}%")