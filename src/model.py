import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from preprocess import generate_multimodal_tensor

class MultimodalDisasterModel(nn.Module):
    def __init__(self, num_channels=6, num_classes=3):
        """
        Adapts a standard ResNet50 backbone to accept multi-sensor satellite data.
        """
        super().__init__()
        
        # Load a pretrained ResNet (it already understands earthly spatial features)
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # ---------------------------------------------------------
        # THE ECE HARDWARE-TO-SOFTWARE ADAPTATION
        # Original layer: accepts 3 channels (Red, Green, Blue)
        # New layer: accepts 6 channels (RGB + NIR + VV + VH)
        # ---------------------------------------------------------
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=num_channels, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=old_conv.bias
        )
        
        # Initialize the new channel weights to prevent breaking the pre-trained brain
        # We copy the original RGB weights into the new Radar/NIR channels to start
        with torch.no_grad():
            self.model.conv1.weight[:, :3, :, :] = old_conv.weight              # Keep RGB
            self.model.conv1.weight[:, 3:6, :, :] = old_conv.weight[:, :3, :, :] # Init NIR/Radar
            
        # Modify the final output head for our specific disaster types
        # e.g., Output classes: 0 = Flood, 1 = Wildfire, 2 = Landslide
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

def run_vision_inference(stacked_tensor: torch.Tensor) -> float:
    """
    Passes the 6-channel tensor through the AI and outputs a confidence score.
    """
    print("Initializing Multimodal ResNet50...")
    model = MultimodalDisasterModel(num_channels=6)
    model.eval() # Set to evaluation mode (turns off training nodes)

    # PyTorch expects a batch dimension: [Batch, Channels, Height, Width]
    # We add a batch of 1 using unsqueeze
    batched_tensor = stacked_tensor.unsqueeze(0)
    
    print(f"Feeding tensor shape {batched_tensor.shape} into the neural network...")
    
    with torch.no_grad():
        # Forward pass through the network
        raw_logits = model(batched_tensor)
        
        # Convert raw network output into probabilities (0.0 to 1.0)
        probabilities = torch.softmax(raw_logits, dim=1)
        
        # For this test, grab the confidence score of the highest predicted class
        max_confidence = torch.max(probabilities).item()
        
    return round(max_confidence, 3)

if __name__ == "__main__":
    print("--- ECE AI INFERENCE TEST ---")
    
    # 1. Grab a mock 6-channel tensor from your preprocessing script
    sample_bbox = [88.20, 22.45, 88.50, 22.70]
    input_tensor = generate_multimodal_tensor(sample_bbox)
    
    # 2. Run the tensor through the newly built AI model
    confidence_score = run_vision_inference(input_tensor)
    
    print(f"\n✅ Inference Complete!")
    print(f"Visual AI Confidence Score: {confidence_score * 100:.1f}%")