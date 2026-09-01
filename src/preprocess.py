import numpy as np
import torch
import rasterio
from rasterio.warp import reproject, Resampling

def normalize_optical(band_data: np.ndarray) -> np.ndarray:
    """
    Normalizes Sentinel-2 optical reflectance values.
    Standard L2A optical data ranges roughly from 0 to 10000.
    """
    # Clip extreme bright anomalies (like cloud reflections) and scale to 0.0 - 1.0
    clipped = np.clip(band_data, 0, 10000)
    return clipped / 10000.0

def normalize_radar(band_data: np.ndarray) -> np.ndarray:
    """
    Normalizes Sentinel-1 SAR (Radar) backscatter values.
    SAR data is logarithmic (decibels), typically ranging from -30 dB to 0 dB.
    """
    # Clip extreme noise and scale to 0.0 - 1.0
    clipped = np.clip(band_data, -30.0, 0.0)
    return (clipped + 30.0) / 30.0

def generate_multimodal_tensor(bbox: list, height: int = 256, width: int = 256) -> torch.Tensor:
    """
    Simulates the spatial alignment of Optical and Radar signals.
    In production, rasterio.warp.reproject aligns the SAR array to the Optical transform.
    
    Args:
        bbox (list): Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
        height (int): Target pixel height for the AI model (default 256)
        width (int): Target pixel width for the AI model (default 256)
        
    Returns:
        torch.Tensor: A 6-channel tensor [Channels, Height, Width] ready for TorchGeo.
    """
    print(f"Aligning spatial footprints for Bounding Box: {bbox}...")
    
    # 1. Simulate the normalized Optical Signals (Red, Green, Blue, Near-Infrared)
    # Shape: (4, 256, 256)
    optical_signals = np.random.uniform(0.0, 1.0, size=(4, height, width)).astype(np.float32)
    
    # 2. Simulate the normalized Radar Signals (VV polarization, VH polarization)
    # Shape: (2, 256, 256)
    radar_signals = np.random.uniform(0.0, 1.0, size=(2, height, width)).astype(np.float32)
    
    # 3. Channel Concatenation (The core multimodal fusion step)
    # We stack the optical and radar matrices perfectly on top of each other.
    stacked_array = np.concatenate((optical_signals, radar_signals), axis=0)
    
    # 4. Convert the fused NumPy array into a PyTorch Tensor
    fused_tensor = torch.from_numpy(stacked_array)
    
    return fused_tensor

if __name__ == "__main__":
    # Test the ECE signal processing block
    sample_bbox = [88.20, 22.45, 88.50, 22.70]
    print("--- ECE SIGNAL PROCESSING TEST ---")
    
    try:
        # Generate the multi-channel tensor
        ai_input_tensor = generate_multimodal_tensor(sample_bbox)
        
        print("\n✅ Tensor Successfully Generated!")
        print(f"Data Type: {ai_input_tensor.dtype}")
        print(f"Tensor Shape: {ai_input_tensor.shape} -> [Channels, Height, Width]")
        
        # Verify the structure for the Vision AI
        channels = ai_input_tensor.shape[0]
        if channels == 6:
            print("\nChannel Breakdown:")
            print("- Channels 0-3: Optical Light (RGB + Near-Infrared)")
            print("- Channels 4-5: Microwave Radar Backscatter (VV + VH)")
            print("\nStatus: Ready for TorchGeo Model Ingestion.")
            
    except Exception as e:
        print(f"Error during signal processing: {e}")