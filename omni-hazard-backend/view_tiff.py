import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# Replace with your exact downloaded filename
filename = "scan_composite_2026-09-05_to_2026-09-19.tiff"

try:
    with rasterio.open(filename) as src:
        print(f"Bands: {src.count}, Width: {src.width}, Height: {src.height}")

        # Plot the satellite data using a grayscale color map
        plt.figure(figsize=(10, 10))
        show(src, cmap='gray')
        plt.show()
except FileNotFoundError:
    print(f"Could not find {filename}. Make sure it is in this folder.")