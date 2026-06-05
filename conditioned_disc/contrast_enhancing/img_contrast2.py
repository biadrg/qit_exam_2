import os
import numpy as np
from PIL import Image

# Create the dedicated output directory
os.makedirs('contrast2', exist_ok=True)

try:
    # Load the image and convert to grayscale
    img = Image.open('image_a95d1a.jpg')
    gray_np = np.array(img.convert('L'))

    # Get dimensions and initialise an empty RGB array
    h, w = gray_np.shape
    result_np = np.zeros((h, w, 3), dtype=np.uint8)

    # Segment channels directly to map specific intensities
    # White areas (>135) get maximum RGB values [255, 255, 255]
    # Darker spots (between 45 and 135) get a purple hue [140, 0, 180]
    
    # Red Channel
    result_np[:, :, 0] = np.where(gray_np > 135, 255, np.where(gray_np > 45, 140, 0))
    # Green Channel
    result_np[:, :, 1] = np.where(gray_np > 135, 255, 0)
    # Blue Channel
    result_np[:, :, 2] = np.where(gray_np > 135, 255, np.where(gray_np > 45, 180, 0))

    # Convert back to an image and save
    result_img = Image.fromarray(result_np)
    output_path = 'contrast2/processed_image.jpg'
    result_img.save(output_path)
    print(f"Image successfully saved to {output_path}")

except FileNotFoundError:
    print("Error: Could not find 'image_a95d1a.jpg'. Please check the path.")
