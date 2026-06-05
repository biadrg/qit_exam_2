import os
import cv2
import numpy as np

# Create the dedicated output directory
os.makedirs('contrast1', exist_ok=True)

# Load the image
img = cv2.imread('image_a95d1a.jpg')

if img is None:
    print("Error: Could not find 'image_a95d1a.jpg'. Please check the path.")
else:
    # Convert to grayscale to assess brightness levels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialise a blank black image for the result
    result = np.zeros_like(img)

    # Define masks based on brightness intensity
    # Brighter areas become completely white
    white_mask = gray > 135
    
    # Darker patches around the edges become purple (BGR: 180, 0, 180)
    purple_mask = (gray <= 135) & (gray > 45)

    # Apply the colours to the masks
    result[white_mask] = [255, 255, 255]
    result[purple_mask] = [180, 0, 180] 

    # Save the output to the specified folder
    output_path = 'contrast1/processed_image.jpg'
    cv2.imwrite(output_path, result)
    print(f"Image successfully saved to {output_path}")
