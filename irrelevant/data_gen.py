import os
import cv2
import numpy as np
import csv

def generate_synthetic_batch(template_image_path, num_images=20):
    output_dir = "synthetic_data"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. Extract Original Geometry & Masks
    # ==========================================
    img = cv2.imread(template_image_path)
    if img is None:
        print("Error: Template image not found.")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape

    # Detect outer disc
    blur = cv2.GaussianBlur(img_gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=height // 2,
        param1=50, param2=30, minRadius=0, maxRadius=0
    )

    circle_mask = np.zeros_like(img_gray)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]
        cv2.circle(circle_mask, (cx, cy), r - 5, 255, -1)
    else:
        print("Error: Could not detect disc in template.")
        return

    mask_bool = circle_mask > 0

    # Isolate black grooves using the strict adaptive threshold method
    thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, blockSize=21, C=10
    )
    thresh = cv2.bitwise_and(thresh, thresh, mask=circle_mask)
    kernel_clean = np.ones((3, 3), np.uint8)
    cleaned_grooves = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    kernel_buffer = np.ones((5, 5), np.uint8)
    grooves_dilated = cv2.dilate(cleaned_grooves, kernel_buffer, iterations=2)
    
    grooves_bool = grooves_dilated > 0
    surface_bool = mask_bool & (~grooves_bool)

    total_disc_area = np.sum(mask_bool)
    grooves_area = np.sum(grooves_bool)
    abs_grooves_pct = (grooves_area / total_disc_area) * 100

    # Pre-calculate radial gradient for edge clustering
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    radial_gradient = np.clip(dist_from_center / r, 0, 1)

    log_data = []

    # ==========================================
    # 2. Batch Generation Loop
    # ==========================================
    for i in range(1, num_images + 1):
        # Pick a random target percentage for unconditioned spots
        target_uncond_ratio = np.random.uniform(0.05, 0.75)
        
        # Generate low-frequency procedural noise (smooth blobs)
        raw_noise = np.random.rand(height, width).astype(np.float32)
        smooth_noise = cv2.GaussianBlur(raw_noise, (101, 101), 20)
        
        # Combine noise with radial gradient (pushes higher values to the edges)
        texture_map = smooth_noise + (radial_gradient * 0.3)
        
        # Extract only the valid surface pixels to find the exact threshold
        surface_values = texture_map[surface_bool]
        
        # Find the precise split point to hit the target percentage
        split_percentile = (1.0 - target_uncond_ratio) * 100.0
        threshold_val = np.percentile(surface_values, split_percentile)
        
        uncond_mask = (texture_map >= threshold_val) & surface_bool
        shiny_mask = (texture_map < threshold_val) & surface_bool

        # Create the blank synthetic image canvas
        synth_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Base colours for the metal
        shiny_base = np.array([210, 210, 210], dtype=np.int16)
        uncond_base = np.array([120, 110, 120], dtype=np.int16) # Darker, rougher grey
        
        # Add high-frequency noise to simulate physical texture/grain
        grain = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
        
        # Apply textures to the respective masks
        synth_img[shiny_mask] = np.clip(shiny_base + grain[shiny_mask], 0, 255)
        synth_img[uncond_mask] = np.clip(uncond_base + grain[uncond_mask], 0, 255)
        
        # Overlay the solid black grooves perfectly
        synth_img[grooves_bool] = [0, 0, 0]
        
        # Force the background to black
        synth_img[~mask_bool] = [0, 0, 0]
        
        # ==========================================
        # 3. Save File and Log Ground Truth
        # ==========================================
        filename = f"synthetic_disc_{i:02d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), synth_img)
        
        abs_shiny_pct = (np.sum(shiny_mask) / total_disc_area) * 100
        abs_uncond_pct = (np.sum(uncond_mask) / total_disc_area) * 100
        
        log_data.append([
            filename, 
            f"{abs_shiny_pct:.2f}%", 
            f"{abs_uncond_pct:.2f}%", 
            f"{abs_grooves_pct:.2f}%"
        ])
        
        print(f"Generated {filename} -> Unconditioned: {abs_uncond_pct:.2f}%")

    # Write the CSV log file
    csv_path = os.path.join(output_dir, "ground_truth_log.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Shiny %", "Unconditioned %", "Black Grooves %"])
        writer.writerows(log_data)
        
    print(f"\nBatch complete. Log saved to {csv_path}")

if __name__ == "__main__":
    generate_synthetic_batch("Bild.jpg", num_images=20)
