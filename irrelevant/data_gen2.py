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
    # Read as greyscale directly
    img_gray = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error: Template image not found.")
        return

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

    # Isolate black grooves (using the strict method)
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

    # Pre-calculate radial gradient for edge clustering and shading
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    radial_gradient = np.clip(dist_from_center / r, 0, 1)

    log_data = []

    # ==========================================
    # 2. Batch Generation Loop
    # ==========================================
    for i in range(1, num_images + 1):
        target_uncond_ratio = np.random.uniform(0.05, 0.75)
        
        # Generate procedural map to dictate where the rough patches grow
        raw_noise = np.random.rand(height, width).astype(np.float32)
        smooth_noise = cv2.GaussianBlur(raw_noise, (101, 101), 20)
        texture_map = smooth_noise + (radial_gradient * 0.3)
        
        surface_values = texture_map[surface_bool]
        split_percentile = (1.0 - target_uncond_ratio) * 100.0
        threshold_val = np.percentile(surface_values, split_percentile)
        
        # Exact boolean masks for ground truth math
        uncond_mask = (texture_map >= threshold_val) & surface_bool
        shiny_mask = (texture_map < threshold_val) & surface_bool

        # --- TEXTURE GENERATION ---
        
        # 1. Shiny Texture: Bright, smooth, slight rolling gradient
        raw_shiny = np.random.normal(210, 10, (height, width)).astype(np.float32)
        shiny_texture = cv2.GaussianBlur(raw_shiny, (31, 31), 0)

        # 2. Unconditioned Texture: Crunchy, pitted, high-contrast
        raw_rough = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        # Slight blur creates tiny blobs instead of single-pixel static
        raw_rough = cv2.GaussianBlur(raw_rough, (3, 3), 0) 
        # Apply CLAHE to force harsh contrast (creates the deep pitting effect)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        rough_base = clahe.apply(raw_rough)
        # Darken the rough layer to match the reference image
        rough_texture = cv2.convertScaleAbs(rough_base, alpha=0.7, beta=-20).astype(np.float32)

        # Combine the textures using the generated mask
        synth_img = np.where(uncond_mask, rough_texture, shiny_texture)
        
        # Apply a subtle 3D radial shadow (darkens the outer edge slightly for realism)
        shadow_map = np.clip(1.0 - (radial_gradient ** 2) * 0.3, 0, 1)
        synth_img = synth_img * shadow_map
        
        # Convert to final greyscale format
        synth_img = np.clip(synth_img, 0, 255).astype(np.uint8)

        # Force the absolute black grooves and background on top
        synth_img[grooves_bool] = 0
        synth_img[~mask_bool] = 0
        
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

    csv_path = os.path.join(output_dir, "ground_truth_log.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Shiny %", "Unconditioned %", "Black Grooves %"])
        writer.writerows(log_data)
        
    print(f"\nBatch complete. Log saved to {csv_path}")

if __name__ == "__main__":
    generate_synthetic_batch("Bild.jpg", num_images=20)
