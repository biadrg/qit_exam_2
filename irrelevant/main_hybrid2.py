"""
Hybrid approach combining best practices from main4.py and main13.py
- Uses bilateral filter + CLAHE for robust preprocessing (from main13)
- Uses LAB color space for better perceptual clustering (from main4)
- Uses dual manual masks for grooves and center (from main13)
- Explicit 3-cluster identification with margin handling (from main4)
"""

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans


def brightness(lab_value):
    """Extract brightness (L channel) from LAB color space"""
    if len(lab_value.shape) == 1:
        return lab_value[0]
    return lab_value[0]


def process_switchgear_disc(image_path):
    if not os.path.exists("images_hybrid2"):
        os.makedirs("images_hybrid2")

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape

    # ==========================================
    # 1. Isolate the Disc (Masking)
    # ==========================================
    blur_hough = cv2.GaussianBlur(img_gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur_hough,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=height // 2,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0,
    )

    mask = np.zeros_like(img_gray)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]
        cv2.circle(mask, (cx, cy), r - 5, 255, -1)

    mask_bool = mask > 0
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("images_hybrid2/1_masked_disc.jpg", img_masked)

    # ==========================================
    # 2. Advanced Preprocessing (Bilateral + CLAHE)
    # ==========================================
    # Bilateral filter: removes central grain/noise while keeping pit edges sharp
    # Apply to BGR image to preserve color information
    smoothed = cv2.bilateralFilter(img_masked, d=11, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite("images_hybrid2/2_smoothed.jpg", smoothed)

    # Convert to LAB for CLAHE (better to enhance luminance in LAB space)
    lab_smooth = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)

    # Apply CLAHE to L channel only (not a and b which are color info)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    lab_smooth[:, :, 0] = clahe.apply(lab_smooth[:, :, 0])

    # Convert back to BGR for further processing
    img_clahe_masked = cv2.cvtColor(lab_smooth, cv2.COLOR_LAB2BGR)
    cv2.imwrite("images_hybrid2/3_high_contrast.jpg", img_clahe_masked)

    # ==========================================
    # 3. Load Manual Masks (Grooves & Center)
    # ==========================================
    # Load both manual masks for exclusion zones
    manual_grooves = cv2.imread("manual_mask.jpg", cv2.IMREAD_GRAYSCALE)
    manual_center = cv2.imread("center_mask_optimised.jpg", cv2.IMREAD_GRAYSCALE)

    if manual_grooves is None:
        print("Error: manual_mask.jpg not found.")
        return

    # Threshold to make strictly binary
    _, grooves_binary = cv2.threshold(manual_grooves, 127, 255, cv2.THRESH_BINARY)
    _, center_binary = cv2.threshold(manual_center, 127, 255, cv2.THRESH_BINARY)

    grooves_bool = grooves_binary > 0
    center_bool = center_binary > 0

    # Center mask optional - use if available
    # center_bool = np.zeros_like(mask_bool)
    # if manual_center is not None:
    #     _, center_binary = cv2.threshold(manual_center, 127, 255, cv2.THRESH_BINARY)
    #     center_bool = center_binary > 0

    # Create exclusion zone: grooves OR center
    exclusion_zone = grooves_bool | center_bool

    # Define surface area: disc AND NOT grooves AND NOT center
    surface_mask = mask_bool & (~grooves_bool) & (~center_bool)
    cv2.imwrite(
        "images_hybrid2/4_surface_mask.jpg", (surface_mask.astype(np.uint8) * 255)
    )

    # ==========================================
    # 4. Convert to LAB for 3D Clustering (Better discrimination than grayscale)
    # ==========================================
    # Convert the contrast-enhanced image to LAB color space
    # LAB provides perceptually uniform feature space for better clustering
    lab_img = cv2.cvtColor(img_clahe_masked, cv2.COLOR_BGR2LAB)

    # Extract only surface pixels for clustering
    pixels_to_cluster = lab_img[surface_mask].reshape(-1, 3)

    # ==========================================
    # 5. KMeans Clustering (3 clusters)
    # ==========================================
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_to_cluster)
    centers = kmeans.cluster_centers_

    # Map labels back to full image dimensions
    labels_full = np.full((height, width), -1, dtype=np.int32)
    labels_full[surface_mask] = labels
    labels_img = labels_full

    # ==========================================
    # 6. Cluster Identification by Brightness
    # ==========================================
    # Extract brightness (L channel) for each cluster center
    brightness_vals = np.array([brightness(c) for c in centers])

    # Sort clusters by brightness
    sorted_idx = np.argsort(brightness_vals)

    # Identify clusters by brightness ranking
    darkest_cluster = sorted_idx[0]  # Unconditioned/pitted surface
    brightest_cluster = sorted_idx[-1]  # Shiny/conditioned surface
    middle_clusters = sorted_idx[1:-1]  # Mid-tone regions

    # ==========================================
    # 7. Create Region Masks
    # ==========================================
    # Shiny (conditioned) regions: brightest cluster on valid surface
    shiny_mask = (labels_img == brightest_cluster) & surface_mask

    # Unconditioned (pitted) regions: darkest cluster on valid surface
    unconditioned_mask = (labels_img == darkest_cluster) & surface_mask

    # Combine middle clusters (noise/transition regions)
    middle_mask = np.zeros((height, width), dtype=bool)
    for c in middle_clusters:
        middle_mask |= (labels_img == c) & surface_mask

    # ==========================================
    # 8. Apply Margin Buffer (from main4 approach)
    # ==========================================
    # Create dilated margin around exclusion zones as safety buffer
    kernel3 = np.ones((3, 3), np.uint8)
    exclusion_u8 = exclusion_zone.astype(np.uint8) * 255
    margin = cv2.dilate(exclusion_u8, kernel3, iterations=2) > 0

    # Remove margin from surface regions
    shiny_mask &= ~margin
    unconditioned_mask &= ~margin
    middle_mask &= ~margin

    total_disc_area = np.sum(mask_bool)
    exclusion_area = np.sum(exclusion_zone)
    # total_disc_area -= exclusion_area  # Adjust total area to exclude grooves/center
    surface_area = total_disc_area - exclusion_area

    shiny_pixels = np.sum(shiny_mask)
    unconditioned_pixels = np.sum(unconditioned_mask)
    middle_pixels = np.sum(middle_mask)

    # Calculate percentages based only on shiny + unconditioned pixels
    # This ensures the two percentages add up to 100%
    active_surface_pixels = shiny_pixels + unconditioned_pixels

    if active_surface_pixels > 0:
        shiny_pct = (shiny_pixels / active_surface_pixels) * 100
        uncond_pct = (unconditioned_pixels / active_surface_pixels) * 100
    else:
        shiny_pct = 0
        uncond_pct = 0

    print("\nResults:")
    print(f"  Shiny:      {shiny_pct:.2f}%")
    print(f"  Unconditioned:   {uncond_pct:.2f}%\n")

    # ==========================================
    # 11. Create Color-Coded Visualization
    # ==========================================
    vis = np.zeros((height, width, 3), dtype=np.uint8)

    # Apply colors in order (later overwrites earlier)
    vis[shiny_mask] = [255, 255, 255]  # White for shiny
    vis[unconditioned_mask] = [138, 73, 138]  # Purple for unconditioned
    # vis[middle_mask] = [255, 255, 255]  # White for mid-tone (blend with shiny)
    vis[center_bool] = [255, 255, 255]  # Force center to white (blends with shiny)
    # vis[grooves_bool] = [0, 0, 0]  # Force grooves to black (last, overwrites)

    # Apply disc mask to keep only circular region
    vis = cv2.bitwise_and(vis, vis, mask=mask)
    cv2.imwrite("images_hybrid2/5_final_segmentation.jpg", vis)

    # print("Output directory: images_hybrid/")
    # print("Segmentation saved to: 5_final_segmentation.jpg")


if __name__ == "__main__":
    process_switchgear_disc("Bild.jpg")
