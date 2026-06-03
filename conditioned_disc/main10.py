import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def process_switchgear_disc(image_path):
    # Ensure the output directory exists for intermediate steps
    if not os.path.exists("images"):
        os.makedirs("images")

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found. Check the file path.")
        return

    # Convert to greyscale for processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape

    # ==========================================
    # 1. Isolate the Disc (Masking)
    # ==========================================
    # We blur the greyscale image to reduce noise for the circle detection algorithm.
    blur = cv2.GaussianBlur(img_gray, (9, 9), 2)

    # We use HoughCircles to find the exact boundary of the outer disc.
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=height // 2,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0
    )

    # Create a blank mask to zero out the background.
    mask = np.zeros_like(img_gray)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]
        # Draw a solid white circle. We subtract a 5-pixel buffer from the radius to avoid capturing edge artifacts.
        cv2.circle(mask, (cx, cy), r - 5, 255, -1)

    mask_bool = mask > 0

    # Apply the mask to the original image to black out the background entirely.
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("images/1_masked_disc.jpg", img_masked)

    # ==========================================
    # 2. Aggressive Contrast Enhancement
    # ==========================================
    # We apply CLAHE to aggressively enhance local contrast. 
    # This separates the rough, unconditioned textures from the smooth metal far better than basic alpha/beta scaling.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # Re-apply the disc mask to keep the background clean and completely black.
    img_clahe_masked = cv2.bitwise_and(img_clahe, img_clahe, mask=mask)
    cv2.imwrite("images/2_high_contrast.jpg", img_clahe_masked)

    # ==========================================
    # 3. Robust Black Groove Isolation
    # ==========================================
    # We find the baseline threshold using Otsu's method on the pixels inside the disc.
    pixels_inside = img_gray[mask_bool]
    otsu_thresh, _ = cv2.threshold(pixels_inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # We multiply the threshold by 0.8 to be stricter, capturing only the deepest black pixels of the grooves.
    black_mask_raw = (img_gray < otsu_thresh * 0.8).astype(np.uint8) * 255
    black_mask_raw = cv2.bitwise_and(black_mask_raw, black_mask_raw, mask=mask)

    # We use morphological operations to clean speckles and close small gaps in the detected lines.
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    cleaned_grooves = cv2.morphologyEx(black_mask_raw, cv2.MORPH_OPEN, kernel3)
    cleaned_grooves = cv2.morphologyEx(cleaned_grooves, cv2.MORPH_CLOSE, kernel5)

    # We dilate the mask to create a physical exclusion buffer around the grooves.
    # This stops the unconditioned (purple) cluster from bleeding into the black edges during segmentation.
    black_mask_dilated = cv2.dilate(cleaned_grooves, kernel3, iterations=2)
    cv2.imwrite("images/3_black_grooves_mask.jpg", black_mask_dilated)

    black_mask_bool = black_mask_dilated > 0

    # ==========================================
    # 4. Surface Segmentation (Shiny vs. Unconditioned)
    # ==========================================
    # We define the active surface by taking the full disc mask and excluding the dilated black grooves.
    surface_mask = mask_bool & (~black_mask_bool)

    # We pull only the valid surface pixels from the high-contrast CLAHE image to prepare them for clustering.
    pixels_to_cluster = img_clahe_masked[surface_mask].reshape(-1, 1)

    # We use KMeans with 2 clusters to group the pixels into "Shiny" (bright) and "Unconditioned" (dark/rough).
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_to_cluster)
    centers = kmeans.cluster_centers_.flatten()

    # We sort the cluster centres by intensity. The darker cluster contains the rough blemishes.
    sorted_idx = np.argsort(centers)
    unconditioned_lbl = sorted_idx[0]
    shiny_lbl = sorted_idx[1]

    # Map the clustered labels back onto a 2D array matching the image dimensions.
    labels_full = np.full(img_gray.shape, -1)
    labels_full[surface_mask] = labels

    unconditioned_mask = (labels_full == unconditioned_lbl)
    shiny_mask = (labels_full == shiny_lbl)

    # Create the final colour-coded visualization map.
    vis = np.zeros((height, width, 3), dtype=np.uint8)

    # Set White for shiny metal areas
    vis[shiny_mask] = [255, 255, 255]
    
    # Set Purple for unconditioned areas (OpenCV uses BGR format)
    vis[unconditioned_mask] = [138, 73, 138]
    
    # Set Black for the central grooves
    vis[black_mask_bool] = [0, 0, 0]

    # Apply the master mask one last time to ensure the background remains zeroed out.
    vis = cv2.bitwise_and(vis, vis, mask=mask)
    cv2.imwrite("images/4_final_segmentation.jpg", vis)

    # ==========================================
    # Output Calculations
    # ==========================================
    total_disc_area = np.sum(mask_bool)
    black_area = np.sum(black_mask_bool)
    surface_area = total_disc_area - black_area

    shiny_area = np.sum(shiny_mask)
    unconditioned_area = np.sum(unconditioned_mask)

    # Relative Composition Calculations
    rel_shiny = (shiny_area / surface_area) * 100
    rel_uncond = (unconditioned_area / surface_area) * 100

    # Absolute Composition Calculations
    abs_shiny = (shiny_area / total_disc_area) * 100
    abs_uncond = (unconditioned_area / total_disc_area) * 100
    abs_black = (black_area / total_disc_area) * 100

    # Console Output formatting exactly as requested
    print("\nAnalysis 1: Relative Conditioning (Excluding Grooves)")
    print(f"Total Area = {surface_area} pixels")
    print(f"Shiny Area: {rel_shiny:.2f}%")
    print(f"Unconditioned Area: {rel_uncond:.2f}%")

    print("\nAnalysis 2: Absolute Composition (Including Grooves)")
    print(f"Total Area = {total_disc_area} pixels")
    print(f"Shiny Area: {abs_shiny:.2f}%")
    print(f"Unconditioned Area: {abs_uncond:.2f}%")
    print(f"Black Grooves Area: {abs_black:.2f}%\n")

if __name__ == "__main__":
    # Point this to your specific image file
    process_switchgear_disc("Bild.jpg")
