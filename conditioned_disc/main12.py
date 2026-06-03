import os
import cv2
import numpy as np
from sklearn.cluster import KMeans


def process_switchgear_disc(image_path):
    if not os.path.exists("images12"):
        os.makedirs("images12")

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape

    # ==========================================
    # 1. Isolate the Disc & Define Centre Zone
    # ==========================================
    blur = cv2.GaussianBlur(img_gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=height // 2,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0,
    )

    mask = np.zeros_like(img_gray)
    inner_mask = np.zeros_like(img_gray)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]

        # Outer disc boundary
        cv2.circle(mask, (cx, cy), r - 5, 255, -1)

        # Inner exclusion zone (e.g., inner 65% of the disc)
        inner_r = int(r * 0.65)
        cv2.circle(inner_mask, (cx, cy), inner_r, 255, -1)

    mask_bool = mask > 0
    inner_mask_bool = inner_mask > 0

    img_masked = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite("images12/1_masked_disc.jpg", img_masked)

    # ==========================================
    # 2. Aggressive Contrast Enhancement
    # ==========================================
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_clahe_masked = cv2.bitwise_and(img_clahe, img_clahe, mask=mask)
    cv2.imwrite("images12/2_high_contrast.jpg", img_clahe_masked)

    # ==========================================
    # 3. Robust Black Groove Isolation
    # ==========================================
    # thresh = cv2.adaptiveThreshold(
    #     img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV, blockSize=21, C=10
    # )
    # thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

    # kernel_clean = np.ones((3, 3), np.uint8)
    # cleaned_grooves = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)

    # kernel_buffer = np.ones((5, 5), np.uint8)
    # black_mask_dilated = cv2.dilate(cleaned_grooves, kernel_buffer, iterations=2)
    # cv2.imwrite("images/3_black_grooves_mask.jpg", black_mask_dilated)

    # black_mask_bool = black_mask_dilated > 0

    manual_mask_path = "manual_mask.jpg"
    manual_mask = cv2.imread(manual_mask_path, cv2.IMREAD_GRAYSCALE)

    if manual_mask is None:
        print("Error: Manual mask file not found.")
        return

    # Force the mask to be strictly binary (just in case of compression artifacts)
    _, binary_mask = cv2.threshold(manual_mask, 127, 255, cv2.THRESH_BINARY)

    # Convert to boolean for the clustering exclusion logic
    black_mask_bool = binary_mask > 0

    # Save a copy to the output folder for your records
    cv2.imwrite("images12/3_black_grooves_mask.jpg", binary_mask)

    # ==========================================
    # 4. Surface Segmentation (Shiny vs. Unconditioned)
    # ==========================================
    surface_mask = mask_bool & (~black_mask_bool)
    pixels_to_cluster = img_clahe_masked[surface_mask].reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_to_cluster)
    centers = kmeans.cluster_centers_.flatten()

    sorted_idx = np.argsort(centers)
    unconditioned_lbl = sorted_idx[0]

    labels_full = np.full(img_gray.shape, -1)
    labels_full[surface_mask] = labels

    unconditioned_mask = labels_full == unconditioned_lbl

    # EXCLUSION LOGIC: Remove the inner circle from the unconditioned map
    unconditioned_mask = unconditioned_mask & (~inner_mask_bool)

    # Everything else on the surface becomes shiny
    shiny_mask = surface_mask & (~unconditioned_mask)

    # Visualisation map
    vis = np.zeros((height, width, 3), dtype=np.uint8)

    vis[shiny_mask] = [255, 255, 255]
    vis[unconditioned_mask] = [138, 73, 138]
    vis[black_mask_bool] = [0, 0, 0]

    vis = cv2.bitwise_and(vis, vis, mask=mask)
    cv2.imwrite("images12/4_final_segmentation.jpg", vis)

    # ==========================================
    # Output Calculations
    # ==========================================
    total_disc_area = np.sum(mask_bool)
    black_area = np.sum(black_mask_bool)
    surface_area = total_disc_area - black_area

    shiny_area = np.sum(shiny_mask)
    unconditioned_area = np.sum(unconditioned_mask)

    rel_shiny = (shiny_area / surface_area) * 100
    rel_uncond = (unconditioned_area / surface_area) * 100

    abs_shiny = (shiny_area / total_disc_area) * 100
    abs_uncond = (unconditioned_area / total_disc_area) * 100
    abs_black = (black_area / total_disc_area) * 100

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
    process_switchgear_disc("Bild.jpg")
