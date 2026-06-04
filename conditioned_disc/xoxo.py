# convert BGR image to LAB colour space for clustering
lab = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2LAB)

# ==========================================
# 1. Load Masks BEFORE Clustering
# ==========================================
# Load grooves mask
black_mask = cv2.imread("set_mask.jpg", cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(black_mask, 127, 255, cv2.THRESH_BINARY)
black_mask_bool = binary_mask > 0
cv2.imwrite("images4/7_image_mask.jpg", (black_mask_bool.astype(np.uint8) * 255))

# Load centre mask
center_mask = cv2.imread("center_mask.jpg", cv2.IMREAD_GRAYSCALE)
_, center_binary = cv2.threshold(center_mask, 127, 255, cv2.THRESH_BINARY)
center_bool = center_binary > 0

# Create margin around black grooves to prevent edge bleeding
kernel3 = np.ones((3, 3), np.uint8)
black_margin = cv2.dilate(black_mask_bool.astype(np.uint8) * 255, kernel3, iterations=2) > 0

# ==========================================
# 2. Define Surface and Extract Valid Pixels
# ==========================================
# Combine grooves and centre into one exclusion zone
exclusion_zone = black_margin | center_bool

# The valid surface is the disc MINUS the exclusion zone
surface_mask = mask_circle & (~exclusion_zone)

# 2D image to 1D array, extracting ONLY valid surface pixels
pixels = lab.reshape(-1, 3)
surface_flat = surface_mask.flatten()
pixels_to_cluster = pixels[surface_flat]

# ==========================================
# 3. Run KMeans on Surface Only
# ==========================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=30)
labels = kmeans.fit_predict(pixels_to_cluster)

# Map labels back to full image dimensions
labels_full = np.full(surface_flat.shape, -1)
labels_full[surface_flat] = labels
labels_img = labels_full.reshape(height, width)
centers = kmeans.cluster_centers_

# Sort clusters by brightness
brightness_vals = np.array([brightness(c) for c in centers])
sorted_idx = np.argsort(brightness_vals)

white_cluster = sorted_idx[-1]
target_clusters = sorted_idx[1:-1] 

# ==========================================
# 4. Generate Final Masks and Visualisation
# ==========================================
# Isolate regions based strictly on the valid surface mask
white_mask = (labels_img == white_cluster) & surface_mask

target_mask = np.zeros((height, width), dtype=bool)
for c in target_clusters:
    target_mask |= (labels_img == c) & surface_mask

# Area measurements
total_pixels = np.sum(surface_mask)
white_pixels = np.sum(white_mask)
edge_pixels = np.sum(target_mask)

white_pct = 100 * white_pixels / total_pixels if total_pixels > 0 else 0
edge_pct = 100 * edge_pixels / total_pixels if total_pixels > 0 else 0

print("\nResults:")
print(f"White pixels (Shiny): {white_pct:.2f}%")
print(f"Edge pixels (Unconditioned): {edge_pct:.2f}%\n")

# Apply colours
vis = np.zeros((height, width, 3), dtype=np.uint8)
vis[white_mask] = [255, 255, 255]       # White for shiny
vis[target_mask] = [138, 73, 138]       # Purple for unconditioned

# Force exclusions to specific colours
vis[center_bool] = [255, 255, 255]      # Centre forced to white
vis[black_mask_bool] = [0, 0, 0]        # Grooves forced to black

# Clean up outer background
vis_circle = cv2.bitwise_and(vis, vis, mask=mask.astype(np.uint8))
cv2.imwrite("images4/8_image_highlighted.jpg", vis_circle)
