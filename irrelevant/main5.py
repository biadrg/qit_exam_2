def brightness(lab):
    return lab[0]


import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if not os.path.exists("images5"):
    os.makedirs("images5")

img = cv2.imread("Bild.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# save gray and rgb images as files
cv2.imwrite("images5/image.jpg", img)
cv2.imwrite("images5/image_gray.jpg", img_gray)
cv2.imwrite("images5/image_rgb.jpg", img_rgb)

height, width = img_rgb.shape[:2]

# blur image to reduce noise for circle detection
blur1 = cv2.GaussianBlur(img_gray, (9, 9), 2)
cv2.imwrite("images5/image_blur.jpg", blur1)

# circle detection
circle = cv2.HoughCircles(
    blur1,
    cv2.HOUGH_GRADIENT,
    1,
    img.shape[0] // 2,
    param1=50,
    param2=30,
    minRadius=0,
    maxRadius=0,
)

# mask for disc
mask = np.zeros_like(img_gray)
if circle is not None:
    circle = np.uint16(np.around(circle))
    cx, cy, r = circle[0][0][0], circle[0][0][1], circle[0][0][2]
    cv2.circle(mask, (cx, cy), r - 5, 255, -1)

mask_circle = mask > 0

# apply mask to image
img_masked = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
cv2.imwrite("images5/image_masked.jpg", img_masked)

# brightness and contrast
alpha = 1.75
beta = -50
img_adjusted = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)
cv2.imwrite("images5/image_adjusted.jpg", img_adjusted)

# convert to LAB for clustering
lab = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2LAB)
pixels = lab.reshape(-1, 3)
mask_flat = mask_circle.flatten()
pixels_masked = pixels[mask_flat]

# kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixels_masked)

# labels back into full image
labels_full = np.full(mask_flat.shape, -1)
labels_full[mask_flat] = labels
labels_img = labels_full.reshape(height, width)
centers = kmeans.cluster_centers_

brightness_vals = np.array([brightness(c) for c in centers])
sorted_idx = np.argsort(brightness_vals)

black_cluster = sorted_idx[0]
white_cluster = sorted_idx[-1]
target_clusters = sorted_idx[1:-1]

# ============================================================
# Detect black contour as a dark FILLED structure
# ============================================================

# grayscale only inside circle
gray_inside = img_gray[mask_circle]

# Otsu threshold on disc pixels only
otsu_thresh, _ = cv2.threshold(
    gray_inside.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# black contour = pixels darker than threshold inside circle
black_mask = (img_gray < otsu_thresh) & mask_circle

# morphology to clean contour and make it solid
kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)

black_mask_u8 = black_mask.astype(np.uint8) * 255

# remove speckles
black_mask_u8 = cv2.morphologyEx(black_mask_u8, cv2.MORPH_OPEN, kernel3)

# close gaps in the line
black_mask_u8 = cv2.morphologyEx(black_mask_u8, cv2.MORPH_CLOSE, kernel5)

# thicken line to better match target image
# black_mask_u8 = cv2.dilate(black_mask_u8, kernel3, iterations=2)

# black_mask = black_mask_u8 > 0
# cv2.imwrite("images/black_mask.jpg", black_mask_u8)

# thicken line to better match target image
black_mask_u8 = cv2.dilate(black_mask_u8, kernel3, iterations=2)

# ------------------------------------------------------------
# NEW: fill small missed gaps / holes near the black contour
# ------------------------------------------------------------

# region around black line where small missed pixels may exist
black_neighborhood = cv2.dilate(black_mask_u8, kernel5, iterations=1) > 0

# candidate gaps = inside black neighborhood but not currently black
gap_candidates = black_neighborhood & ~(black_mask_u8 > 0) & mask_circle

# connected components on candidate gaps
num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
    gap_candidates.astype(np.uint8), connectivity=8
)

# fill only small components (tune area threshold if needed)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 300:
        black_mask_u8[labels_cc == i] = 255

# optional final close to smooth repaired contour
black_mask_u8 = cv2.morphologyEx(black_mask_u8, cv2.MORPH_CLOSE, kernel3)

# black_mask = black_mask_u8 > 0
black_mask = (img_gray < otsu_thresh) & mask_circle
cv2.imwrite("images5/black_mask.jpg", black_mask_u8)

# ============================================================
# Refine purple / white regions so purple does not hug black
# ============================================================

# purple candidate region from kmeans
target_mask = np.zeros((height, width), dtype=bool)
for c in target_clusters:
    target_mask |= (labels_img == c) & mask_circle

# stronger keep-out region around black contour
black_keepout = cv2.dilate(black_mask_u8, kernel5, iterations=2) > 0

# prevent purple from touching black lines
target_mask &= ~black_keepout

# define white as everything left inside the circle
white_mask = mask_circle & (~black_mask) & (~target_mask)

# det separate areas
total_pixels = np.sum(mask_circle)
white_pixels = np.sum(white_mask)
black_pixels = np.sum(black_mask)
edge_pixels = np.sum(target_mask)

# prints and plots
white_pct = 100 * white_pixels / total_pixels
black_pct = 100 * black_pixels / total_pixels
edge_pct = 100 * edge_pixels / total_pixels

print("\nResults:")
print(f"White pixels: {white_pct:.2f}%")
print(f"Black pixels: {black_pct:.2f}%")
print(f"Edge pixels: {edge_pct:.2f}%\n")

# ============================================================
# Visualization
# ============================================================

vis = np.zeros((height, width, 3), dtype=np.uint8)

# draw purple first
vis[target_mask] = [138, 73, 138]

# then white fills everything else inside the disc
vis[white_mask] = [255, 255, 255]

# black contour on top
vis[black_mask] = [0, 0, 0]

# keep only circle
vis_circle = cv2.bitwise_and(vis, vis, mask=mask.astype(np.uint8))

cv2.imwrite("images5/image_highlighted.jpg", vis_circle)
