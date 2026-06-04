# extracts brightness from color space and returns lightness (0-255)
def brightness(lab):
    return lab[0]


import os
import cv2
import numpy as np
from sklearn.cluster import KMeans  # clustering algorithm for pixel segmentation
import matplotlib.pyplot as plt

if not os.path.exists("images4"):
    os.makedirs("images4")


img = cv2.imread("Bild.jpg")  # BGR format
img_gray = cv2.cvtColor(
    img, cv2.COLOR_BGR2GRAY
)  # grayscale for circle detection and intensity-based analysis
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB for visualizations

cv2.imwrite("images4/1_image.jpg", img)  # BGR image
cv2.imwrite("images4/2_image_gray.jpg", img_gray)  # grayscale version
cv2.imwrite("images4/3_image_rgb.jpg", img_rgb)  # RGB version

# extract image dimensions, ignore color channels
height, width = img_rgb.shape[:2]


# Gaussian blur to reduce noise before circle detection (kernel size for more blur, standard deviation controls blur intensity)
blur1 = cv2.GaussianBlur(img_gray, (9, 9), 2)
cv2.imwrite("images4/4_image_blur.jpg", blur1)


# detect circular edges
circle = cv2.HoughCircles(
    blur1,
    cv2.HOUGH_GRADIENT,
    1,
    img.shape[0] // 2,  # minimum distance between circle centers
    param1=50,  # Canny edge detection upper threshold
    param2=30,  # circle accumulator threshold
    minRadius=0,
    maxRadius=0,  # search all radius sizes
)

mask = np.zeros_like(img_gray)
circle = np.uint16(np.around(circle))  #  float to integer for cv2.circle
cx, cy, r = (
    circle[0][0][0],
    circle[0][0][1],
    circle[0][0][2],
)  # extract center (cx, cy) and radius (r) from detection result

cv2.circle(mask, (cx, cy), r, 255, -1)  # white fill mask

mask_circle = mask > 0  # binary to boolean for masking

# apply mask to isolate the disc region
img_masked = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
cv2.imwrite("images4/5_image_masked.jpg", img_masked)

img_smoothed = cv2.bilateralFilter(img_masked, d=15, sigmaColor=80, sigmaSpace=80)
cv2.imwrite("images4/5_image_smoothed.jpg", img_smoothed)

# adjust image parameters
alpha = 1.8  # contrast multiplier
beta = -50  # brightness offset => darker
img_adjusted = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)
img_adjusted = cv2.bitwise_and(img_adjusted, img_adjusted, mask=mask.astype(np.uint8))
cv2.imwrite("images4/6_image_adjusted.jpg", img_adjusted)


# convert BGR image to LAB color space for clustering
lab = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2LAB)

# apply mask for lines
black_mask = cv2.imread("manual_mask.jpg", cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(black_mask, 127, 255, cv2.THRESH_BINARY)
black_mask = binary_mask > 0
cv2.imwrite("images4/7_image_mask.jpg", (black_mask.astype(np.uint8) * 255))

# apply center mask nope

# 2D image to 1D array for clustering
pixels = lab.reshape(-1, 3)
mask_flat = mask_circle.flatten()
pixels_masked = pixels[mask_flat]

# try kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=30)
labels = kmeans.fit_predict(pixels_masked)

# map cluster labels back to full image dimensions
labels_full = np.full(mask_flat.shape, -1)
# sssign predicted labels only to pixels inside the disc
labels_full[mask_flat] = labels
# back to 2D image format
labels_img = labels_full.reshape(height, width)
# center (representative color) of each cluster in LAB space
centers = kmeans.cluster_centers_

# cluster labels by brightness
brightness_vals = np.array([brightness(c) for c in centers])
sorted_idx = np.argsort(brightness_vals)
black_cluster = sorted_idx[0]
white_cluster = sorted_idx[-1]
target_clusters = sorted_idx[1:-1]


#####


# kernel for margin operations
kernel3 = np.ones((3, 3), np.uint8)

# refined masks for surface regions
# White (shiny) mask: pixels assigned to white_cluster AND within disc AND NOT in black grooves
white_mask = (labels_img == white_cluster) & mask_circle & (~black_mask)

# Target (unconditioned) mask: combine middle-brightness clusters
target_mask = np.zeros((height, width), dtype=bool)
for c in target_clusters:
    target_mask |= (labels_img == c) & mask_circle

# Remove black groove pixels from both regions
target_mask &= ~black_mask
white_mask &= ~black_mask

# dilated margin around black grooves
black_mask_u8 = black_mask.astype(np.uint8) * 255
black_margin = cv2.dilate(black_mask_u8, kernel3, iterations=2) > 0

# no margin from both regions
target_mask &= ~black_margin
white_mask &= ~black_margin

#  area measurements
black_pixels = np.sum(black_mask)
white_pixels = np.sum(white_mask)
edge_pixels = np.sum(target_mask)

# Calculate percentages using only active pixels (shiny + unconditioned)
# This ensures they sum to exactly 100%
active_pixels = white_pixels + edge_pixels

if active_pixels > 0:
    white_pct = 100 * white_pixels / active_pixels
    edge_pct = 100 * edge_pixels / active_pixels
else:
    white_pct = 0
    edge_pct = 0

# print("\nResults:")
# print(f"Shiny area: {white_pct:.2f}%")  # Shiny/conditioned percentage
# print(f"Unconditioned area: {edge_pct:.2f}%\n")  # Unconditioned/rough percentage
# print(f"Sum: {white_pct + edge_pct:.2f}%\n")  # Should equal 100%

total_disc_area = np.sum(mask_circle)
abs_shiny_pct = 100 * white_pixels / total_disc_area
abs_edge_pct = 100 * edge_pixels / total_disc_area
abs_black_pct = 100 * black_pixels / total_disc_area

# print("\nResults:")
# print(f"Shiny area: {abs_shiny_pct:.2f}%")
print(f"Unconditioned area: {abs_edge_pct:.2f}%")
# print(f"Black lines area: {abs_black_pct:.2f}%\n")

vis = np.zeros((height, width, 3), dtype=np.uint8)
vis[white_mask] = [255, 255, 255]  # White
vis[target_mask] = [138, 73, 138]  # Purple
vis[black_mask] = [0, 0, 0]  # Black

vis_circle = cv2.bitwise_and(vis, vis, mask=mask.astype(np.uint8))
cv2.imwrite("images4/8_image_highlighted.jpg", vis_circle)
