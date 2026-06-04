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

# adjust image parameters
alpha = 2.0  # contrast multiplier
beta = -100  # brightness offset => darker
img_adjusted = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)
img_adjusted = cv2.bitwise_and(img_adjusted, img_adjusted, mask=mask.astype(np.uint8))
cv2.imwrite("images4/6_image_adjusted.jpg", img_adjusted)

# convert BGR image to LAB color space for clustering
lab = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2LAB)

# apply mask for lines
black_mask = cv2.imread("set_mask.jpg", cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(black_mask, 127, 255, cv2.THRESH_BINARY)
black_mask_bool = binary_mask > 0
cv2.imwrite("images4/7_image_mask.jpg", (black_mask_bool.astype(np.uint8) * 255))

# apply center mask
center_mask = cv2.imread("center_mask_optimised.jpg", cv2.IMREAD_GRAYSCALE)
_, center_binary = cv2.threshold(center_mask, 127, 255, cv2.THRESH_BINARY)
center_bool = center_binary > 0

# kernel for margin operations
kernel3 = np.ones((3, 3), np.uint8)

#####
# black_margin = (
#     cv2.dilate(black_mask_bool.astype(np.uint8) * 255, kernel3, iterations=2) > 0
# )
# lines and centre into one exclusion zone
exclusion_zone = black_mask_bool | center_bool
surface_mask = mask_circle & (~exclusion_zone)
#####

# 2D image to 1D array for clustering
pixels = lab.reshape(-1, 3)
## mask_flat = mask_circle.flatten()
surface_flat = surface_mask.flatten()
# pixels_masked = pixels[mask_flat]
pixels_masked = pixels[surface_flat]

pixels_to_cluster = img_adjusted[surface_mask].reshape(-1, 1)

# try kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=30)
labels = kmeans.fit_predict(pixels_masked)

# map cluster labels back to full image dimensions
###### labels_full = np.full(mask_flat.shape, -1)
labels_full = np.full(surface_flat.shape, -1)
# sssign predicted labels only to pixels inside the disc
##### labels_full[mask_flat] = labels
labels_full[surface_flat] = labels
# back to 2D image format
labels_img = labels_full.reshape(height, width)
# center (representative color) of each cluster in LAB space
centers = kmeans.cluster_centers_

# cluster labels by brightness
brightness_vals = np.array([brightness(c) for c in centers])
sorted_idx = np.argsort(brightness_vals)
###### black_cluster = sorted_idx[0]
white_cluster = sorted_idx[-1]
target_clusters = sorted_idx[1:-1]


# refined masks for surface regions
# White (shiny) mask: pixels assigned to white_cluster AND within disc AND NOT in black grooves
###### white_mask = (labels_img == white_cluster) & mask_circle & (~black_mask)
white_mask = (labels_img == white_cluster) & surface_mask

# Target (unconditioned) mask: combine middle-brightness clusters
target_mask = np.zeros((height, width), dtype=bool)
for c in target_clusters:
    ###### target_mask |= (labels_img == c) & mask_circle
    target_mask |= (labels_img == c) & surface_mask

#####
# # Remove black groove pixels from both regions
# target_mask &= ~black_mask
# white_mask &= ~black_mask

# # dilated margin around black grooves
# black_mask_u8 = black_mask.astype(np.uint8) * 255
# black_margin = cv2.dilate(black_mask_u8, kernel3, iterations=2) > 0

# # no margin from both regions
# target_mask &= ~black_margin
# white_mask &= ~black_margin
#####

#  area measurements
# black_pixels = np.sum(black_mask)
total_pixels = np.sum(surface_mask)
# total_pixels -= black_pixels
white_pixels = np.sum(white_mask)
edge_pixels = np.sum(target_mask)

white_pct = 100 * white_pixels / total_pixels if total_pixels > 0 else 0
edge_pct = 100 * edge_pixels / total_pixels if total_pixels > 0 else 0

print("\nResults:")
print(f"White pixels: {white_pct:.2f}%")  # Shiny/conditioned percentage
print(f"Edge pixels: {edge_pct:.2f}%\n")  # Unconditioned/rough percentage

vis = np.zeros((height, width, 3), dtype=np.uint8)
vis[white_mask] = [255, 255, 255]  # White
vis[target_mask] = [138, 73, 138]  # Purple
# vis[black_mask] = [0, 0, 0]  # Black
vis[center_bool] = [255, 255, 255]  # Centre forced to white
vis[black_mask_bool] = [0, 0, 0]  # Grooves forced to black

vis_circle = cv2.bitwise_and(vis, vis, mask=mask.astype(np.uint8))
cv2.imwrite("images4/8_image_highlighted.jpg", vis_circle)
