def brightness(lab):
    return lab[0]


import os

if not os.path.exists("images2"):
    os.makedirs("images2")

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

img = cv2.imread("Bild.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# save gray and rgb images as files
cv2.imwrite("images2/image.jpg", img)
cv2.imwrite("images2/image_gray.jpg", img_gray)
cv2.imwrite("images2/image_rgb.jpg", img_rgb)

height, width = img_rgb.shape[:2]

# print("\nCheck points:")
# print("images converted")

# blur image to reduce noise for circle detection, TO-DO: maybe try different blurs and parameters
blur1 = cv2.GaussianBlur(img_gray, (9, 9), 2)
cv2.imwrite("images2/image_blur.jpg", blur1)

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
# print("circle detected")

# and therefore mask needed to, results still low though
mask = np.zeros_like(img_gray)
if circle is not None:
    circle = np.uint16(np.around(circle))
    cx, cy, r = circle[0][0][0], circle[0][0][1], circle[0][0][2]
    cv2.circle(mask, (cx, cy), r - 5, 255, -1)
mask_circle = mask > 0  # to boolean

# apply mask to image
img_masked = cv2.bitwise_and(img, img, mask=mask_circle.astype(np.uint8) * 255)
cv2.imwrite("images2/image_masked.jpg", img_masked)

# brightness and contrast
alpha = 1.75
beta = -50
img_adjusted = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)
cv2.imwrite("images2/image_adjusted.jpg", img_adjusted)

# preprep
lab = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2LAB)  # for kmeans
pixels = lab.reshape(-1, 3)  # flatten img to 2d
mask_flat = mask_circle.flatten()  # flatten mask to 1d
pixels_masked = pixels[mask_flat]  # get only inside circle pixels

# kmeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # try without n_init
labels = kmeans.fit_predict(pixels_masked)

# labels back into full image
labels_full = np.full(
    mask_flat.shape, -1
)  #  bg pixels get -1 2 distinguish them from clusters
labels_full[mask_flat] = labels  # reconstruct img with bg pixels 2
labels_img = labels_full.reshape(height, width)  # 1d back to 2d
centers = kmeans.cluster_centers_  # get cluser centers

brightness_vals = np.array(
    [brightness(c) for c in centers]
)  # get brightness lvl of each cluster center
sorted_idx = np.argsort(brightness_vals)

black_cluster = sorted_idx[0]
white_cluster = sorted_idx[-1]
target_clusters = sorted_idx[1:-1]

# remove black lines
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# det separate areas
total_pixels = np.sum(mask_circle)
white_pixels = np.sum((labels_img == white_cluster) & mask_circle)
black_pixels = np.sum((labels_img == black_cluster) & (edges > 0) & mask_circle)
edge_pixels = 0
for c in target_clusters:
    edge_pixels += np.sum((labels_img == c) & mask_circle)

# prints n plots
white_pct = 100 * white_pixels / total_pixels
black_pct = 100 * black_pixels / total_pixels
edge_pct = 100 * edge_pixels / total_pixels

print("\nResults:")
print(f"White pixels: {white_pct:.2f}%")
print(f"Black pixels: {black_pct:.2f}%")
print(f"Edge pixels: {edge_pct:.2f}%\n")

vis = np.zeros(
    (height, width, 3), dtype=np.uint8
)  # generate highlighted version of the image

vis[(labels_img == white_cluster) & mask_circle] = [
    255,
    255,
    255,
]  # highlight white as white

vis[(labels_img == black_cluster) & (edges > 0)] = [
    222,
    201,
    45,
]  # highlight black lines as black

for c in target_clusters:
    vis[(labels_img == c) & mask_circle] = [
        138,
        73,
        138,
    ]  # highlight edge patches as blue

cv2.imwrite(
    "images2/image_highlighted.jpg",
    cv2.bitwise_and(vis, vis, mask=mask_circle.astype(np.uint8) * 255),
)


# do not consider thingies within lines
# apply mask on them
# figure out black percentage
# increase cluster count

# idea from ana:
# generate image,
# get measurmeents of disc from og,
# get coordinates of black lines,
# have a predefined percentage of polished and unpolished,
# check if algortihm works on generated image
