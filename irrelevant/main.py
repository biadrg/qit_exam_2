import os

if not os.path.exists("results"):
    os.makedirs("results")

# os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"
# os.environ["QT_DEBUG_PLUGINS"] = "0"

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

img = cv2.imread("Bild.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# save gray and rgb images as files
cv2.imwrite("results/image.jpg", img)
cv2.imwrite("results/image_gray.jpg", img_gray)
cv2.imwrite("results/image_rgb.jpg", img_rgb)

print("\nCheck points:")
print("images converted")
# print(img.shape[0])

# check if (checked) circle needed
blur1 = cv2.GaussianBlur(img_gray, (9, 9), 2)  # TO-DO: try different blurs
circle = cv2.HoughCircles(
    blur1,
    cv2.HOUGH_GRADIENT,
    1,
    img.shape[0] // 2,
    param1=50,
    param2=30,
    minRadius=0,
    maxRadius=0,  # TO-DO: maybe optimise
)

print("circle detected")

# and therefore mask needed to, results still low though
mask = np.zeros_like(img_gray)
if circle is not None:
    circle = np.uint16(np.around(circle))
    cx, cy, r = circle[0][0][0], circle[0][0][1], circle[0][0][2]
    cv2.circle(mask, (cx, cy), r - 5, 255, -1)

print("mask created")

# cv2.imshow("Masked Image", cv2.bitwise_and(img_rgb, img_rgb, mask=mask))
# cv2.waitKey(0)

# save masked image as file
img_masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
cv2.imwrite("results/image_masked.jpg", img_masked)

alpha = 1.75
beta = -50

img_adjusted = cv2.convertScaleAbs(img_masked, alpha=alpha, beta=beta)

cv2.imwrite("results/image_adjusted.jpg", img_adjusted)
# cv2.imshow("Adjusted Image", img_adjusted)
cv2.waitKey(0)

print("image adjusted")

# mask is okay, must add second mask
mask2 = np.zeros_like(img_gray)

disc_pixels = img_gray[mask == 255].reshape(-1, 1)
model = KMeans(n_clusters=3, random_state=42)

print("model created")

labels = model.fit_predict(disc_pixels)
labels_img = labels.reshape(img_rgb.shape[:2])
# print(labels.shape[0])
# print(np.size(labels))
centers = model.cluster_centers_
# brightness_threshold = np.mean(centers)  # axis?


def brightness(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


brightness_vals = np.array([brightness(c) for c in centers])

sorted_idx = np.argsort(brightness_vals)

polished = np.argmax(brightness_threshold)
unpolished = np.argmin(brightness_threshold)


polished_points = np.sum(labels == polished) / centers[polished][0]
unpolished_points = np.sum(labels == unpolished) / centers[unpolished][0]

# print(centers[polished][0])


# labels_final = np.zeros_like(img_gray)
# labels_final[mask == 255] = labels

# img_final = np.zeros_like(img_rgb)
# img_final[labels_final == 0] = [212, 212, 212]
# img_final[labels_final == 1] = [110, 110, 110]

print("model fitted")

total = len(labels)
type0 = np.sum(labels == 0)
type1 = np.sum(labels == 1)

print("\nResults:")
print(f"Total pixels: {total}")
print(f"Polished pixels: {polished_points} ({polished_points/total:.2%})")
print(f"Unpolished pixels: {unpolished_points} ({unpolished_points/total:.2%})\n")

img_final = np.zeros_like(img_rgb)
labels_final = np.zeros_like(img_gray)
labels_final[mask == 255] = labels

img_final[labels_final == polished] = [255, 255, 255]
img_final[labels_final == unpolished] = [235, 233, 132]

# save final image
cv2.imwrite("results/image_final.jpg", cv2.bitwise_and(img_final, img_final, mask=mask))
