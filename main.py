import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load the original, clean image
img = cv2.imread("image_49b37f.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Find the contact disc boundary
blurred = cv2.medianBlur(gray, 11)
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=img.shape[0] // 2,
    param1=50,
    param2=30,
    minRadius=int(img.shape[0] * 0.35),
    maxRadius=int(img.shape[0] * 0.48),
)

# 3. Create the mask and filter pixels
mask = np.zeros_like(gray)
if circles is not None:
    circles = np.uint16(np.around(circles))
    cx, cy, r = circles[0][0][0], circles[0][0][1], circles[0][0][2]
    cv2.circle(mask, (cx, cy), r - 5, 255, -1)

disc_pixels = img_rgb[mask == 255]

# 4. K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(disc_pixels)

# 5. Identify the regions mathematically
centers = kmeans.cluster_centers_
brightness = np.mean(centers, axis=1)

# The bright centre is the lighter cluster; the patchy edges are the darker one
bright_type2_idx = np.argmax(brightness)
patchy_type1_idx = np.argmin(brightness)

type1_pct = (np.sum(labels == patchy_type1_idx) / len(labels)) * 100
type2_pct = (np.sum(labels == bright_type2_idx) / len(labels)) * 100

print(f"Type 1 (Patchy Edges): {type1_pct:.2f}%")
print(f"Type 2 (Bright Centre): {type2_pct:.2f}%")

# 6. Plotting the results
segmented_img = np.zeros_like(img_rgb)
full_labels = np.full(gray.shape, -1, dtype=int)
full_labels[mask == 255] = labels

# Paint Type 1 Green and Type 2 Blue
segmented_img[full_labels == patchy_type1_idx] = [100, 200, 100]  # Green
segmented_img[full_labels == bright_type2_idx] = [100, 150, 220]  # Blue

# Display the side-by-side comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Isolated Contact Disc")
plt.imshow(cv2.bitwise_and(img_rgb, img_rgb, mask=mask))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Type 1 (Green): {type1_pct:.1f}% | Type 2 (Blue): {type2_pct:.1f}%")
plt.imshow(cv2.bitwise_and(segmented_img, segmented_img, mask=mask))
plt.axis("off")

plt.tight_layout()
plt.show()
