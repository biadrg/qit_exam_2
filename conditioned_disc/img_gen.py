import numpy as np
import cv2
from PIL import Image


# ---------- LOAD ----------
def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def save_image(arr, path):
    Image.fromarray(arr).save(path)


# ---------- STEP 1: DISC MASK ----------
def get_disc_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # threshold: background is dark
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # clean it
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15)))

    return (mask > 0).astype(np.uint8)


# ---------- STEP 2: RANDOM DEFECT FIELD ----------
def generate_defect_field(shape, smoothness):
    h, w = shape[:2]

    noise = np.random.rand(h, w)

    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=smoothness, sigmaY=smoothness)

    noise = (noise - noise.min()) / (noise.max() - noise.min())

    return noise


# ---------- STEP 3: CONTROL COVERAGE ----------
def threshold_by_coverage(field, disc_mask, coverage):
    # only consider inside disc
    values = field[disc_mask == 1]

    thresh = np.quantile(values, 1 - coverage)

    mask = (field > thresh).astype(np.uint8)

    return mask


# ---------- STEP 4: APPLY DISC CONSTRAINT ----------
def restrict_to_disc(mask, disc_mask):
    return mask * disc_mask


# ---------- STEP 5: APPLY DEFECT TEXTURE ----------
def apply_texture(image, mask, intensity=50):
    img = image.astype(np.float32)

    noise = np.random.normal(0, intensity, img.shape)

    mask3 = np.stack([mask] * 3, axis=-1)

    img = img + noise * mask3

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- MAIN ----------
image = load_image("Bild.jpg")

# ✅ get disc region
disc_mask = get_disc_mask(image)

# ✅ control parameters
coverage = 0.3  # % defect
smoothness = 12  # shape size (lower = small blobs, higher = big blobs)
intensity = 40  # visual strength

# ✅ generate field
field = generate_defect_field(image.shape, smoothness)

# ✅ threshold with coverage constraint
mask = threshold_by_coverage(field, disc_mask, coverage)

# ✅ enforce disc restriction
mask = restrict_to_disc(mask, disc_mask)

# ✅ apply defect
result = apply_texture(image, mask, intensity)

# ✅ save outputs
save_image(result, "synthetic.png")
save_image(mask * 255, "mask.png")
save_image(disc_mask * 255, "disc_mask.png")
