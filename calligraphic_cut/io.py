import numpy as np
from PIL import Image


def load(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0


def save(path, img):
    Image.fromarray(np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)).save(path)
