import cv2
import numpy as np

def to_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, ksize=(5, 5), sigma=0):
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(image, ksize, sigma)

def enhance_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Enhance image contrast using CLAHE (adaptive histogram equalization)."""
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    return clahe.apply(image)