import cv2
import os
import matplotlib.pyplot as plt

def load_image(path):
    """Load an image from file."""
    return cv2.imread(path)

def save_image(path, image):
    """Save an image to file."""
    cv2.imwrite(path, image)

def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on the image.
    bboxes: list of (x1, y1, x2, y2)
    """
    img = image.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def show_image(image, title="Image"):
    """Display an image using matplotlib."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show() 