import os
import cv2
from .preprocess import to_grayscale, apply_gaussian_blur, enhance_contrast
from .utils import load_image, save_image, draw_bboxes
from .inference import YOLODetector
import numpy as np

def detect_defects(input_path, method, output_dir, model_path=None, conf=0.25, iou=0.45, min_area=100):
    """
    Main function to detect defects on images or videos.
    method: 'classical' or 'advanced_classical'
    """
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        raise ValueError('Input path not found.')

    if method == 'classical':
        for img_path in files:
            img = load_image(img_path)
            gray = to_grayscale(img)
            blur = apply_gaussian_blur(gray)
            contrast = enhance_contrast(blur)
            # Simple thresholding and contour detection
            _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > min_area]  # filter small areas
            bboxes = [(x, y, x + w, y + h) for (x, y, w, h) in bboxes]
            img_out = draw_bboxes(img, bboxes, color=(0, 0, 255))
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            save_image(out_path, img_out)
    elif method == 'advanced_classical':
        for img_path in files:
            img = load_image(img_path)
            gray = to_grayscale(img)
            blur = apply_gaussian_blur(gray)
            contrast = enhance_contrast(blur)
            # Canny edge detection and morphological closing
            edges = cv2.Canny(contrast, 100, 200)
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            # Contour detection
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > min_area]  # filter small areas
            bboxes = [(x, y, x + w, y + h) for (x, y, w, h) in bboxes]
            img_out = draw_bboxes(img, bboxes, color=(0, 0, 255))
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            save_image(out_path, img_out)
    elif method == 'yolo':
        detector = YOLODetector(weights_path=model_path if model_path else None, model_type='yolov5', conf_thres=conf, iou_thres=iou)
        for img_path in files:
            img = load_image(img_path)
            bboxes, confs, classes = detector.detect(img)
            img_out = draw_bboxes(img, bboxes, color=(0, 255, 0))
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            save_image(out_path, img_out)
    else:
        raise ValueError('Unknown method: choose "classical", "advanced_classical", or "yolo"')