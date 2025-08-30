import torch
import cv2
import numpy as np
import os

class YOLODetector:
    def __init__(self, weights_path=None, model_type='yolov5', conf_thres=0.25, iou_thres=0.45):
        """
        Initialize YOLO detector with pre-trained weights.
        If weights_path is missing, use the standard 'yolov5s' model from Ultralytics.
        model_type: 'yolov5' or 'yolov7' (yolov7 support is placeholder)
        conf_thres and iou_thres are applied if supported by the model implementation.
        """
        if model_type == 'yolov5':
            # Require a valid, local weights file; prevent falling back to generic COCO model
            if not weights_path or not os.path.exists(weights_path):
                raise FileNotFoundError('YOLO weights not found. Please provide a valid path to trained steel-defect weights.')
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
        elif model_type == 'yolov7':
            # Placeholder: yolov7 loading would require a different repo or ONNX
            raise NotImplementedError('YOLOv7 loading not implemented in this template.')
        else:
            raise ValueError('Unsupported model type')

        # Try to set thresholds if available
        try:
            if hasattr(self.model, 'conf'):
                self.model.conf = float(conf_thres)
            if hasattr(self.model, 'iou'):
                self.model.iou = float(iou_thres)
        except Exception:
            pass

    def detect(self, image):
        """
        Run inference on an image (numpy array, BGR).
        Returns: list of bounding boxes [(x1, y1, x2, y2)], confidences, and class names.
        """
        results = self.model(image)
        bboxes = []
        confidences = []
        class_names = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            bboxes.append((x1, y1, x2, y2))
            confidences.append(float(conf))
            class_names.append(self.model.names[int(cls)])
        return bboxes, confidences, class_names 