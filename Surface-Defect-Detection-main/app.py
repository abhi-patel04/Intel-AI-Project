import streamlit as st
import os
import io
import tempfile
import zipfile
import numpy as np
from src.utils import load_image, draw_bboxes
from src.preprocess import to_grayscale, apply_gaussian_blur, enhance_contrast
from src.inference import YOLODetector
import cv2
import pandas as pd

st.title('Surface Defect Detection on Steel Strips')

# Sidebar controls
st.sidebar.header('Controls')
method = st.sidebar.selectbox('Detection Method', ['classical', 'advanced classical', 'yolo', 'compare methods'])
uploaded_files = st.file_uploader('Upload image(s)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Preprocessing sliders
st.sidebar.subheader('Preprocessing')
blur_ksize = st.sidebar.slider('Gaussian Blur Kernel (odd)', min_value=1, max_value=15, value=5, step=2)
blur_sigma = st.sidebar.slider('Gaussian Sigma', min_value=0, max_value=10, value=0, step=1)
clahe_clip = st.sidebar.slider('CLAHE Clip Limit', min_value=1.0, max_value=5.0, value=2.0, step=0.5)
clahe_tile = st.sidebar.slider('CLAHE Tile Size', min_value=4, max_value=16, value=8, step=2)

# Detection parameters
st.sidebar.subheader('Classical/Edge Detection')
canny_low = st.sidebar.slider('Canny Low Threshold', min_value=0, max_value=255, value=50, step=1)
canny_high = st.sidebar.slider('Canny High Threshold', min_value=0, max_value=255, value=150, step=1)
morph_kernel = st.sidebar.slider('Morphological Kernel Size', min_value=1, max_value=15, value=5, step=2)
morph_op = st.sidebar.selectbox('Morph Operation', ['close', 'open', 'dilate', 'erode'])
min_area = st.sidebar.slider('Min Contour Area', min_value=0, max_value=5000, value=100, step=50)

st.sidebar.subheader('YOLO')
yolo_weights = st.sidebar.text_input('YOLO Weights Path (optional)', 'models/yolov5_weights.pt')
yolo_conf = st.sidebar.slider('YOLO Confidence', min_value=0.05, max_value=0.95, value=0.25, step=0.05)
yolo_iou = st.sidebar.slider('YOLO NMS IoU', min_value=0.1, max_value=0.9, value=0.45, step=0.05)

def classical_detection(img):
    gray = to_grayscale(img)
    blur = apply_gaussian_blur(gray, ksize=(blur_ksize, blur_ksize), sigma=blur_sigma)
    contrast = enhance_contrast(blur, clip_limit=clahe_clip, tile_grid_size=(clahe_tile, clahe_tile))
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > min_area]
    bboxes = [(x, y, x + w, y + h) for (x, y, w, h) in bboxes]
    return bboxes, thresh

def advanced_classical_detection(img):
    gray = to_grayscale(img)
    blur = apply_gaussian_blur(gray, ksize=(blur_ksize, blur_ksize), sigma=blur_sigma)
    contrast = enhance_contrast(blur, clip_limit=clahe_clip, tile_grid_size=(clahe_tile, clahe_tile))
    # Robust edge/segmentation: adaptive threshold fallback if Canny yields no contours
    edges = cv2.Canny(contrast, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    op_map = {
        'close': cv2.MORPH_CLOSE,
        'open': cv2.MORPH_OPEN,
        'dilate': cv2.MORPH_DILATE,
        'erode': cv2.MORPH_ERODE,
    }
    morphed = cv2.morphologyEx(edges, op_map[morph_op], kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        adap = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        morphed = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > min_area]
    bboxes = [(x, y, x + w, y + h) for (x, y, w, h) in bboxes]
    return bboxes, morphed

@st.cache_resource(show_spinner=False)
def load_yolo(weights_path, conf, iou):
    return YOLODetector(weights_path=weights_path if weights_path else None, model_type='yolov5', conf_thres=conf, iou_thres=iou)

def yolo_detection(img):
    detector = load_yolo(yolo_weights if os.path.exists(yolo_weights) else None, yolo_conf, yolo_iou)
    bboxes, confs, classes = detector.detect(img)
    return bboxes, confs, classes

def bbox_metrics(bboxes):
    rows = []
    for (x1, y1, x2, y2) in bboxes:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        rows.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'width': w, 'height': h, 'area': area, 'cx': cx, 'cy': cy})
    return pd.DataFrame(rows)

def make_crops_zip(img, bboxes, filename_prefix='crop'):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
            crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            ok, buf = cv2.imencode('.jpg', crop)
            if ok:
                zf.writestr(f"{filename_prefix}_{idx}.jpg", buf.tobytes())
    mem.seek(0)
    return mem

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        img = load_image(tmp_path)

        st.subheader(f'File: {uploaded_file.name}')

        if method == 'classical':
            bboxes, aux = classical_detection(img)
            img_out = draw_bboxes(img, bboxes, color=(0, 0, 255))
            st.image(img_out, caption='Detected Defects (Classical)', channels='BGR')
            st.image(aux, caption='Threshold Map', channels='GRAY')
            st.write(f"**Number of detected defects:** {len(bboxes)}")
            if len(bboxes) == 0:
                st.warning('No defects detected. If this is unexpected, try another image.')
            if len(bboxes) > 0:
                df = bbox_metrics(bboxes)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button('Download Detections (CSV)', csv, f'{os.path.splitext(uploaded_file.name)[0]}_detections.csv', 'text/csv', key=f'csv_classical_{uploaded_file.name}')
                crops_zip = make_crops_zip(img, bboxes, filename_prefix='defect')
                st.download_button('Download Crops (ZIP)', crops_zip.getvalue(), f'{os.path.splitext(uploaded_file.name)[0]}_crops.zip', 'application/zip', key=f'zip_classical_{uploaded_file.name}')
            ok, buffer = cv2.imencode('.jpg', img_out)
            if ok:
                st.download_button('Download Result Image', buffer.tobytes(), f'{os.path.splitext(uploaded_file.name)[0]}_result.jpg', 'image/jpeg', key=f'img_classical_{uploaded_file.name}')

        elif method == 'advanced classical':
            bboxes, closed = advanced_classical_detection(img)
            img_out = draw_bboxes(img, bboxes, color=(255, 0, 0))
            st.image(img_out, caption='Detected Defects (Advanced Classical)', channels='BGR')
            st.image(closed, caption='Edge/Morph Map', channels='GRAY')
            st.write(f"**Number of detected defects:** {len(bboxes)}")
            if len(bboxes) == 0:
                st.warning('No defects detected. If this is unexpected, try another image.')
            if len(bboxes) > 0:
                df = bbox_metrics(bboxes)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button('Download Detections (CSV)', csv, f'{os.path.splitext(uploaded_file.name)[0]}_detections.csv', 'text/csv', key=f'csv_adv_{uploaded_file.name}')
                crops_zip = make_crops_zip(img, bboxes, filename_prefix='defect')
                st.download_button('Download Crops (ZIP)', crops_zip.getvalue(), f'{os.path.splitext(uploaded_file.name)[0]}_crops.zip', 'application/zip', key=f'zip_adv_{uploaded_file.name}')
            ok, buffer = cv2.imencode('.jpg', img_out)
            if ok:
                st.download_button('Download Result Image', buffer.tobytes(), f'{os.path.splitext(uploaded_file.name)[0]}_result.jpg', 'image/jpeg', key=f'img_adv_{uploaded_file.name}')

        elif method == 'yolo':
            try:
                bboxes, confs, classes = yolo_detection(img)
                img_out = draw_bboxes(img, bboxes, color=(0, 255, 0))
                st.image(img_out, caption='Detected Defects (YOLO)', channels='BGR')
                st.write(f"**Number of detected defects:** {len(bboxes)}")
                if len(bboxes) == 0:
                    st.warning('No defects detected. Ensure a valid steel-defect YOLO weights file is set and try lowering confidence/raising IoU.')
                if len(bboxes) > 0:
                    df = bbox_metrics(bboxes)
                    df['confidence'] = confs
                    df['class'] = classes
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download Detections (CSV)', csv, f'{os.path.splitext(uploaded_file.name)[0]}_detections_yolo.csv', 'text/csv', key=f'csv_yolo_{uploaded_file.name}')
                    # YOLO txt export
                    yolo_lines = []
                    h, w = img.shape[:2]
                    name_to_id = {name: idx for idx, name in enumerate(sorted(set(classes)))}
                    for (x1, y1, x2, y2), cls_name in zip(bboxes, classes):
                        xc = (x1 + x2) / 2.0 / w
                        yc = (y1 + y2) / 2.0 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        cls_id = name_to_id.get(cls_name, 0)
                        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    st.download_button('Download YOLO TXT', "\n".join(yolo_lines).encode('utf-8'), f'{os.path.splitext(uploaded_file.name)[0]}_detections.txt', 'text/plain', key=f'txt_yolo_{uploaded_file.name}')
                    crops_zip = make_crops_zip(img, bboxes, filename_prefix='defect')
                    st.download_button('Download Crops (ZIP)', crops_zip.getvalue(), f'{os.path.splitext(uploaded_file.name)[0]}_crops.zip', 'application/zip', key=f'zip_yolo_{uploaded_file.name}')
                ok, buffer = cv2.imencode('.jpg', img_out)
                if ok:
                    st.download_button('Download Result Image', buffer.tobytes(), f'{os.path.splitext(uploaded_file.name)[0]}_result.jpg', 'image/jpeg', key=f'img_yolo_{uploaded_file.name}')
            except Exception as e:
                st.error(f"YOLO inference error: {e}")
                st.info('Tip: Provide a local path to trained steel-defect weights (e.g., NEU-DET/Severstal) in the sidebar.')

        elif method == 'compare methods':
            col1, col2 = st.columns(2)
            with col1:
                b1, aux1 = classical_detection(img)
                out1 = draw_bboxes(img, b1, color=(0, 0, 255))
                st.image(out1, caption=f'Classical ({len(b1)} boxes)', channels='BGR')
            with col2:
                b2, aux2 = advanced_classical_detection(img)
                out2 = draw_bboxes(img, b2, color=(255, 0, 0))
                st.image(out2, caption=f'Advanced Classical ({len(b2)} boxes)', channels='BGR')
            # Optional YOLO preview
            if os.path.exists(yolo_weights):
                try:
                    b3, confs, classes = yolo_detection(img)
                    out3 = draw_bboxes(img, b3, color=(0, 255, 0))
                    st.image(out3, caption=f'YOLO ({len(b3)} boxes)', channels='BGR')
                except Exception as e:
                    st.warning(f"YOLO comparison skipped: {e}")
                    st.info('Tip: Set a valid trained steel-defect YOLO weights file in the sidebar.')