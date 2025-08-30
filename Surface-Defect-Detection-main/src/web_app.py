import streamlit as st
import os
import tempfile
from .defect_detection import detect_defects
from .utils import load_image, show_image

st.title('Surface Defect Detection on Steel Strips')

method = st.selectbox('Detection Method', ['yolo', 'classical'])
model_path = st.text_input('YOLO Weights Path', 'models/yolov5_weights.pt')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    output_dir = tempfile.mkdtemp()
    detect_defects(tmp_path, method, output_dir, model_path)
    out_img_path = os.path.join(output_dir, os.path.basename(tmp_path))
    result_img = load_image(out_img_path)
    st.subheader('Detection Result')
    st.image(result_img, channels='BGR', caption='Detected Defects') 