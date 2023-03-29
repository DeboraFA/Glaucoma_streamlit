import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np
import os
from itertools import cycle
st.set_page_config(page_title="YOLO Object Detection",
                   layout='wide',
                   page_icon='./images/object.png')

st.header('Get Object Detection for any Image')
st.write('Please Upload Image to get detections')

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')
    #st.balloons()

# Allow the user to upload multiple image files
uploaded_files = st.file_uploader("Choose images to display", accept_multiple_files=True)

# If the user has uploaded any files
if uploaded_files:
    # Create a list to hold the uploaded images
    images = []
    button = st.button('Get Detection from YOLO')
    # Iterate over the uploaded files

    cols = cycle(st.columns(3)) # st.columns here since it is out of beta at the time I'm writing this
    for idx, img in enumerate(uploaded_files):
        # obj to array
        if button:
            image = Image.open(img)
            image_array = np.array(image)
            pred_img = yolo.predictions(image_array)
            pred_img_obj = Image.fromarray(pred_img)
            next(cols).image(pred_img_obj, width=240, caption=img.name)
        

    