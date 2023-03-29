

import streamlit as st
from glaucoma_segmentation import test_segmentation_disc, test_segmentation_cup
from PIL import Image
import numpy as np
import cv2
from itertools import cycle


st.set_page_config(page_title="Segmentação Disco Ótico e Escavação",
                   layout='wide',
                   page_icon='./images/object.png')

st.header('Segmentação Disco Ótico e Escavação')
st.write('Please Upload Image to get detections')

# # test evaluate
# timm-mobilenetv3_large_075
seg_encoder = 'resnet50' # https://github.com/qubvel/segmentation_models.pytorch
seg_model = 'FPN' #  Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3,DeepLabV3Plus

# Allow the user to upload multiple image files
uploaded_files = st.file_uploader("Choose images to display", accept_multiple_files=True)

# If the user has uploaded any files
if uploaded_files:
    # Create a list to hold the uploaded images
    images = []
    button = st.button('Segmentation optic disc and cup')
    # Iterate over the uploaded files
    cols = cycle(st.columns(3)) 
    for idx, img in enumerate(uploaded_files):
        
        if button:
            image_test = Image.open(img)
            image = np.array(image_test)
            contours_cup = test_segmentation_cup(image_test, seg_encoder, seg_model)
            contours_disc = test_segmentation_disc(image_test, seg_encoder, seg_model)

            result = image.copy()
            result = cv2.resize(result, (224,224))
            for cnt in contours_cup:
                img_contours1 = cv2.drawContours(result, cnt, -1, (0, 255, 0), 2)

            for cnt2 in contours_disc:
                img_contours = cv2.drawContours(img_contours1, cnt2, -1, (255, 0, 0), 2)
            next(cols).image(img_contours, width=240, caption=img.name)
        



