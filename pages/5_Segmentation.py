

import streamlit as st
from glaucoma_segmentation import test_segmentation_disc, test_segmentation_cup, CDR, BVR, NRR
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
seg_encoder = 'mobilenet_v2' # https://github.com/qubvel/segmentation_models.pytorch
seg_model = 'Unetplusplus' #  Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3,DeepLabV3Plus

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
            for cnt_disc in contours_disc:
                img_contours1 = cv2.drawContours(result, [cnt_disc], -1, (0, 255, 0), 2)

            for cnt_cup in contours_cup:
                img_contours = cv2.drawContours(img_contours1, [cnt_cup], -1, (255, 0, 0), 2)

            cdr = CDR(cnt_disc, cnt_cup)
            nrr = NRR(cnt_disc, cnt_cup, image)
            bvr = BVR(cnt_disc, image)

            next(cols).image(img_contours, width=240, caption=img.name)
        



