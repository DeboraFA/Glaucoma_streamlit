
import streamlit as st
from PIL import Image
import numpy as np
import cv2

from brisque import BRISQUE

def variance_of_laplacian(path):
    image = Image.open(path)
    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fm =  cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


st.set_page_config(page_title="Avaliar qualidade da imagem",
                   layout='wide',
                   page_icon='./images/object.png')

st.header('Avaliar qualidade das imagens individualmente')
st.write('Please Upload Image to get detections')


# Allow the user to upload multiple image files
uploaded_files = st.file_uploader("Choose images to display", accept_multiple_files=True)

# If the user has uploaded any files
if uploaded_files:
    # Create a list to hold the uploaded images
    images = []
    button = st.button('Avaliar')

    idx = 0 
    for _ in range(len(uploaded_files)): 
        cols = st.columns(4) 
        if button:
            if idx < len(uploaded_files): 
                obj = BRISQUE(uploaded_files[idx], url=False)
                laplace = variance_of_laplacian(uploaded_files[idx])
                cols[0].write(np.round(laplace,2))
                cols[0].write(np.round(obj.score(),2))
                cols[0].image(uploaded_files[idx], width=180, caption = uploaded_files[idx].name)
            idx+=1
            
            if idx < len(uploaded_files):
                obj = BRISQUE(uploaded_files[idx], url=False)
                laplace = variance_of_laplacian(uploaded_files[idx])
                cols[1].write(np.round(laplace,2))
                cols[1].write(np.round(obj.score(),2))
                cols[1].image(uploaded_files[idx], width=180, caption = uploaded_files[idx].name)
            idx+=1

            if idx < len(uploaded_files):
                obj = BRISQUE(uploaded_files[idx], url=False)
                laplace = variance_of_laplacian(uploaded_files[idx])
                cols[2].write(np.round(laplace,2))
                cols[2].write(np.round(obj.score(),2))
                cols[2].image(uploaded_files[idx], width=180, caption = uploaded_files[idx].name)
            idx+=1 
            if idx < len(uploaded_files): 
                obj = BRISQUE(uploaded_files[idx], url=False)
                laplace = variance_of_laplacian(uploaded_files[idx])
                cols[3].write(np.round(laplace,2))
                cols[3].write(np.round(obj.score(),2))
                cols[3].image(uploaded_files[idx], width=180, caption = uploaded_files[idx].name)
                idx = idx + 1
            else:
                break