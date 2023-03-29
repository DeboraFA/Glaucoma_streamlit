
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from itertools import cycle
import sys


st.set_page_config(page_title="Aplicar pré-processamentos na imagem",
                   layout='wide',
                   page_icon='./images/object.png')

# st.header('Avaliar qualidade das imagens individualmente')
st.write('Please Upload Image to get detections')


# Allow the user to upload multiple image files
uploaded_files = st.file_uploader("Choose images to display")
caption = []
col1 , col2 = st.columns(2)


    
# If the user has uploaded any files
if uploaded_files:
    # Create a list to hold the uploaded images
    # with col1:
    #     button1 = st.button('Todos')
    with col1:
        button1 = st.button('Color Restoration')
    with col2:
        button2 = st.button('Image Enhancement')

    sys.path.insert(0,'./preprocessamento/ColorRestoration_Enhancement')
    if button1:
        filteredImages = []
        
        from DCP import DCP
        from GBdehazingRCorrection import GBdehazingRC
        from IBLA import IBLA
        from LowComplexityDCP import LowComplexityDCP
        # from MIP import MIP2
        import MIP
        from NewOpticalModel import NewOpticalModel
        from UDCP import UDCP2
        from Rows import Rows
        from ULAP import ULAP2


        dcp = DCP(uploaded_files)
        gbd = cv2.cvtColor(GBdehazingRC(uploaded_files),  cv2.COLOR_BGR2RGB)
        ibla = IBLA(uploaded_files)
        lc = LowComplexityDCP(uploaded_files)
        mip = MIP.MIP2(uploaded_files)
        no = NewOpticalModel(uploaded_files)
        udcp = UDCP2(uploaded_files)
        rows = Rows(uploaded_files)
        ulap = ULAP2(uploaded_files)

        # st.image(uploaded_files, width=180, caption = 'Original')
        # st.image(dcp, width=180, caption = 'DCP')
        # st.image(gbd, width=180, caption = 'GBdehazingRCorrection')
        # st.image(ibla, width=180, caption = 'IBLA')
        # st.image(lc, width=180, caption = 'LowComplexityDCP')
        # st.image(mip, width=180, caption = 'MIP')
        # st.image(no, width=180, caption = 'NewOpticalModel')
        # st.image(udcp, width=180, caption = 'UDCP')
        # st.image(rows, width=180, caption = 'Rows')
        # st.image(ulap, width=180, caption = 'ULAP')



        filteredImages.append([uploaded_files , dcp, gbd, ibla, lc, mip, no, udcp, rows, ulap])
        caption.append(['Original', 'DCP', 'GBdehazingRCorrection', 'IBLA', 'LowComplexityDCP', 'MIP', 'NewOpticalModel', 'UDCP', 'Rows', 'ULAP'])
    
        cols = cycle(st.columns(4))
        for idx, img in enumerate(filteredImages):
            next(cols).image(img, width=230, caption=caption[idx])
        

    elif button2:
        filteredImages = []

        import CLAHE
        import ICM
        import GC
        import HE 
        import Rayleigh
        import RGHS 
        import UCM

        clahe = CLAHE.CLAHE2(uploaded_files)
        gc = GC.GC2(uploaded_files)
        he = HE.HE2(uploaded_files)
        icm = ICM.ICM2(uploaded_files)
        rayleigh = Rayleigh.Rayleigh2(uploaded_files)
        rghs = RGHS.RGHS2(uploaded_files)
        ucm = UCM.UCM2(uploaded_files)

        filteredImages.append([uploaded_files , clahe, gc, he, icm, rayleigh, rghs, ucm])
        caption.append(['Original', 'CLAHE', 'GC', 'HE', 'ICM','Rayleigh', 'RGHS', 'UCM'])
    
        cols = cycle(st.columns(4))
        for idx, img in enumerate(filteredImages):
            next(cols).image(img, width=230, caption=caption[idx], clamp=True, channels='BGR')
        
        


   