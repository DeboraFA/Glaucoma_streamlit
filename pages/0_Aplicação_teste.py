import streamlit as st 
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import cv2
from yolo_predictions import YOLO_Pred
from PIL import Image
from glaucoma_segmentation import test_segmentation_disc, test_segmentation_cup, CDR, BVR, NRR


# load yolo model
yolo = YOLO_Pred('./models/best.onnx',
                 './models/data.yaml')

i = 0

def video_frame_callback(frame):
    
    img = frame.to_ndarray(format="bgr24")
    # any operation 
    #flipped = img[::-1,:,:]
    pred_img, confidence, boxes, index, img_copy = yolo.predictions(img)
    global i
    if confidence >= 90 and i<3:  
        boxes_np = np.array(boxes).tolist()
        for ind in index:      
            # extract bounding box
            x,y,w,h = boxes_np[ind]
            roi_image = img_copy[y-10:y+h+10,x-10:x+w+10]
                     
            cv2.imwrite(f"./saida/frame_{i}.jpg", roi_image)
                # st.camera_input(roi_image)
        i = i+1

    # return av.VideoFrame.from_ndarray(pred_img, format="bgr24")
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="example", 
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video":True,"audio":False})


# # test evaluate
# timm-mobilenetv3_large_075
seg_encoder = 'mobilenet_v2' # https://github.com/qubvel/segmentation_models.pytorch
seg_model = 'Unetplusplus' #  Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3,DeepLabV3Plus


button = st.button('CLIQUE')
if button:
    for img in range(3):

        col1 , col2, col3 = st.columns(3)
        with col1:
            image = Image.open(f'./saida/frame_{img}.jpg')
            
            st.image(image,  width=240, caption=f'Image{i}')
        with col2:
            contours_cup = test_segmentation_cup(image, seg_encoder, seg_model)
            contours_disc = test_segmentation_disc(image, seg_encoder, seg_model)

            result = np.array(image).copy()
            result = cv2.resize(result, (224,224))
            for cnt_disc in contours_cup:
                img_contours1 = cv2.drawContours(result, [cnt_disc], -1, (0, 255, 0), 2)

            for cnt_cup in contours_disc:
                img_contours = cv2.drawContours(img_contours1, [cnt_cup], -1, (255, 0, 0), 2)

            st.image(img_contours, width=257, caption="segmentacao")

        with col3:
            cdr = CDR(cnt_disc, cnt_cup)
            nrr = NRR(cnt_disc, cnt_cup, np.array(image))
            bvr = BVR(cnt_disc, np.array(image))

            st.write(f'CDR: {np.round(cdr,2)}')
            st.write(f'NRR: {np.round(nrr,2)}')
            st.write(f'BVR: {np.round(bvr,2)}')
