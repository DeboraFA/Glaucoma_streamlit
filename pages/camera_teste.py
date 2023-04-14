import streamlit as st
from PIL import Image
import cv2
import av
import numpy as np
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.count = 0
        self.frames = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.count < 5:
            self.frames.append(frame.to_ndarray(format="bgr24"))
            self.count += 1
            # imagee = Image.open(frame)
            # cv2.imwrite(f"frame_{self.count}.jpg", imagee)
        else:
            # Stop the video stream and plot the saved frames
            webrtc_streamer.stop()
            for i in range(5):
                st.image(self.frames[i], channels="BGR")

webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
)
