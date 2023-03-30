from streamlit_webrtc import VideoProcessorBase, RTCConfiguration,WebRtcMode,webrtc_streamer
from utils import *
import cv2
import streamlit as st
import mediapipe as mp
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
    st.header("Live stream processing")

    sign_language_det = "Sign Language Live Detector"
    app_mode = st.sidebar.selectbox( "Choose the app mode",
        [
            sign_language_det
        ],
    )

    st.subheader(app_mode)

    if app_mode == sign_language_det:
        sign_language_detector()
 

def sign_language_detector():

    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(img,format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )


if __name__ == "__main__":
    main()