import streamlit as st
import streamlit_webrtc as webrtc

def main():
    # Create a WebRTC video chat component
    webrtc_streamer = webrtc.Streamer(
        key="example",
        mode=webrtc.StreamerMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": True},
    )

    # Render the video chat component
    if webrtc_streamer is not None:
        video_stream = webrtc_streamer.subscribe("video")
        st.video(video_stream)

if __name__ == "__main__":
    main()
