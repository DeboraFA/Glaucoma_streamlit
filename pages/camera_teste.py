import cv2
import streamlit as st

def main():
    st.title("Camera Stream")
    run_camera()

def run_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Unable to capture camera.")
            break

        # Display the video stream in Streamlit
        st.image(frame, channels="BGR")

        if st.button("Stop Camera"):
            break

    # Release the camera and close the Streamlit app
    cap.release()
    st.write("Camera Stopped")

if __name__ == "__main__":
    main()
