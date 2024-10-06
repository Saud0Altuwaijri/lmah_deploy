import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import time


col1, col2 , col3 = st.columns([0.0001, 0.0001 ,0.00005])
# Title of the Streamlit app
st.title('LMAH')
# st.subheader('LMMAH is a project aimed at reducing the waiting time for emergency vehicles at traffic signals using artificial intelligence, computer vision, and YOLO technology. By identifying emergency vehicles, the system automatically changes the traffic light to green, ensuring a quicker response.')

st.markdown("##### **LMAH** is a project aimed at reducing the waiting time for emergency vehicles at traffic signals using artificial intelligence, computer vision, and YOLO technology. By identifying emergency vehicles, the system automatically changes the traffic light to green, ensuring a quicker response.")
# Load your pre-trained YOLO model
model = YOLO(r"yolov8_fine_tuned_model.pt")

# Confidence threshold
confidence_threshold = 0.6

# Add logos in the top-left and top-right corners


# Load the logos
with col1:
    st.image("tuwaiq.png", width=200 ,)  # Adjust width as needed

with col2:
    st.image("sdaia.png", width=200)  # Adjust width as needed
with col3:
    st.image("New-LMAH-LOGO.jpg" , width=200)
# Function to display a virtual video
def display_virtual_video(video_path, placeholder):
    cap = cv2.VideoCapture(video_path)
    # while cap.isOpened():
    while True:
        ret, frame = cap.read()
        # if not ret:
            # break
        if not ret:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video if it ends
            # continue
            break
        # Display the video frame
        placeholder.image(frame, channels="BGR")
        time.sleep(0.03)
    

virtual_video_1 = r"predict-lmah-model.mp4"
virtual_video_placeholder_1 = st.empty()
display_virtual_video(virtual_video_1, virtual_video_placeholder_1)


# Upload video file for YOLO processing
st.markdown('#### Try it yoreself!')
uploaded_video = st.file_uploader(".", type=["mp4", "avi", "mov"])

# Display YOLO processed video if uploaded
if uploaded_video is not None:
    # Temporary file to store video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(tfile.name)

    # Create a placeholder for YOLO video display
    yolo_video_placeholder = st.empty()

    # Process YOLO video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection using the YOLO model
        results = model(frame)

        # Filter out detections based on confidence threshold
        for result in results:
            boxes = result.boxes  # Access bounding boxes and other info

            for box in boxes:
                conf = box.conf  # Confidence score for each box
                if conf >= confidence_threshold:
                    # Annotate frame with detections
                    annotated_frame = result.plot()

        # Show the YOLO processed frame
        yolo_video_placeholder.image(annotated_frame, channels="BGR")

    # Release video capture when done
    cap.release()
    os.unlink(tfile.name)
    st.stop()
st.markdown('#### Emergency Vehicle Before And After LMAH!')

# Virtual video paths (use paths to your actual video files)

virtual_video_2 = r"yolo_before_and_after.mp4"
virtual_video_placeholder_2 = st.empty()
display_virtual_video(virtual_video_2, virtual_video_placeholder_2)


st.title('Reinforcement Learing (RL)')

st.markdown('#### Opening traffic signals for emergency vehicles can disrupt normal traffic flow, leading to increased congestion. As well as the congestion that happens already from a regular flow system. ')



st.write("### Video Before And After Reinforcement Learing (RL)")

virtual_video_3 = r"RL_before_and_after.mp4"
virtual_video_placeholder_3 = st.empty()
display_virtual_video(virtual_video_3, virtual_video_placeholder_3)