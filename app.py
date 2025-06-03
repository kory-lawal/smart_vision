import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

# Load YOLOv8 model (supports face and object detection)
model = YOLO("yolov8n.pt")  # lightweight version

st.set_page_config(page_title="Smart Vision App", layout="centered")

st.title("🧠 Smart Vision App")
st.markdown("A simple app for **face detection** and **object detection** using YOLOv8.")

# Sidebar options
mode = st.sidebar.selectbox("Select Mode", ["Face Detection", "Object Detection"])
input_type = st.sidebar.radio("Select Input", ["Webcam", "Upload Image/Video"])
run_detection = st.sidebar.button("Run Detection")

# Helper to run detection
def run_yolo_detection(frame):
    results = model(frame)[0]
    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cls = int(r.cls[0])
        conf = r.conf[0]
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return frame

# Webcam input
if run_detection and input_type == "Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    st.info("Press **Stop** button (top right) to exit webcam.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        # Run detection
        result_frame = run_yolo_detection(frame)
        stframe.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

# Upload input
elif run_detection and input_type == "Upload Image/Video":
    file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov"])

    if file is not None:
        if file.type.startswith("image"):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(file.read())
            frame = cv2.imread(temp_file.name)
            result_frame = run_yolo_detection(frame)
            st.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        elif file.type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result_frame = run_yolo_detection(frame)
                stframe.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            cap.release()
