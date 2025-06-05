import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# Streamlit page setup
st.set_page_config(page_title="Smart Vision App", layout="centered")

# Sidebar
mode = st.sidebar.selectbox("Select Mode", ["Face Detection", "Object Detection"])
input_type = st.sidebar.radio("Select Input", ["Webcam", "Upload Image/Video"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
run_detection = st.sidebar.button("Run Detection")

# Load appropriate YOLO model
if mode == "Face Detection":
    model = YOLO("face_yolov8n.pt")
else:
    model = YOLO("yolov8n.pt")

st.title("🧠 Smart Vision App")
st.markdown("A simple app for **face detection** and **object detection** using YOLOv8.")

# Detection function
def run_yolo_detection(frame, confidence_threshold):
    results = model(frame)[0]
    detection_counts = {}

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf[0])
        cls_id = int(r.cls[0]) if hasattr(r, 'cls') and r.cls is not None else -1

        # Decide label and color
        if conf >= confidence_threshold and cls_id in model.names:
            label = model.names[cls_id]
            color = (0, 255, 0)  # Green for known
        else:
            label = "Unknown"
            color = (0, 0, 255)  # Red for unknown

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Count occurrences
        detection_counts[label] = detection_counts.get(label, 0) + 1

    return frame, detection_counts

# Webcam
if run_detection and input_type == "Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    st.info("Press **Stop** button (top right) to exit webcam.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        result_frame, counts = run_yolo_detection(frame, confidence_threshold)
        stframe.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Show count below the image
        st.markdown("### Detections:")
        for label, count in counts.items():
            st.markdown(f"- **{label}**: {count}")

    cap.release()

# Upload: image or video
uploaded_file = None
if input_type == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov"])

if run_detection and uploaded_file is not None:
    # Handle image
    if uploaded_file.type.startswith("image"):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        frame = cv2.imread(temp_file.name)

        if frame is None:
            st.error("Could not read the uploaded image.")
        else:
            result_frame, counts = run_yolo_detection(frame, confidence_threshold)
            st.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Show detection summary
            st.markdown("### Detections:")
            for label, count in counts.items():
                st.markdown(f"- **{label}**: {count}")

    # Handle video
    elif uploaded_file.type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_frame, counts = run_yolo_detection(frame, confidence_threshold)
            stframe.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Show detection summary
            st.markdown("### Detections:")
            for label, count in counts.items():
                st.markdown(f"- **{label}**: {count}")

        cap.release()
