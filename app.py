import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="KFS OBB Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 KFS Object Detection (YOLOv8 OBB)")
st.caption("Webcam • Image • External Camera | Real-Time Ready")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Settings")

conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

img_size = st.sidebar.selectbox(
    "Inference Image Size",
    [640, 800, 1024],
    index=0
)

# ================= INPUT SOURCE =================
input_source = st.radio(
    "Select Input Source",
    ("Webcam", "Upload Image", "External Camera"),
    horizontal=True
)

st.divider()

# ================= YOLO INFERENCE FUNCTION =================
def run_yolo(frame):
    start = time.time()

    results = model(
        frame,
        imgsz=img_size,
        conf=conf_thres,
        verbose=False
    )

    annotated = results[0].plot()

    latency = (time.time() - start) * 1000  # ms
    fps = 1000 / latency if latency > 0 else 0

    return annotated, fps, latency

# ================= WEBCAM =================
if input_source == "Webcam":
    st.subheader("📷 Webcam Detection")

    run = st.checkbox("Start Webcam")

    frame_window = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Webcam not accessible")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read frame")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output, fps, latency = run_yolo(frame)

                cv2.putText(
                    output,
                    f"FPS: {fps:.1f} | Latency: {latency:.1f} ms",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                frame_window.image(output)

            cap.release()

# ================= IMAGE UPLOAD =================
elif input_source == "Upload Image":
    st.subheader("🖼️ Image Detection")

    file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if file:
        image = Image.open(file)
        image_np = np.array(image)

        st.image(image_np, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            output, fps, latency = run_yolo(image_np)

            st.success(f"Done | FPS: {fps:.1f} | Latency: {latency:.1f} ms")
            st.image(output, caption="Detected Output", use_column_width=True)

# ================= EXTERNAL CAMERA =================
elif input_source == "External Camera":
    st.subheader("🎥 External / Phone Camera")

    cam_index = st.number_input(
        "Camera Index (0 = webcam, 1/2 = external)",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    run_ext = st.checkbox("Start External Camera")

    frame_window = st.image([])

    if run_ext:
        cap = cv2.VideoCapture(cam_index)

        if not cap.isOpened():
            st.error("❌ Camera not detected")
        else:
            while run_ext:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read frame")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output, fps, latency = run_yolo(frame)

                cv2.putText(
                    output,
                    f"FPS: {fps:.1f} | Latency: {latency:.1f} ms",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                frame_window.image(output)

            cap.release()

# ================= FOOTER =================
st.divider()
st.caption("🚀 YOLOv8-OBB | Streamlit | Robot-Ready Deployment")
