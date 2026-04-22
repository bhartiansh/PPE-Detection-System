import streamlit as st
import cv2
import time
import backend

st.set_page_config(layout="wide")

st.title("PPE Monitoring Dashboard")

# Start backend once
if "started" not in st.session_state:
    backend.start_backend()
    st.session_state.started = True

# Model selection (stable)
if "model" not in st.session_state:
    st.session_state.model = "YOLOv12"

model_choice = st.selectbox(
    "Select YOLO Version",
    ["YOLOv5", "YOLOv8", "YOLOv11", "YOLOv12"],
    index=["YOLOv5","YOLOv8","YOLOv11","YOLOv12"].index(st.session_state.model)
)

if model_choice != st.session_state.model:
    backend.set_model(model_choice)
    st.session_state.model = model_choice

# Layout
col1, col2, col3 = st.columns(3)

frame_boxes = {
    "Laptop": col1.empty(),
    "Mobile": col2.empty(),
    "ESP32": col3.empty()
}

st.markdown("---")

colA, colB = st.columns(2)
alert_box = colA.empty()
chart_box = colB.empty()

# 🔥 LIVE VIDEO LOOP (ONLY updates frames, NOT page)
while True:
    for cam in backend.frames:
        frame = backend.frames.get(cam)

        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_boxes[cam].image(frame, channels="RGB")

    if backend.alerts:
        alert_box.error(backend.alerts[-1])

    chart_box.bar_chart(backend.violations)

    time.sleep(0.05)  # smooth video (~20 FPS)