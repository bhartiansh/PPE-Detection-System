import cv2
from ultralytics import YOLO
import threading
import requests
import numpy as np
import time

# 🔥 Load all models once
print("Loading models...")
models = {
    "YOLOv5": YOLO("best_v5n.pt"),
    "YOLOv8": YOLO("best_v8n.pt"),
    "YOLOv11": YOLO("best_v11.pt"),
    "YOLOv12": YOLO("best_v12n.pt")
}

current_model_name = "YOLOv12"
model = models[current_model_name]

print("✓ Models loaded")

# 🔥 TEMP: START WITH ONLY LAPTOP (IMPORTANT)
sources = {
    "Laptop": 0,
    "Mobile": "http://postage-indicating-modems-mathematical.trycloudflare.com/video",
    "ESP32": "http://192.168.1.16:81/stream"
}
frames = {}
alerts = []
violations = {k: 0 for k in sources}

lock = threading.Lock()

def set_model(name):
    global model, current_model_name

    if name != current_model_name:
        model = models[name]
        current_model_name = name
        print(f"Switched to {name}")

# 🔥 FRAME PROCESSING
def process_frame(name, frame):
    global model

    frame = cv2.resize(frame, (480, 360))

    results = model(frame, conf=0.3, verbose=False)
    annotated = results[0].plot()

    labels = [model.names[int(c)] for c in results[0].boxes.cls]

    if "NO-Hardhat" in labels or "NO-Safety Vest" in labels:
        with lock:
            violations[name] += 1
            alerts.append(f"⚠️ {name} violation")

    with lock:
        frames[name] = annotated

# 🔥 CAMERA HANDLER
def process_camera(name, src):
    print(f"{name} thread started")

    # 💻 Laptop (Mac fix)
    if src == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"❌ {name} failed to open")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        process_frame(name, frame)

# 🔥 START THREADS
print("Starting camera threads...")

def start_backend():
    print("Starting backend...")

    for name, src in sources.items():
        print(f"Starting {name}...")

        threading.Thread(
            target=process_camera,
            args=(name, src),
            daemon=True
        ).start()