#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import sys

print("Loading model...")
try:
    model = YOLO('best_v12n.pt')
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("✗ Error: best_v12n.pt not found in current directory")
    sys.exit(1)

print("Opening ESP32 camera stream...")
cap = cv2.VideoCapture("http://192.168.1.16:81/stream")

if not cap.isOpened():
    print("✗ Error: Cannot open ESP32 stream")
    sys.exit(1)

# Reduce buffer to prevent lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("✓ Stream opened")
print("\nStarting Live PPE Detection...")
print("Press 'q' to quit")
print("-" * 40)

frame_count = 0

try:
    while True:

        ret, frame = cap.read()

        # Handle dropped frames
        if not ret:
            print("Frame dropped, reconnecting...")
            cap.release()
            cap = cv2.VideoCapture("http://192.168.1.16:81/stream")
            continue

        frame_count += 1

        # Skip frames to improve speed
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (640, 480))

        # Run YOLO detection
        results = model(frame, conf=0.25, verbose=False)

        # Draw boxes
        annotated_frame = results[0].plot()

        # Display
        cv2.imshow('PPE Detection - Press Q to Quit', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✓ Stopping...")
            break

except KeyboardInterrupt:
    print("\n✓ Stopped")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Done!")