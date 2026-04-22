#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import sys

print("Loading model...")
try:
    model = YOLO('best_v11.pt')
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("✗ Error: best.pt not found in current directory")
    sys.exit(1)

print("Opening webcam...")
cap = cv2.VideoCapture("http://hailee-armoured-sensationally.ngrok-free.dev/video")

if not cap.isOpened():
    print("✗ Error: Cannot open webcam")
    print("Allow Terminal/Python camera access in System Preferences")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✓ Webcam opened")
print("\nStarting Live PPE Detection...")
print("Press 'q' to quit")
print("-" * 40)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=0.25, verbose=False)
        
        # Get annotated frame with boxes
        annotated_frame = results[0].plot()
        
        # Display live
        cv2.imshow('PPE Detection - Press Q to Quit', annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n✓ Stopping...")
            break

except KeyboardInterrupt:
    print("\n✓ Stopped")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Done!")

