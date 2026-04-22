import cv2
from ultralytics import YOLO

model_path = './best_v5n.pt'  # replace with your YOLOv8n model weights
source_path = './1.mp4'  # e.g. "video.mp4" or "image.jpg"

# Load model
model = YOLO(model_path)

# Open video or image
if source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
    # Video inference
    cap = cv2.VideoCapture(source_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow('YOLOv8n Detection', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
else:
    # Image inference
    results = model(source_path)
    annotated = results[0].plot()
    cv2.imshow('YOLOv8n Detection', annotated)
    cv2.waitKey(0)

cv2.destroyAllWindows()
