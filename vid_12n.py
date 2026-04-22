import cv2
from ultralytics import YOLO

model_path = './best_v12n.pt'  # replace with your YOLOv12n weights file path
source_path = './3.mp4'  # e.g. "video.mp4" or "image.jpg"

model = YOLO(model_path)

if source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
    cap = cv2.VideoCapture(source_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow('YOLOv12n Detection', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
else:
    results = model(source_path)
    annotated = results[0].plot()
    cv2.imshow('YOLOv12n Detection', annotated)
    cv2.waitKey(0)

cv2.destroyAllWindows()
