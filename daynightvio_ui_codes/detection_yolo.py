"""
YOLO-based traffic violation detection (logic from app.py).
No UI dependencies - used by Flask app.
"""
import os
from functools import lru_cache
import cv2
import numpy as np
from ultralytics import YOLO


def get_yolo_model_path():
    """Model path: prefer project newmodel_yolo, fallback to original path."""
    base = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(base, "newmodel_yolo", "best.pt")
    if os.path.isfile(local):
        return local
    return r"D:\VITCS_FINAL-20250814T172227Z-1-001\updations\new one\newmodel_yolo\best.pt"


@lru_cache(maxsize=1)
def load_model():
    return YOLO(get_yolo_model_path())


def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def process_frame(frame, conf=0.25):
    model = load_model()
    results = model.predict(frame, conf=conf, verbose=False)
    boxes = results[0].boxes

    riders, helmets, bikes = [], [], []

    for b in boxes:
        cls = model.names[int(b.cls)]
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, cls, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if cls == "rider":
            riders.append([x1, y1, x2, y2])
        elif cls == "helmet":
            helmets.append([x1, y1, x2, y2])
        elif cls == "motorcycle":
            bikes.append([x1, y1, x2, y2])

    helmetless = 0
    overloaded = 0

    for r in riders:
        has_helmet = False
        for h in helmets:
            if iou(r, h) > 0.1:
                has_helmet = True
                break
        if not has_helmet:
            helmetless += 1
            cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 3)
            cv2.putText(frame, "Helmetless Rider", (r[0], r[3] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if len(riders) > len(bikes):
        overloaded = 1
        for b in bikes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 3)
            cv2.putText(frame, "Overloading", (b[0], b[3] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Helmetless Riders: {helmetless}", (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(frame, f"Overloaded Bikes: {overloaded}", (350, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame, helmetless, overloaded
