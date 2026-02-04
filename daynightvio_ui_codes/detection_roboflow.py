"""
Roboflow-based traffic violation detection (logic from apiapp.py).
No UI dependencies - used by Flask app.
"""
import os
import tempfile
from functools import lru_cache
import cv2
from roboflow import Roboflow

HEAD_RATIO = 0.25
HELMET_IOU = 0.20
FRAME_SKIP = 5


@lru_cache(maxsize=1)
def load_model():
    rf = Roboflow(api_key="HIOrnDyDEYCsm6i6tjfo")
    project = rf.workspace().project("iiser_mohali_violation_detection-bjzxd")
    return project.version(3).model


def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def process_frame(frame):
    model = load_model()
    h, w = frame.shape[:2]

    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(temp_path, frame)
    try:
        result = model.predict(temp_path, confidence=40, overlap=30).json()
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    riders, helmets, bikes = [], [], []

    for p in result["predictions"]:
        x1 = int(p["x"] - p["width"] / 2)
        y1 = int(p["y"] - p["height"] / 2)
        x2 = int(p["x"] + p["width"] / 2)
        y2 = int(p["y"] + p["height"] / 2)
        cls = p["class"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, cls, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if cls == "rider":
            riders.append([x1, y1, x2, y2])
        elif cls == "helmet":
            helmets.append([x1, y1, x2, y2])
        elif cls == "motorcycle":
            bikes.append([x1, y1, x2, y2])

    helmetless = 0
    overloaded = 0

    for bike in bikes:
        bike_riders = []
        for r in riders:
            cx = (r[0] + r[2]) // 2
            cy = (r[1] + r[3]) // 2
            if bike[0] <= cx <= bike[2] and bike[1] <= cy <= bike[3]:
                bike_riders.append(r)

        if len(bike_riders) > 1:
            overloaded += 1
            cv2.rectangle(frame, (bike[0], bike[1]), (bike[2], bike[3]), (255, 0, 0), 3)
            cv2.putText(frame, "OVERLOADING", (bike[0], bike[3] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        for r in bike_riders:
            head_y2 = int(r[1] + 0.25 * (r[3] - r[1]))
            head_box = [r[0], r[1], r[2], head_y2]
            has_helmet = False
            for h in helmets:
                if iou(head_box, h) >= 0.20:
                    has_helmet = True
                    break
            if not has_helmet:
                helmetless += 1
                cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 3)
                cv2.putText(frame, "HELMETLESS", (r[0], r[3] + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, helmetless, overloaded
