import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# LOAD MODEL & VIDEO
# -----------------------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(r"D:\VITCS_FINAL-20250814T172227Z-1-001\updations\new one\3759638031-preview.mp4")

FRAME_W, FRAME_H = 1020, 600
VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

# -----------------------------
# HSV RANGES FOR TRAFFIC LIGHT
# -----------------------------
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

GREEN_LOWER = np.array([40, 40, 40])
GREEN_UPPER = np.array([90, 255, 255])

# -----------------------------
# STOP LINE DRAWING
# -----------------------------
stop_line = []

def draw_line(event, x, y, flags, param):
    global stop_line
    if event == cv2.EVENT_LBUTTONDOWN and len(stop_line) < 2:
        stop_line.append((x, y))

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_line)

# -----------------------------
# TRAFFIC LIGHT COLOR
# -----------------------------
def traffic_light_color(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y1+(y2-y1)//2, x1:x2]

    if roi.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1) + \
               cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    red_ratio = cv2.countNonZero(red_mask) / roi.size
    green_ratio = cv2.countNonZero(green_mask) / roi.size

    if red_ratio > 0.02 and red_ratio > green_ratio:
        return "RED"
    elif green_ratio > 0.02 and green_ratio > red_ratio:
        return "GREEN"
    else:
        return "UNKNOWN"

# -----------------------------
# LINE SIDE FUNCTION
# -----------------------------
def point_side(p, a, b):
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

# -----------------------------
# MEMORY
# -----------------------------
vehicle_states = {}   # id -> last side
violated_ids = set()

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    light_state = "UNKNOWN"

    results = model.track(frame, persist=True, conf=0.4, verbose=False)

    # -----------------------------
    # TRAFFIC LIGHT DETECTION
    # -----------------------------
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label == "traffic light":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            light_state = traffic_light_color(frame, (x1, y1, x2, y2))

            color = (0, 255, 0) if light_state == "GREEN" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, light_state, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            break

    # -----------------------------
    # DRAW STOP LINE
    # -----------------------------
    if len(stop_line) == 2:
        cv2.line(frame, stop_line[0], stop_line[1], (0, 0, 255), 2)
        cv2.putText(frame, "STOP LINE", stop_line[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # -----------------------------
    # VEHICLE VIOLATION LOGIC
    # -----------------------------
    for box in results[0].boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        if label not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        if len(stop_line) == 2:
            side = point_side((cx, cy), stop_line[0], stop_line[1])

            if track_id not in vehicle_states:
                vehicle_states[track_id] = side
            else:
                prev_side = vehicle_states[track_id]

                # Crossing happened
                if prev_side * side < 0:
                    if light_state == "RED":
                        violated_ids.add(track_id)

                vehicle_states[track_id] = side

        # -----------------------------
        # DRAW BOX
        # -----------------------------
        if track_id in violated_ids:
            color = (0, 0, 255)
            text = f"VIOLATION {track_id}"
        else:
            color = (0, 255, 0)
            text = f"{label} {track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
