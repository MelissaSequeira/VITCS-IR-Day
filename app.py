# from flask import Flask, render_template, request, Response
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# from datetime import datetime

# app = Flask(__name__)

# # ================= PATHS =================
# VIOLATION_DIR = "static/violations"
# os.makedirs(VIOLATION_DIR, exist_ok=True)

# # ================= MODELS =================
# day_violation_model = YOLO(
#     r"D:\VITCS_FINAL-20250814T172227Z-1-001\new one\violation_integration\violation_logic\daymodel\best.pt"
# )

# night_violation_model = YOLO(
#     r"D:\VITCS_FINAL-20250814T172227Z-1-001\new one\violation_integration\violation_logic\nightmodel\best.pt"
# )

# # ================= HELMET CHECK =================
# def helmet_on_rider(rider, helmet):
#     rx1, ry1, rx2, ry2 = rider
#     hx1, hy1, hx2, hy2 = helmet
#     cx = (hx1 + hx2) // 2
#     cy = (hy1 + hy2) // 2
#     head_limit = ry1 + (ry2 - ry1) * 0.4
#     return rx1 <= cx <= rx2 and ry1 <= cy <= head_limit

# # ================= VIOLATION PROCESS =================
# def process_frame(frame, conf, model, save_violation=False):
#     results = model.predict(frame, conf=conf, verbose=False)
#     boxes = results[0].boxes

#     riders, helmets, bikes = [], [], []

#     for b in boxes:
#         cls = model.names[int(b.cls)]
#         x1, y1, x2, y2 = map(int, b.xyxy[0])

#         if cls == "rider":
#             riders.append([x1,y1,x2,y2])
#         elif cls == "helmet":
#             helmets.append([x1,y1,x2,y2])
#         elif cls == "motorcycle":
#             bikes.append([x1,y1,x2,y2])

#         cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
#         cv2.putText(frame,cls,(x1,y1-5),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

#     violation = False

#     for r in riders:
#         if not any(helmet_on_rider(r,h) for h in helmets):
#             violation = True
#             cv2.rectangle(frame,(r[0],r[1]),(r[2],r[3]),(0,0,255),3)
#             cv2.putText(frame,"Helmetless",(r[0],r[3]+20),
#                         cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

#     if len(riders) > len(bikes):
#         violation = True

#     if save_violation and violation:
#         name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
#         cv2.imwrite(os.path.join(VIOLATION_DIR, name), frame)

#     return frame

# # ================= VIDEO STREAM =================
# def stream_video(conf, model):
#     cap = cv2.VideoCapture("temp_video.mp4")
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = process_frame(frame, conf, model, True)
#         _, buffer = cv2.imencode(".jpg", frame)
#         yield (b"--frame\r\n"
#                b"Content-Type: image/jpeg\r\n\r\n" +
#                buffer.tobytes() + b"\r\n")
#     cap.release()

# # ================= MAIN PAGE =================
# @app.route("/", methods=["GET","POST"])
# def index():
#     if request.method == "POST":
#         conf = float(request.form["conf"])
#         mode = request.form["mode"]          # image / video
#         time_mode = request.form["time"]     # day / night
#         file = request.files["file"]

#         model = night_violation_model if time_mode == "night" else day_violation_model

#         if mode == "image":
#             img = np.frombuffer(file.read(), np.uint8)
#             frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
#             process_frame(frame, conf, model, True)

#         elif mode == "video":
#             file.save("temp_video.mp4")

#     images = sorted(os.listdir(VIOLATION_DIR), reverse=True)
#     return render_template("vio.html", violation_images=images)

# @app.route("/video_feed/<time_mode>")
# def video_feed(time_mode):
#     model = night_violation_model if time_mode == "night" else day_violation_model
#     return Response(
#         stream_video(0.25, model),
#         mimetype="multipart/x-mixed-replace; boundary=frame"
#     )

# # ================= RUN =================
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, Response, redirect, url_for
from ultralytics import YOLO
import cv2, os, sqlite3, numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ===================== PATHS =====================
UPLOAD_DIR = "static/uploads"
VIOLATION_DIR = "static/violations"
DB_NAME = "database.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VIOLATION_DIR, exist_ok=True)

# ===================== MODELS =====================
day_model = YOLO( r"D:\VITCS_FINAL-20250814T172227Z-1-001\new one\violation_integration\violation_logic\daymodel\best.pt")
night_model = YOLO(r"D:\VITCS_FINAL-20250814T172227Z-1-001\new one\violation_integration\violation_logic\nightmodel\best.pt")

plate_model = YOLO(r"D:\VITCS_FINAL-20250814T172227Z-1-001\new one\violation_integration\ocr_logic\plate.pt")
char_model  = YOLO(r"D:\VITCS_FINAL-20250814T172227Z-1-001\new one\violation_integration\ocr_logic\char_detect_new.pt")

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ===================== DATABASE =====================
def init_db():
    with sqlite3.connect(DB_NAME) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            cropped_plate TEXT,
            number_plate TEXT,
            violation TEXT,
            date TEXT,
            time TEXT
        )
        """)
init_db()

def stream_video(conf, model):
    cap = cv2.VideoCapture("temp_video.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, conf, model, True)
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")
    cap.release()

# ===================== HELMET LOGIC =====================
def helmet_on_rider(rider, helmet):
    rx1, ry1, rx2, ry2 = rider
    hx1, hy1, hx2, hy2 = helmet
    cx = (hx1 + hx2) // 2
    cy = (hy1 + hy2) // 2
    head_limit = ry1 + (ry2 - ry1) * 0.4
    return rx1 <= cx <= rx2 and ry1 <= cy <= head_limit

@app.route("/delete/<int:record_id>")
def delete_record(record_id):
    with sqlite3.connect(DB_NAME) as c:
        c.execute("DELETE FROM vehicle_data WHERE id=?", (record_id,))
    return redirect(url_for("records"))

def process_frame(frame, conf, model, save_violation=False):
    results = model.predict(frame, conf=conf, verbose=False)
    boxes = results[0].boxes

    riders, helmets, bikes = [], [], []

    for b in boxes:
        cls = model.names[int(b.cls)]
        x1, y1, x2, y2 = map(int, b.xyxy[0])

        if cls == "rider":
            riders.append([x1,y1,x2,y2])
        elif cls == "helmet":
            helmets.append([x1,y1,x2,y2])
        elif cls == "motorcycle":
            bikes.append([x1,y1,x2,y2])

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,cls,(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    violation = False

    for r in riders:
        if not any(helmet_on_rider(r,h) for h in helmets):
            violation = True
            cv2.rectangle(frame,(r[0],r[1]),(r[2],r[3]),(0,0,255),3)
            cv2.putText(frame,"Helmetless",(r[0],r[3]+20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    if len(riders) > len(bikes):
        violation = True

    if save_violation and violation:
        name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        cv2.imwrite(os.path.join(VIOLATION_DIR, name), frame)

    return frame

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f: return redirect(url_for("index"))

    name = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)

    crop = detect_and_crop(path)
    if crop:
        plate_text = read_plate(crop)
    else:
        plate_text = "Not Detected"
        crop = "None" # Handle no-crop case

    now = datetime.now()
    with sqlite3.connect(DB_NAME) as c:
        c.execute("""
        INSERT INTO vehicle_data
        (filename, cropped_plate, number_plate, violation, date, time)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name,
            os.path.basename(crop) if crop != "None" else "None",
            plate_text,
            "None",
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ))

    return redirect(url_for("index"))
# ===================== OCR FUNCTIONS =====================
def preprocess_plate(img):
    """
    Aggressive enhancement for night/grainy images.
    """
    # 1. Upscale: Helps separate merged letters (like 'M' and 'H')
    scale = 2
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE (Adaptive Contrast): Fixes glare and shadows
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. Denoise: Removes graininess from night shots
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    # 5. Sharpen: Makes edges of numbers crisp
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Return BGR
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def detect_and_crop(image_path):
    img = cv2.imread(image_path)
    if img is None: return None

    # Use high confidence to find the plate
    results = plate_model.predict(img, conf=0.40, verbose=False)
    
    best_box = None
    max_conf = -1

    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            if box.conf > max_conf:
                max_conf = box.conf
                best_box = box

    if best_box is None: return None

    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

    # SMART PADDING:
    # Add extra space around the plate so characters on the edge aren't cut
    h, w = img.shape[:2]
    pad_w = int((x2 - x1) * 0.12) # 12% width padding
    pad_h = int((y2 - y1) * 0.20) # 20% height padding

    crop = img[max(0, y1-pad_h):min(h, y2+pad_h),
               max(0, x1-pad_w):min(w, x2+pad_w)]

    out = os.path.join(UPLOAD_DIR, "crop_" + os.path.basename(image_path))
    cv2.imwrite(out, crop)
    return out

def fix_indian_plate(text):
    """
    Forcefully corrects text based on Indian License Plate Format:
    SS DD SS NNNN (State District Series Number)
    """
    text = text.upper().replace(" ", "").replace("-", "")
    
    # Mappings for common OCR errors
    dict_char_to_num = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8'}
    dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}

    text_list = list(text)
    length = len(text_list)

    # RULE 1: Fix State Code (First 2 chars must be Letters)
    if length >= 2:
        # Heuristic for Maharashtra (your dataset)
        # If it detects LH, VH, WH -> convert to MH
        if text_list[1] == 'H' and text_list[0] in ['L', 'V', 'W', 'N']:
            text_list[0] = 'M'
        
        # Force Letters for pos 0 and 1
        for i in [0, 1]:
            if text_list[i] in dict_num_to_char:
                text_list[i] = dict_num_to_char[text_list[i]]

    # RULE 2: Fix District Code (Pos 2 and 3 must be Numbers)
    if length >= 4:
        for i in [2, 3]:
            if text_list[i] in dict_char_to_num:
                text_list[i] = dict_char_to_num[text_list[i]]

    # RULE 3: Fix Last 4 Characters (Must be Numbers)
    if length > 4:
        suffix_len = 4 if length >= 8 else (length - 6) # Approximate
        start_idx = length - suffix_len
        
        for i in range(start_idx, length):
            if text_list[i] in dict_char_to_num:
                text_list[i] = dict_char_to_num[text_list[i]]

    # Re-assemble
    final = "".join(text_list)
    
    # Validation Regex (Optional check)
    # If the fix resulted in something weird, fallback or keep as is
    return final

def dedupe(chars):
    # Sort by confidence to keep the best version of a character
    chars = sorted(chars, key=lambda x: x["conf"], reverse=True)
    final = []
    
    for c in chars:
        is_duplicate = False
        for f in final:
            # Check physical overlap
            dist = np.sqrt((c["cx"] - f["cx"])**2 + (c["cy"] - f["cy"])**2)
            if dist < min(c["w"], f["w"]) * 0.6:
                is_duplicate = True
                break
        if not is_duplicate:
            final.append(c)
            
    return final

def group_rows(chars):
    # Handle stacked plates (2 rows) vs single line
    rows = []
    # Sort by Y position to find top and bottom lines
    chars = sorted(chars, key=lambda x: x["cy"])
    
    for c in chars:
        placed = False
        for r in rows:
            # If the char is vertically close to the row's average Y
            if abs(np.mean([x["cy"] for x in r]) - c["cy"]) < c["h"] * 0.6:
                r.append(c)
                placed = True
                break
        if not placed:
            rows.append([c])
    
    # Sort rows Top to Bottom
    rows = sorted(rows, key=lambda r: np.mean([x["cy"] for x in r]))
    return rows

def detect_chars(img):
    # Low confidence to ensure we don't miss faint characters
    results = char_model.predict(img, conf=0.15, imgsz=640, verbose=False)
    detections = []
    
    h_img, w_img, _ = img.shape

    for r in results:
        if r.boxes is None: continue

        for (x1,y1,x2,y2), cls, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy()
        ):
            width = x2 - x1
            height = y2 - y1
            
            # --- FILTER NOISE ---
            # Reject very small specks (noise)
            if height < h_img * 0.15: continue 
            # Reject horizontal lines (borders)
            if width > height * 2.5: continue

            detections.append({
                "char": CHARS[int(cls)],
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "x1": x1,
                "w": width,
                "h": height,
                "conf": conf
            })

    return detections

def read_plate(crop_path):
    img = cv2.imread(crop_path)
    if img is None: return "Not Detected"

    # 1. Enhanced Preprocessing
    img = preprocess_plate(img)

    # 2. Detect
    chars = detect_chars(img)
    if len(chars) < 3: return "Not Detected" # Too few chars

    # 3. Clean
    chars = dedupe(chars)
    rows = group_rows(chars)

    # 4. Assemble Text
    text = ""
    for r in rows:
        # Sort characters Left to Right within the row
        r.sort(key=lambda x: x["x1"])
        text += "".join(c["char"] for c in r)

    # 5. Logic Correction
    final_text = fix_indian_plate(text)

    return final_text

# ===================== ROUTES =====================
@app.route("/", methods=["GET","POST"])
def vio():
    if request.method=="POST":
        f = request.files["file"]
        conf = float(request.form["conf"])
        time = request.form["time"]
        mode = request.form["mode"]
        model = night_model if time=="night" else day_model

        if mode=="image":
            img = cv2.imdecode(np.frombuffer(f.read(),np.uint8),1)
            process_frame(img, conf, model, True)
        else:
            f.save("temp.mp4")

    images = sorted(os.listdir(VIOLATION_DIR), reverse=True)
    return render_template("vio.html", violation_images=images)

@app.route("/do_ocr/<img>")
def do_ocr(img):
    path = os.path.join(VIOLATION_DIR, img)
    crop = detect_and_crop(path)
    plate = read_plate(crop) if crop else "Not Detected"

    now = datetime.now()
    with sqlite3.connect(DB_NAME) as c:
        c.execute("""
        INSERT INTO vehicle_data
        (filename,cropped_plate,number_plate,violation,date,time)
        VALUES (?,?,?,?,?,?)
        """,(img, os.path.basename(crop) if crop else "None",
             plate,"Helmetless", now.date(), now.strftime("%H:%M:%S")))

    return redirect(url_for("records"))

@app.route("/records")
def records():
    with sqlite3.connect(DB_NAME) as c:
        data = c.execute("SELECT * FROM vehicle_data ORDER BY id DESC").fetchall()
    return render_template("index.html", data=data)

def process_violation_folder():
    images = os.listdir(VIOLATION_DIR)

    for img_name in images:
        img_path = os.path.join(VIOLATION_DIR, img_name)

        # Skip if already processed
        with sqlite3.connect(DB_NAME) as c:
            already = c.execute(
                "SELECT 1 FROM vehicle_data WHERE filename=?",
                (img_name,)
            ).fetchone()
            if already:
                continue

        # Detect & OCR
        crop = detect_and_crop(img_path)
        if crop:
            plate_text = read_plate(crop)
        else:
            plate_text = "Not Detected"
            crop = "None"

        now = datetime.now()
        with sqlite3.connect(DB_NAME) as c:
            c.execute("""
                INSERT INTO vehicle_data
                (filename, cropped_plate, number_plate, violation, date, time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                img_name,
                os.path.basename(crop) if crop != "None" else "None",
                plate_text,
                "Helmetless",
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S")
            ))

# ===================== RUN =====================
if __name__=="__main__":
    app.run(debug=True)


