from flask import Flask, render_template, request, redirect, url_for, Response
import os, sqlite3, cv2
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import Counter, defaultdict
import difflib

# ================= CONFIG =================
app = Flask(__name__)
UPLOAD_DIR = "static/uploads"
DB_NAME = "database.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

FRAME_SKIP = 5
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ================= MODELS =================
vehicle_model  = YOLO("yolov8n.pt")
plate_model    = YOLO(r"D:/VITCS_FINAL-20250814T172227Z-1-001/new one/violation_integration/models/plate.pt") # Ensure this is in models/
vid_char_model = YOLO(r"D:/VITCS_FINAL-20250814T172227Z-1-001/new one/violation_integration/models/char_detect_new.pt")   # Video OCR model
img_char_model = YOLO(r"D:/VITCS_FINAL-20250814T172227Z-1-001/new one/violation_integration/models/char_detect_new.pt")   # Image OCR model     
night_model = YOLO(r"D:/VITCS_FINAL-20250814T172227Z-1-001/new one/violation_integration/models/best.pt") 

# ================= DATABASE =================
def init_db():
    with sqlite3.connect(DB_NAME) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT,
            track_id TEXT,
            vehicle_img TEXT,
            cropped_plate TEXT,
            number_plate TEXT,
            date TEXT,
            time TEXT
        )
        """)
init_db()


# =====================================================================
#                        1. IMAGE PIPELINE LOGIC
# =====================================================================

def img_preprocess_plate(img):
    scale = 2
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    kernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def img_detect_and_crop(image_path):
    img = cv2.imread(image_path)
    if img is None: return None

    results = plate_model.predict(img, conf=0.40, verbose=False)

    best_box, max_conf = None, -1

    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            if box.conf > max_conf:
                max_conf, best_box = box.conf, box

    if best_box is None: return None

    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
    h, w = img.shape[:2]

    pad_w = int((x2 - x1) * 0.12)
    pad_h = int((y2 - y1) * 0.20)

    crop = img[max(0, y1-pad_h):min(h, y2+pad_h),
               max(0, x1-pad_w):min(w, x2+pad_w)]

    crop_filename = "crop_" + os.path.basename(image_path)
    out = os.path.join(UPLOAD_DIR, crop_filename)
    cv2.imwrite(out, crop)

    return out

def img_detect_chars(img):
    results = img_char_model.predict(img, conf=0.15, imgsz=640, verbose=False)
    detections =[]
    h_img, w_img, _ = img.shape

    for r in results:
        if r.boxes is None: continue
        for (x1,y1,x2,y2), cls, conf in zip(
            r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
        ):
            width, height = x2 - x1, y2 - y1
            if height < h_img * 0.15: continue 
            if width > height * 2.5: continue

            detections.append({
                "char": CHARS[int(cls)], "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                "x1": x1, "w": width, "h": height, "conf": conf
            })
    return detections

def img_dedupe(chars):
    chars = sorted(chars, key=lambda x: x["conf"], reverse=True)
    final =[]
    for c in chars:
        is_duplicate = False
        for f in final:
            dist = np.sqrt((c["cx"] - f["cx"])**2 + (c["cy"] - f["cy"])**2)
            if dist < min(c["w"], f["w"]) * 0.6:
                is_duplicate = True
                break
        if not is_duplicate: final.append(c)
    return final

def img_group_rows(chars):
    rows = []
    chars = sorted(chars, key=lambda x: x["cy"])
    for c in chars:
        placed = False
        for r in rows:
            if abs(np.mean([x["cy"] for x in r]) - c["cy"]) < c["h"] * 0.6:
                r.append(c)
                placed = True
                break
        if not placed: rows.append([c])
    return sorted(rows, key=lambda r: np.mean([x["cy"] for x in r]))

def img_fix_indian_plate(text):
    text = text.upper().replace(" ", "").replace("-", "")
    dict_char_to_num = {'O':'0','Q':'0','D':'0','I':'1','L':'1','Z':'2','A':'4','S':'5','G':'6','B':'8'}
    dict_num_to_char = {'0':'O','1':'I','2':'Z','4':'A','5':'S','6':'G','8':'B'}
    text_list, length = list(text), len(text)

    if length >= 2:
        if text_list[1] == 'H' and text_list[0] in['L', 'V', 'W', 'N']: text_list[0] = 'M'
        for i in[0, 1]:
            if text_list[i] in dict_num_to_char: text_list[i] = dict_num_to_char[text_list[i]]
    if length >= 4:
        for i in [2, 3]:
            if text_list[i] in dict_char_to_num: text_list[i] = dict_char_to_num[text_list[i]]
    if length > 4:
        start_idx = length - (4 if length >= 8 else (length - 6))
        for i in range(start_idx, length):
            if text_list[i] in dict_char_to_num: text_list[i] = dict_char_to_num[text_list[i]]
    return "".join(text_list)

def ocr_from_plate_bgr(plate_bgr):
    """OCR on plate pixels from original frame (BGR). Returns (plate_text, success)."""
    if plate_bgr is None or plate_bgr.size == 0:
        return "Not Detected", False
    img = img_preprocess_plate(plate_bgr.copy())
    chars = img_detect_chars(img)
    if len(chars) < 3:
        return "Not Detected", False
    chars = img_dedupe(chars)
    rows = img_group_rows(chars)
    text = ""
    for r in rows:
        r.sort(key=lambda x: x["x1"])
        text += "".join(c["char"] for c in r)
    return img_fix_indian_plate(text), True


def best_plate_crop_in_vehicle(veh_bgr):
    if veh_bgr is None or veh_bgr.size == 0:
        return None

    results = plate_model.predict(veh_bgr, conf=0.40, verbose=False)

    best_box, max_conf = None, -1.0

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            if float(box.conf) > max_conf:
                max_conf, best_box = float(box.conf), box

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
    h, w = veh_bgr.shape[:2]

    pad_w = int((x2 - x1) * 0.12)
    pad_h = int((y2 - y1) * 0.20)

    crop = veh_bgr[
        max(0, y1 - pad_h):min(h, y2 + pad_h),
        max(0, x1 - pad_w):min(w, x2 + pad_w)
    ]

    return crop if crop.size > 0 else None

def process_image(image_path, filename):
    crop_path = img_detect_and_crop(image_path)
    if not crop_path:
        return "Not Detected", "None"

    img = cv2.imread(crop_path)
    plate_text, _ = ocr_from_plate_bgr(img)
    return plate_text, os.path.basename(crop_path)


# =====================================================================
#                        2. VIDEO PIPELINE LOGIC
# =====================================================================

def vid_is_moving(history):
    if len(history) < 2: return False
    pts = np.array([(h[0], h[1]) for h in history])
    areas = np.array([(h[2] * h[3]) for h in history])
    displacement = np.linalg.norm(pts[-1] - pts[0])
    area_change = abs(areas[-1] - areas[0]) / (areas[0] + 1e-5)
    return displacement > 15 or area_change > 0.10

def vid_enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast = clahe.apply(blur)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    return cv2.filter2D(contrast, -1, kernel)

def vid_enhance_plate_for_ocr(plate):
    plate_resized = cv2.resize(plate, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
    hsv = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    hsv_enhanced = cv2.merge((h, s, clahe.apply(v)))
    plate_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    return cv2.bilateralFilter(plate_enhanced, 5, 50, 50)

def vid_format_indian_plate(text):
    text = text.replace(" ", "").replace(".", "")
    formatted = list(text)
    
    # Fully updated mappings
    letter_to_num = {'O':'0','D':'0','Q':'0','I':'1','J':'1','Z':'7','A':'4','S':'5','G':'6','B':'8','T':'7'}
    num_to_letter = {'0':'O','1':'I','2':'Z','4':'A','5':'S','6':'G','8':'B','7':'Z'}

    for i in range(min(2, len(formatted))):
        if formatted[i] in num_to_letter: formatted[i] = num_to_letter[formatted[i]]

    if len(formatted) >= 2:
        state_code = "".join(formatted[0:2])
        state_corrections = {
            'LH':'MH', 'TL':'TN', 'ML':'MN', 'AL':'AN', 'DL':'DN', 'HL':'HR', 
            '8R':'BR', 'P0':'PU', 'P0':'UP', '0P':'UP'
        }
        if state_code in state_corrections:
            formatted[0] = state_corrections[state_code][0]
            formatted[1] = state_corrections[state_code][1]

    for i in range(max(0, len(formatted)-4), len(formatted)):
        if formatted[i] in letter_to_num: formatted[i] = letter_to_num[formatted[i]]

    if len(formatted) >= 4:
        for i in range(2, min(4, len(formatted))):
            if formatted[i] in letter_to_num: formatted[i] = letter_to_num[formatted[i]]

    return "".join(formatted)

def vid_detect_chars(img):
    # Confidence slightly adjusted to avoid hallucinatory letters
    results = vid_char_model.predict(img, conf=0.20, imgsz=640, verbose=False)
    chars =[]
    
    for r in results:
        if r.boxes is None: continue
        for (x1, y1, x2, y2), cls, conf in zip(
            r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
        ):
            chars.append([x1, y1, x2, y2, CHARS[int(cls)], conf])

    if not chars: return ""

    # Ported 2D-Sorting Algorithm
    chars = sorted(chars, key=lambda x: x[1])
    lines =[]
    for c in chars:
        if not lines:
            lines.append([c])
            continue
        last_line = lines[-1]
        avg_y = sum([char[1] for char in last_line]) / len(last_line)
        char_height = c[3] - c[1]
        
        if abs(c[1] - avg_y) < (char_height * 0.5):
            lines[-1].append(c)
        else:
            lines.append([c])

    sorted_chars =[]
    for line in lines:
        sorted_chars.extend(sorted(line, key=lambda x: x[0]))

    filtered_chars =[]
    for c in sorted_chars:
        if not filtered_chars:
            filtered_chars.append(c)
            continue

        prev = filtered_chars[-1]
        overlap = max(0, min(c[2], prev[2]) - max(c[0], prev[0]))
        width_prev = prev[2] - prev[0]
        width_curr = c[2] - c[0]
        y_overlap = max(0, min(c[3], prev[3]) - max(c[1], prev[1]))

        if y_overlap > 0 and overlap > 0.40 * min(width_prev, width_curr):
            if c[5] > prev[5]:
                filtered_chars[-1] = c
        else:
            filtered_chars.append(c)

    return "".join([c[4] for c in filtered_chars])

def vid_is_valid_plate(text):
    return False if len(text) < 7 or len(set(text)) <= 2 else True

def vid_vote_plate(strings):
    if not strings: return ""
    # THE ULTIMATE FIX: Most frequent FULL string wins. Prevents Frankenstein mixing.
    return Counter(strings).most_common(1)[0][0]

def process_video(video_path):
    print(f"\n[INFO] Starting video processing: {video_path}")
    track_history = defaultdict(list)
    vehicles = {}
    results = vehicle_model.track(source=video_path, stream=True, tracker="bytetrack.yaml", classes=[2,3,5,7], verbose=False)

    frame_count = 0
    for r in results:
        frame_count += 1
        if frame_count % FRAME_SKIP != 0: continue
        if r.boxes is None or r.boxes.id is None: continue

        frame = r.orig_img
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int)

        for box, tid_np in zip(boxes, ids):
            tid = int(tid_np)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w//2, y1 + h//2

            track_history[tid].append((cx, cy, w, h))

            if tid not in vehicles:
                vehicles[tid] = {"plates":[], "best_score":0, "veh_img":None, "plate_img":None}

            veh_crop = frame[y1:y2, x1:x2]
            if veh_crop.size == 0: continue

            enhanced_bgr = cv2.cvtColor(vid_enhance_image(veh_crop), cv2.COLOR_GRAY2BGR)
            p_res = plate_model.predict(enhanced_bgr, conf=0.35, imgsz=320, verbose=False)

            for pr in p_res:
                if pr.boxes is None: continue
                for pb in pr.boxes:
                    px1, py1, px2, py2 = pb.xyxy[0].cpu().numpy().astype(int)
                    plate_crop = veh_crop[py1:py2, px1:px2]
                    if plate_crop.size == 0: continue

                    plate = vid_enhance_plate_for_ocr(plate_crop)
                    raw_text = vid_detect_chars(plate)
                    corrected_text = vid_format_indian_plate(raw_text)

                    if vid_is_valid_plate(corrected_text):
                        vehicles[tid]["plates"].append(corrected_text)

                    score = cv2.Laplacian(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    if score > vehicles[tid]["best_score"]:
                        vehicles[tid]["best_score"] = score
                        vehicles[tid]["veh_img"] = veh_crop.copy()
                        vehicles[tid]["plate_img"] = plate_crop.copy()

    # ================= SAVE VIDEO DATA (With Smart Deduplication) =================
    now = datetime.now()
    saved_vehicle_count = 0
    saved_tids = set()
    
    # 1. Gather all valid tracks and their final OCR strings
    valid_tracks =[]
    for tid, data in vehicles.items():
        if not vid_is_moving(track_history[tid]): continue
        if data["veh_img"] is not None and len(data["plates"]) > 0:
            final_text = vid_vote_plate(data["plates"])
            valid_tracks.append({
                "tid": tid,
                "text": final_text,
                "score": data["best_score"],
                "data": data
            })
            
    # 2. Sort tracks by image clarity (sharpest plates evaluated first)
    valid_tracks = sorted(valid_tracks, key=lambda x: x["score"], reverse=True)
    
    saved_plates_in_video =[]

    with sqlite3.connect(DB_NAME) as c:
        for track in valid_tracks:
            final_text = track["text"]
            
            # 3. Prevent Duplicate Saving using String Similarity (Levenshtein)
            is_duplicate = False
            for saved_plate in saved_plates_in_video:
                # If the string is 75% or more similar, it's the exact same vehicle tracking fragment
                similarity = difflib.SequenceMatcher(None, final_text, saved_plate).ratio()
                if similarity > 0.75:
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                continue # Skip saving this fragmented track
                
            # If unique, proceed to save
            saved_plates_in_video.append(final_text)
            tid = track["tid"]
            data = track["data"]
            
            saved_vehicle_count += 1
            saved_tids.add(tid)

            veh_name = f"vid_veh_{tid}_{int(now.timestamp())}.jpg"
            plt_name = f"vid_plt_{tid}_{int(now.timestamp())}.jpg"
            cv2.imwrite(os.path.join(UPLOAD_DIR, veh_name), data["veh_img"])
            cv2.imwrite(os.path.join(UPLOAD_DIR, plt_name), data["plate_img"])

            c.execute("""
            INSERT INTO vehicle_data (source_type, track_id, vehicle_img, cropped_plate, number_plate, date, time)
            VALUES (?,?,?,?,?,?,?)
            """, ("Video", f"TRK-{tid}", veh_name, plt_name, final_text, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))

    print(f"\n[SUCCESS] Video Processing Complete. Unique 4-wheelers saved: {saved_vehicle_count}\n")

# =====================================================================
#                            VIOLATION LOGIC
# =====================================================================



# =====================================================================
#                             ROUTES
# =====================================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def index():
    with sqlite3.connect(DB_NAME) as c:
        data = c.execute("SELECT * FROM vehicle_data ORDER BY id DESC").fetchall()
    return render_template("index.html", data=data)


@app.route("/upload_image", methods=["POST"])
def upload_image():
    f = request.files.get("file")
    if not f: return redirect(url_for("index"))

    name = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)

    plate_text, crop_name = process_image(path, name)

    now = datetime.now()
    with sqlite3.connect(DB_NAME) as c:
        c.execute("""
        INSERT INTO vehicle_data (source_type, track_id, vehicle_img, cropped_plate, number_plate, date, time)
        VALUES (?,?,?,?,?,?,?)
        """, ("Image", "N/A", name, crop_name, plate_text, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))

    return redirect(url_for("index"))  # redirects to dashboard


@app.route("/upload_video", methods=["POST"])
def upload_video():
    f = request.files.get("file")
    if not f: return redirect(url_for("index"))

    name = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)

    process_video(path)
    return redirect(url_for("index"))  # redirects to dashboard


@app.route("/delete/<int:record_id>")
def delete_record(record_id):
    with sqlite3.connect(DB_NAME) as c:
        c.execute("DELETE FROM vehicle_data WHERE id=?", (record_id,))
    return redirect(url_for("index"))
# ===================== VIOLATION MODULE =====================

def enhance_image(img):
    # Convert to YCrCb (better than HSV for lighting)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # CLAHE on brightness
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    y = clahe.apply(y)

    # Merge back
    enhanced = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)

    # Denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced

def helmet_on_rider(rider, helmet):
    rx1, ry1, rx2, ry2 = rider
    hx1, hy1, hx2, hy2 = helmet

    cx = (hx1 + hx2) // 2
    cy = (hy1 + hy2) // 2

    head_limit = ry1 + (ry2 - ry1) * 0.4

    return rx1 <= cx <= rx2 and ry1 <= cy <= head_limit


def detect_violation_and_ocr(image_path):

    original = cv2.imread(image_path)
    if original is None:
        return None, False, "Not Detected", "None", None, None

    frame = original.copy()
    model = night_model

    # ================= STEP 1: ENHANCE =================
    enhanced = enhance_image(frame)

    # ================= STEP 2: DETECTION =================
    results = model.predict(enhanced, conf=0.4, verbose=False)

    riders, helmets = [], []

    if results and results[0].boxes is not None:
        for b in results[0].boxes:
            cls_id = int(b.cls)
            cls = model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            if cls == "rider":
                riders.append([x1, y1, x2, y2])
            elif cls == "helmet":
                helmets.append([x1, y1, x2, y2])

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    violation = False
    viol_box = None

    # ================= STEP 3: VIOLATION CHECK =================
    for r in riders:
        if not any(helmet_on_rider(r, h) for h in helmets):
            violation = True
            viol_box = r

            cv2.rectangle(frame, (r[0],r[1]), (r[2],r[3]), (0,0,255), 3)
            cv2.putText(frame, "No Helmet", (r[0],r[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            break

    # ================= STEP 4: OCR (STRICT) =================
    plate_text = "Plate Not Visible"
    crop_name = "None"

    if viol_box is not None:
        x1, y1, x2, y2 = viol_box

        # 🔥 ADD PADDING (IMPORTANT)
        pad = 40
        h, w = original.shape[:2]

        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)

        rider_crop = original[ry1:ry2, rx1:rx2]

        # ✅ ONLY inside rider (NO FALLBACK)
        plate_crop = best_plate_crop_in_vehicle(rider_crop)

        if plate_crop is not None and plate_crop.size > 0:

            plate_text, success = ocr_from_plate_bgr(plate_crop)

            if success and len(plate_text) >= 7:
                crop_name = "crop_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
                cv2.imwrite(os.path.join(UPLOAD_DIR, crop_name), plate_crop)
            else:
                plate_text = "Unclear Plate"

    # ================= SAVE IMAGES =================
    out_name = "viol_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, out_name), frame)

    enhanced_name = "enh_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, enhanced_name), enhanced)

    original_name = "orig_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, original_name), original)

    return out_name, violation, plate_text, crop_name, original_name, enhanced_name

def detect_violation_and_ocr(image_path):

    original = cv2.imread(image_path)
    if original is None:
        return None, False, "Not Detected", "None", None, None

    frame = original.copy()
    model = night_model

    # ================= STEP 1: ENHANCE =================
    enhanced = enhance_image(frame)

    # ================= STEP 2: DETECTION =================
    results = model.predict(enhanced, conf=0.4, verbose=False)

    riders, helmets = [], []

    if results and results[0].boxes is not None:
        for b in results[0].boxes:
            cls_id = int(b.cls)
            cls = model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            if cls == "rider":
                riders.append([x1, y1, x2, y2])
            elif cls == "helmet":
                helmets.append([x1, y1, x2, y2])

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    violation = False
    viol_box = None

    # ================= STEP 3: VIOLATION CHECK =================
    for r in riders:
        if not any(helmet_on_rider(r, h) for h in helmets):
            violation = True
            viol_box = r

            cv2.rectangle(frame, (r[0],r[1]), (r[2],r[3]), (0,0,255), 3)
            cv2.putText(frame, "No Helmet", (r[0],r[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            break

    # ================= STEP 4: OCR (STRICT) =================
    plate_text = "Plate Not Visible"
    crop_name = "None"

    if viol_box is not None:
        x1, y1, x2, y2 = viol_box

        # 🔥 ADD PADDING (IMPORTANT)
        pad = 40
        h, w = original.shape[:2]

        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)

        rider_crop = original[ry1:ry2, rx1:rx2]

        # ✅ ONLY inside rider (NO FALLBACK)
        plate_crop = best_plate_crop_in_vehicle(rider_crop)

        if plate_crop is not None and plate_crop.size > 0:

            plate_text, success = ocr_from_plate_bgr(plate_crop)

            if success and len(plate_text) >= 7:
                crop_name = "crop_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
                cv2.imwrite(os.path.join(UPLOAD_DIR, crop_name), plate_crop)
            else:
                plate_text = "Unclear Plate"

    # ================= SAVE IMAGES =================
    out_name = "viol_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, out_name), frame)

    enhanced_name = "enh_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, enhanced_name), enhanced)

    original_name = "orig_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, original_name), original)

    return out_name, violation, plate_text, crop_name, original_name, enhanced_name

def detect_violation_and_ocr(image_path):

    original = cv2.imread(image_path)
    if original is None:
        return None, False, "Not Detected", "None", None, None

    frame = original.copy()
    model = night_model

    # ================= STEP 1: ENHANCE =================
    enhanced = enhance_image(frame)

    # ================= STEP 2: DETECTION =================
    results = model.predict(enhanced, conf=0.4, verbose=False)

    riders, helmets = [], []

    if results and results[0].boxes is not None:
        for b in results[0].boxes:
            cls_id = int(b.cls)
            cls = model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            if cls == "rider":
                riders.append([x1, y1, x2, y2])
            elif cls == "helmet":
                helmets.append([x1, y1, x2, y2])

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    violation = False
    viol_box = None

    # ================= STEP 3: VIOLATION CHECK =================
    for r in riders:
        if not any(helmet_on_rider(r, h) for h in helmets):
            violation = True
            viol_box = r

            cv2.rectangle(frame, (r[0],r[1]), (r[2],r[3]), (0,0,255), 3)
            cv2.putText(frame, "No Helmet", (r[0],r[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            break

    # ================= STEP 4: OCR (STRICT) =================
    plate_text = "Plate Not Visible"
    crop_name = "None"

    if viol_box is not None:
        x1, y1, x2, y2 = viol_box

        # 🔥 ADD PADDING (IMPORTANT)
        pad = 40
        h, w = original.shape[:2]

        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)

        rider_crop = original[ry1:ry2, rx1:rx2]

        # ✅ ONLY inside rider (NO FALLBACK)
        plate_crop = best_plate_crop_in_vehicle(rider_crop)

        if plate_crop is not None and plate_crop.size > 0:

            plate_text, success = ocr_from_plate_bgr(plate_crop)

            if success and len(plate_text) >= 7:
                crop_name = "crop_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
                cv2.imwrite(os.path.join(UPLOAD_DIR, crop_name), plate_crop)
            else:
                plate_text = "Unclear Plate"

    # ================= SAVE IMAGES =================
    out_name = "viol_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, out_name), frame)

    enhanced_name = "enh_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, enhanced_name), enhanced)

    original_name = "orig_" + datetime.now().strftime("%H%M%S_%f") + ".jpg"
    cv2.imwrite(os.path.join(UPLOAD_DIR, original_name), original)

    return out_name, violation, plate_text, crop_name, original_name, enhanced_name
# ===================== ROUTE (REPLACE YOUR OLD ONE) =====================

@app.route("/violation", methods=["GET", "POST"])
def violation():

    if request.method == "POST":
        f = request.files.get("file")

        if not f:
            return redirect(url_for("violation"))

        name = secure_filename(f.filename)
        path = os.path.join(UPLOAD_DIR, name)
        f.save(path)

        result_img, viol, plate_text, crop_name, original_img, enhanced_img = detect_violation_and_ocr(path)

        return render_template(
            "violation.html",
            result_img=result_img,
            violation=viol,
            plate_text=plate_text,
            crop_name=crop_name,
            original_img=original_img,
            enhanced_img=enhanced_img
        )

    # ✅ THIS WAS MISSING
    return render_template("violation.html")
    


    return render_template("violation.html")
if __name__ == "__main__":
    app.run(debug=True)