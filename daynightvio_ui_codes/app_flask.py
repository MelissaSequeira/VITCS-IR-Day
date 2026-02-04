"""
Flask UI for Traffic Violation Detection.
Same functionality as Streamlit app.py (YOLO) and apiapp.py (Roboflow).
"""
import os
import tempfile
import uuid
import base64
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash

FRAME_SKIP = 5  # Roboflow video; matches detection_roboflow


class YOLOUnavailableError(Exception):
    """Raised when YOLO/PyTorch cannot be loaded (e.g. Windows DLL error)."""
    pass


_here = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(_here, "templates"),
    static_folder=os.path.join(_here, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.config["OUTPUT_FOLDER"] = os.path.join(_here, "output")
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)


def get_processor(model_name):
    if model_name == "roboflow":
        from detection_roboflow import process_frame as process_frame_roboflow
        return process_frame_roboflow, None
    try:
        from detection_yolo import process_frame as process_frame_yolo
        return process_frame_yolo, "conf"
    except (OSError, ImportError) as e:
        raise YOLOUnavailableError(
            "YOLO is unavailable (PyTorch failed to load). Use Roboflow."
        ) from e


def process_image_file(file_bytes, model_name, conf=0.25):
    arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, None
    proc, _ = get_processor(model_name)
    if model_name == "yolo":
        out, helmetless, overloaded = proc(frame, conf=conf)
    else:
        out, helmetless, overloaded = proc(frame)
    _, jpeg = cv2.imencode(".jpg", out)
    return jpeg.tobytes(), helmetless, overloaded


def process_video_file(file_bytes, model_name, conf=0.25):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(file_bytes)
    tfile.close()
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        os.unlink(tfile.name)
        return None, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_name = os.path.join(app.config["OUTPUT_FOLDER"], f"{uuid.uuid4().hex}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_name, fourcc, fps, (w, h))

    proc, _ = get_processor(model_name)
    fid = 0
    total_helmetless, total_overloaded = 0, 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fid += 1
            if model_name == "roboflow" and fid % FRAME_SKIP != 0:
                continue
            if model_name == "yolo":
                out, hc, oc = proc(frame.copy(), conf=conf)
            else:
                out, hc, oc = proc(frame.copy())
            writer.write(out)
            total_helmetless = max(total_helmetless, hc)
            total_overloaded = max(total_overloaded, oc)
    finally:
        cap.release()
        writer.release()
        os.unlink(tfile.name)

    return os.path.basename(out_name), total_helmetless, total_overloaded


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-image", methods=["POST"])
def process_image():
    model_name = request.form.get("model", "yolo")
    conf = float(request.form.get("confidence", 0.25))
    if "file" not in request.files:
        return redirect(url_for("index"))
    f = request.files["file"]
    if not f.filename:
        return redirect(url_for("index"))
    try:
        data = f.read()
        img_b64, helmetless, overloaded = process_image_file(data, model_name, conf)
    except YOLOUnavailableError:
        flash("YOLO is unavailable (PyTorch failed to load). Please use Roboflow.", "error")
        return redirect(url_for("index"))
    if img_b64 is None:
        return redirect(url_for("index"))
    img_b64 = base64.b64encode(img_b64).decode("utf-8")
    return render_template(
        "result_image.html",
        image_b64=img_b64,
        helmetless=helmetless,
        overloaded=overloaded,
        model_name=model_name,
    )


@app.route("/process-video", methods=["POST"])
def process_video():
    model_name = request.form.get("model", "yolo")
    conf = float(request.form.get("confidence", 0.25))
    if "file" not in request.files:
        return redirect(url_for("index"))
    f = request.files["file"]
    if not f.filename:
        return redirect(url_for("index"))
    try:
        data = f.read()
        out_filename, helmetless, overloaded = process_video_file(data, model_name, conf)
    except YOLOUnavailableError:
        flash("YOLO is unavailable (PyTorch failed to load). Please use Roboflow.", "error")
        return redirect(url_for("index"))
    if out_filename is None:
        return redirect(url_for("index"))
    return render_template(
        "result_video.html",
        video_filename=out_filename,
        helmetless=helmetless,
        overloaded=overloaded,
        model_name=model_name,
    )


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


if __name__ == "__main__":
    try:
        from detection_roboflow import load_model as load_robo
        load_robo()
        print("Roboflow model loaded.")
    except Exception as e:
        print("Roboflow load skipped:", e)
    print("Open in browser: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
