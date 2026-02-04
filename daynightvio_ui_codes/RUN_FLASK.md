# How to Run the Flask UI (Same Functionality as Before)

This project has a **Flask UI** that does the same thing as the old Streamlit apps (`app.py` = YOLO, `apiapp.py` = Roboflow). No functionality was changed—only the UI layer was switched from Streamlit to Flask.

---

## 1. Run the Flask app (like before)

From a terminal:

```bash
cd E:\iiser\VITCS-IR-Day\daynightvio_ui_codes
pip install -r requirements-flask.txt
python app_flask.py
```

Then open in your browser:

**http://127.0.0.1:5000**

You should see:
- Model: **YOLO** or **Roboflow**
- Confidence slider (for YOLO)
- Input type: **Image** or **Video**
- File upload and **Process** button

---

## 2. Issues you might see (and what to do)

### YOLO / PyTorch DLL error on Windows

**Symptom:** When you choose **YOLO** and process, you get an error like:

`OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed ... c10.dll`

**Cause:** PyTorch (used by YOLO) fails to load its DLLs on your Windows setup.

**What the app does:** The Flask app catches this and sends you back to the home page with a message: *"YOLO is unavailable (PyTorch failed to load). Please use Roboflow."*

**Ways to fix YOLO (optional):**

1. **Use Roboflow only**  
   Keep using the app with **Model = Roboflow**. No code change needed; everything works the same for Roboflow.

2. **Fix PyTorch so YOLO works**  
   - Install/repair **Microsoft Visual C++ Redistributable** (latest, x64).  
   - In the same environment:
     ```bash
     pip uninstall torch
     pip install torch
     ```
   - If it still fails, try a different terminal (e.g. Command Prompt instead of PowerShell) or a fresh Python environment.

### Roboflow API

Roboflow needs network access and a valid API key (already in `detection_roboflow.py`). If the key or project changes, update that file; the Flask UI code does not need to change.

---

## 3. How to test the project (run the model)

1. **Start the app**  
   `python app_flask.py` and open http://127.0.0.1:5000.

2. **Test with Roboflow (most reliable if YOLO fails)**  
   - Model: **Roboflow (Strict)**  
   - Input type: **Image**  
   - Upload a `.jpg`/`.png` with riders/bikes  
   - Click **Process**  
   - You should get a result page with the annotated image and counts (helmetless riders, overloaded bikes).

3. **Test video**  
   - Same model (e.g. Roboflow)  
   - Input type: **Video**  
   - Upload `.mp4`/`.avi`/`.mov`  
   - Click **Process**  
   - After processing, you get a result page with a playable video and counts.

4. **Test YOLO (if PyTorch works)**  
   - Model: **YOLO (Ultralytics)**  
   - Set confidence (e.g. 0.25)  
   - Image or Video, then **Process**  
   - If you see the DLL error, use Roboflow as above; the Flask UI and logic are unchanged.

---

## 4. What’s the same as before (no code functionality change)

- **YOLO path:** Same model (`newmodel_yolo/best.pt` or fallback path), confidence, helmetless/overload logic.  
- **Roboflow path:** Same API, project, version, frame skip, and strict helmet/overload logic.  
- **Image:** Upload → process → show image + counts.  
- **Video:** Upload → process (with frame skip for Roboflow) → output video + counts.  
- **Detection logic:** Lives in `detection_yolo.py` and `detection_roboflow.py`; Flask only handles HTTP and file I/O.

The UI is now HTML/Flask instead of Streamlit; behavior and models are the same.
