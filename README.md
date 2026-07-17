# 🚦 Vehicle Identification and Traffic Compliance System (VITCS)

An AI-powered Vehicle Identification and Traffic Compliance System (VITCS) developed using **YOLOv8**, **Flask**, **OpenCV**, and **PyTorch** for automatic **Automatic Number Plate Recognition (ANPR)** and **traffic violation detection** from images and videos.

The system supports **daytime and nighttime (IR camera)** images and can detect traffic violations such as **helmetless riding**, identify the corresponding vehicle, recognize its number plate, and store the results in a database.

---

# 📌 Features

- 🚗 Vehicle Detection
- 🏍️ Helmet Violation Detection
- 🔢 Automatic Number Plate Recognition (ANPR)
- 🌙 Night-Time / IR Camera Support
- 🎥 Video Processing with Vehicle Tracking
- 📸 Image Processing
- 🧠 Custom YOLOv8 Models
- 🗳 OCR Voting for Improved Accuracy
- 🗄 SQLite Database Storage
- 🌐 Flask Web Application
- 📊 Dashboard to View Detection Results

---

# 🛠 Tech Stack

## Backend

- Python 3.10
- Flask

## AI / Deep Learning

- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy

## Database

- SQLite

## Frontend

- HTML
- CSS
- JavaScript

---

# 📂 Project Structure

```text
VITCS/
│
├── models/
│   ├── plate.pt
│   ├── ocr.pt
│   ├── best.pt
│
├── static/
│   ├── uploads/
│   ├── css/
│   └── js/
│
├── templates/
│   ├── home.html
│   ├── index.html
│   └── violation.html
│
├── database.db
├── app.py
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## Step 1 : Clone the Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

or download the ZIP file from GitHub and extract it.

---

# 📥 Step 2 : Download Datasets

Download the datasets used for training:

- License Plate Detection Dataset
- License Plate OCR Dataset
- Helmet / Rider Detection Dataset

You may use datasets from:

- Roboflow
- Kaggle
- Custom Dataset

---

# 🤖 Step 3 : Train the Models

Open the training notebooks located inside:

```
Violation_training_colab/
```

using **Google Colab**.

Train the following models:

- Plate Detection Model
- OCR Character Detection Model
- Helmet / Rider Detection Model

After training, download the generated weights.

Example:

```
plate.pt
ocr.pt
best.pt
```

Copy all these files into the project's **models/** folder.

---

# 🐍 Step 4 : Create Python Environment

Install Python **3.10**

Create Virtual Environment

```bash
python -m venv yolovenv
```

Activate Environment

### Windows

```bash
yolovenv\Scripts\activate
```

### Linux / Mac

```bash
source yolovenv/bin/activate
```

---

# 📦 Step 5 : Install Dependencies

Install all required libraries.

```bash
pip install -r requirements.txt
```

If requirements.txt is unavailable, install the major libraries manually:

```bash
pip install flask ultralytics torch torchvision torchaudio opencv-python numpy matplotlib pandas scipy pillow tqdm deep-sort-realtime cvzone
```

---

# ⚙ Step 6 : Configure Model Paths

Open **app.py**

Update the paths according to your system.

Example

```python
vehicle_model = YOLO("yolov8n.pt")

plate_model = YOLO("models/plate.pt")

img_char_model = YOLO("models/ocr.pt")

vid_char_model = YOLO("models/ocr.pt")

night_model = YOLO("models/best.pt")
```

---

# ▶ Step 7 : Run the Application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

# 📸 Image Processing Pipeline

1. Upload image

2. Detect Vehicle

3. Detect License Plate

4. Crop Plate

5. Enhance Plate Image

6. Perform OCR

7. Format Indian Number Plate

8. Store Result in Database

---

# 🎥 Video Processing Pipeline

1. Upload video

2. Detect Vehicles

3. Track Vehicles using ByteTrack

4. Detect Number Plate

5. OCR Character Detection

6. Voting Across Multiple Frames

7. Remove Duplicate Vehicles

8. Save Results in Database

---

# 🚨 Traffic Violation Detection Pipeline

1. Upload Image

2. Enhance Image

3. Detect Riders and Helmets

4. Detect Helmet Violation

5. Crop Rider Region

6. Detect License Plate

7. OCR Recognition

8. Display Final Result

---

# 🌙 Night-Time Processing

The project supports ANPR using infrared/night vision cameras.

Image enhancement techniques include:

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- YCrCb Color Space
- Gaussian Blur
- Bilateral Filtering
- Adaptive Thresholding
- Image Sharpening
- Noise Removal

---

# 🗄 Database

SQLite is used for storing:

- Vehicle Image
- Cropped Plate
- Number Plate
- Source Type
- Track ID
- Date
- Time

---

# 📊 Dashboard

The dashboard allows users to:

- Upload Images
- Upload Videos
- Detect Violations
- View Detection History
- Delete Records

---

# 📚 Major Libraries Used

- Flask
- OpenCV
- Ultralytics
- PyTorch
- TorchVision
- TorchAudio
- NumPy
- Pandas
- SciPy
- Pillow
- Matplotlib
- Deep Sort Realtime
- CVZone
- SQLite3
- Werkzeug
- TQDM

---

# 📈 Future Improvements

- Triple Riding Detection
- Signal Jump Detection
- Wrong Side Driving Detection
- Speed Violation Detection
- Automatic Fine Generation
- Vehicle Owner Identification
- Edge Deployment on Raspberry Pi
- Jetson Nano Deployment
- Live CCTV Integration
- REST API Support

---

# 👩‍💻 Authors

- Melissa Sequeira
- Abhinn Amrit
- Siddharth Rane
- Anushka Pawar

---

# 📄 License

This project was developed for academic and research purposes.

---

# ⭐ Acknowledgements

- Ultralytics YOLOv8
- OpenCV
- PyTorch
- Roboflow
- Flask
- Google Colab
- ByteTrack
