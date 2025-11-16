# 🧍‍♂️ Real-Time People Detection System

A Streamlit-based real-time people detection application using **YOLOv8**, **OpenCV**, and **PyTorch (CPU build)**.  
The system detects people from either a **live camera feed** or an **uploaded video file**, draws bounding boxes, and displays a **live counting metric**.

This project is optimized for **Windows**, fully compatible with **Ubuntu**, and built for consistent, stable performance on **CPU-only systems**.

---

## 🚀 Features

- **Dual Input Modes**
  - Live Camera feed
  - Uploaded Video File (mp4, avi, mov, mkv)
- **Real-Time Detection**
  - YOLOv8 Nano (`yolov8n.pt`) for fast performance
- **Bounding Boxes + Labels**
  - Class-specific (Person only)
  - Includes confidence scores
- **Live People Count**
  - Auto-updating metric display
- **15+ FPS Target**
  - Frame pacing logic for smooth visualization
- **Streamlit UI**
  - Clean, responsive interface
- **Automatic Resource Cleanup**
  - Camera release
  - Temporary file deletion
- **Cross-Platform Support**
  - Works on both Windows and Ubuntu

---

## 📦 Prerequisites

### **Required**
- Python **3.10.x** (recommended: 3.10.11)
- A webcam (optional, for camera mode)
- Supported video file (mp4/avi/mov/mkv)

### **Windows Users**
PyTorch needs Microsoft Visual C++ Redistributable:

**Download:**  
https://aka.ms/vs/17/release/vc_redist.x64.exe

### **Ubuntu Users**
Install required system libraries:

```bash
sudo apt-get update
sudo apt-get install python3-opencv libgl1-mesa-glx libglib2.0-0
````

---

## 📁 Project Structure

```
real-time-people-detection/
├── app.py
├── requirements.txt
├── README.md
└── yolov8n.pt
```

---

## 🔧 Installation

### 1️⃣ Clone or Download the Project

```bash
git clone https://github.com/HelloYug/real-time-people-detection.git
cd real-time-people-detection
```

---

### 2️⃣ (Optional but recommended) Create a Virtual Environment

#### **Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

#### **Ubuntu**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

* Streamlit
* YOLOv8 (Ultralytics)
* PyTorch 2.2.2+cpu
* OpenCV
* NumPy 1.26.4 (required for PyTorch compatibility)


If you don’t have it, download manually:
[https://github.com/ultralytics/assets/releases](https://github.com/ultralytics/assets/releases)

Streamlit will also auto-download on first run if missing.

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Your browser will automatically open to:

```
http://localhost:8501
```

---

## 🎥 How to Use

### **1. Choose Input Mode**

* **Camera** → Uses your webcam
* **Video File** → Upload a video

### **2. Processing Begins Automatically**

Each frame is processed by YOLOv8, which:

* Detects **only people** (COCO class ID = 0)
* Draws a green bounding box
* Displays a label with confidence

### **3. Live Updates**

* Bounding boxes update in real-time
* People count metric updates only when changed

### **4. Stop Detection**

Click the **"Stop Detection"** button anytime.

---

## 🛠 Troubleshooting

### ❌ Camera Not Accessible

**Fixes:**

* Close apps using the camera (Zoom, Teams, OBS)
* Check OS camera permissions
* Ubuntu:

  ```bash
  sudo usermod -a -G video $USER
  ```

### ❌ Video Cannot Be Opened

* Ensure format is mp4/avi/mov/mkv
* Try converting with HandBrake or VLC
* File may be corrupted

### ❌ PyTorch DLL Initialization Error (Windows)

Install Visual C++ Runtime:
[https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### ❌ NumPy Version Error

If you see:

```
A module compiled using NumPy 1.x cannot run in NumPy 2.x
```

Run:

```bash
pip install numpy==1.26.4 --force-reinstall
```

### ❌ Streamlit Port Already Running

Use a different port:

```bash
streamlit run app.py --server.port 8502
```

---

## 📦 Requirements (from `requirements.txt`)

```txt
streamlit==1.39.0
ultralytics==8.3.32

torch==2.2.2+cpu
torchvision==0.17.2+cpu
torchaudio==2.2.2+cpu
--extra-index-url https://download.pytorch.org/whl/cpu

opencv-python==4.10.0.84
numpy==1.26.4
pillow==10.4.0
jinja2==3.1.4
sympy==1.12
typing_extensions==4.9.0
filelock==3.13.1
fsspec==2024.10.0
networkx==3.2.1
```

---

## 🧠 Technical Details

* **Model Used:** YOLOv8 Nano (`yolov8n.pt`)
* **Detected Class:** Person (class ID `0`)
* **Bounding Box Color:** Green `(0, 255, 0)`
* **Target FPS:** ~15
* **Display Framework:** Streamlit with `width="stretch"`

---


## ⚖️ License

This project is distributed under the **MIT License**.
You are free to use, modify, and share it with appropriate credit.

---

## 👨‍💻 Author

**Yug Agarwal**

* 📧 [yugagarwal704@gmail.com](mailto:yugagarwal704@gmail.com)
* 🔗 GitHub – [@HelloYug](https://github.com/HelloYug)