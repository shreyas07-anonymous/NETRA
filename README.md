# ⬢ NETRA: Multi-Modal Threat Intelligence System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-brightgreen) ![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red) ![Status](https://img.shields.io/badge/Status-Active-success)

**NETRA (Network for Electronic Tracking and Reconnaissance Assessment)** is a prototype military-grade situational awareness system. It leverages **Multi-Modal Sensor Fusion** by combining real-time Object Detection (Computer Vision) with Audio Forensics (Spectral Analysis) to detect threats that a single sensor might miss.

Designed with a futuristic **Tactical HUD (Head-Up Display)**, NETRA provides operators with real-time intelligence on ground intruders, aerial threats (drones), and acoustic anomalies like gunshots or explosions.

---

## 📸 Interface Preview
*(Place a screenshot of your running application here)*

---

## 🚀 Key Features

### 👁️ Visual Intelligence (Computer Vision)
Powered by **YOLOv8 (You Only Look Once)**, the system identifies and tracks objects in real-time:
* **Intruder Detection:** Identifies humans in restricted zones with tactical bounding boxes.
* **Aerial Defense:** Detects Drones/UAVs, birds, and aircraft.
* **Vehicle Identification:** Classifies cars, trucks, buses, and motorcycles.
* **Suspicious Item Detection:** Flags backpacks, handbags, and suitcases left in public spaces.
* **Tactical HUD:** Draws military-style corners, confidence scores, and a central crosshair overlay.

### 🔊 Audio Intelligence (Acoustic Forensics)
Uses **PyAudio** and **NumPy** to analyze raw microphone data using FFT (Fast Fourier Transform) and ZCR (Zero Crossing Rate):
* **Gunshot Detection:** High amplitude + specific frequency range + low roughness.
* **Explosion/Bomb Detection:** High decibel shockwave analysis.
* **Human Distress:** Detects the specific frequency signature of screams.
* **Drone Buzzing:** Identifies the high-frequency/high-roughness hum of UAV motors.

### 💻 System Capabilities
* **Multi-Threading:** Runs audio and video analysis on separate threads for lag-free performance.
* **Fusion Logging:** A real-time scrolling log that color-codes threats based on severity (CRITICAL, DANGER, AERIAL).
* **Visual Alert System:** The screen pulses Red/Orange/Purple based on the specific threat type detected via audio.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **GUI Framework:** Tkinter (Custom styled for Dark Mode)
* **Computer Vision:** OpenCV (`cv2`), Ultralytics YOLOv8
* **Audio Processing:** PyAudio, NumPy (FFT/RMS calculations)
* **Image Processing:** Pillow (PIL)

---

## ⚙️ Installation & Setup

### Prerequisites
Ensure you have Python installed. You will also need a working webcam and microphone.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/NETRA-Defense-System.git](https://github.com/your-username/NETRA-Defense-System.git)
    cd NETRA-Defense-System
    ```

2.  **Install Dependencies**
    Create a `requirements.txt` file or install directly:
    ```bash
    pip install opencv-python ultralytics pyaudio numpy pillow
    ```
    *(Note: PyAudio can sometimes be tricky to install on Windows. If `pip install pyaudio` fails, download the appropriate .whl file for your Python version).*

3.  **Run the System**
    ```bash
    python main.py
    ```

---

## 🎛️ Configuration

You can tweak the sensitivity at the top of the `main.py` file:
```python
MIC_INDEX = 1           # Change this if your microphone isn't detected
CONFIDENCE_THRESHOLD = 0.5  # Adjust how sure the AI needs to be to flag an object
