# GuardianEye | Smart Driver Drowsiness & Fatigue Monitoring System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Computer Vision](https://img.shields.io/badge/Vision-OpenCV%20%7C%20MediaPipe-green.svg)](https://mediapipe.dev/)
[![ML Model](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20(98.28%25)-brightgreen.svg)](https://scikit-learn.org/)
[![Frontend](https://img.shields.io/badge/Dashboard-Streamlit%20Cyber%20Neon-red.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Database-SQLite%20(Zero--Config)-orange.svg)](https://sqlite.org/)

An intelligent real-time **Advanced Driver Assistance System (ADAS)** and **Fatigue Monitoring Telemetry Engine** built with Python, OpenCV, MediaPipe 3D Face Mesh, and Scikit-Learn. 

By analyzing **Eye Aspect Ratio (EAR)**, **Mouth Opening Ratio (MOR)**, **Blink Frequency**, and **PERCLOS**, the system classifies driver fatigue states (`ACTIVE`, `DROWSY`, `YAWNING`) and triggers automated audio-visual emergency alarms and incident audit logging.

---

## Key Features

- **3D Facial Landmark Tracking:** Sub-millisecond tracking of 468 3D facial landmarks using MediaPipe.
- **Eye Blink & PERCLOS Analysis:** Continuously calculates EAR and Percentage of Eye Closure (PERCLOS) over rolling temporal windows.
- **Yawn Detection Engine:** Mathematical Mouth Opening Ratio (MOR) computation to detect early driver fatigue.
- **Machine Learning Inference:** Random Forest Classifier (100 Decision Trees) achieving **98.28% Accuracy** on multi-class fatigue states.
- **Cyber Neon Cockpit Web App:** Modern, high-contrast, emoji-free Streamlit digital cockpit interface.
- **Zero-Latency Audio Siren:** Multi-tone emergency alarm buzzer powered by Pygame audio hardware buffer.
- **Embedded Audit Logging:** Automated, zero-configuration local SQLite event logging for driver safety analytics.

---

## System Architecture

```text
                     [ Live Webcam Stream ]
                               │
                   [ MediaPipe 3D Face Mesh ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     [ Eye Landmarks ]                     [ Mouth Landmarks ]
            │                                     │
    Eye Aspect Ratio (EAR)                Mouth Opening Ratio (MOR)
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
           Feature Vector: [avg_ear, l_ear, r_ear, mor]
                               │
                  [ Random Forest Classifier ]
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
     [ ACTIVE ]           [ YAWNING ]           [ DROWSY ]
     Green HUD           Fatigue Amber         Red Flashing Alert
                                                      │
                                                      ├──> Emergency Audio Siren
                                                      └──> SQLite Event Logging
```

---

## Project Structure

```text
--EyeBlink--/
│
├── assets/
│   └── alarm.wav                # Emergency loud audio siren sound
│
├── dataset/
│   └── kaggle_features.csv      # 2,898 Extracted Kaggle benchmark samples
│
├── db/
│   ├── db_config.py             # SQLite database connector & logger
│   └── drowsiness_logs.db       # Embedded SQLite incident database
│
├── models/
│   └── rfc_drowsiness_model.pkl # Trained Random Forest model (98.28% Acc)
│
├── reports/
│   ├── confusion_matrix.png     # Figure 2: Random Forest Confusion Matrix
│   ├── ear_trend_curve.png      # Figure 1: Real-Time EAR Drop Curve
│   └── detection_logs_export.csv# Audit log CSV export
│
├── 1_collect_data.py            # Guided 3-in-1 custom webcam dataset collector
├── 2_train_model.py             # Random Forest model trainer script
├── 3_realtime_detect.py         # Standalone OOP OpenCV real-time detector
├── 4_evaluate_and_plots.py      # Evaluation artifacts & visualization generator
├── app.py                       # Streamlit Cyber Neon ADAS web dashboard
├── view_logs.py                 # Terminal database logs & analytics viewer
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/--EyeBlink--.git
cd --EyeBlink--
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Streamlit Cyber Cockpit Web App (Recommended)
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser. Toggle **ENGAGE LIVE MONITORING** to begin real-time driver tracking.

### Option 2: Standalone Native OpenCV Window
```bash
python 3_realtime_detect.py
```
Runs the high-speed OOP detector with native video overlays and alarm buzzer. Press `q` to exit.

### Option 3: View Database Incident History
```bash
python view_logs.py
```
Displays incident audit logs, summary statistics, and exports data to CSV.

---

## Model Evaluation Metrics

| State Class | Precision | Recall | F1-Score | Samples |
| :--- | :---: | :---: | :---: | :---: |
| **ACTIVE (0)** | 98.0% | 100.0% | 0.99 | 284 |
| **DROWSY (1)** | 99.0% | 100.0% | 0.99 | 151 |
| **YAWNING (2)** | 99.0% | 94.0% | 0.96 | 145 |
| **Overall Model** | **98.28%** | **98.28%** | **0.98** | **580** |

---

## Academic Reference
Developed for **Python for Data Science (PDS 2026-27)** curriculum:
- **Module 1 (Python Core & OOP):** Continuous video loop, 3D landmark indexing, class-based modular detector pipeline (`VideoHandler`, `FeatureExtractor`, `Classifier`).
- **Module 2 (DS Life Cycle & Stats):** Rolling window variance, EAR baseline central tendency analysis, PERCLOS formulation.
- **Module 3 (Libraries & DB):** NumPy vector distances, Pandas DataFrames, Scikit-Learn Random Forest, SQLite relational event logging.

---

## License
MIT License. Open source for academic and research purposes.
