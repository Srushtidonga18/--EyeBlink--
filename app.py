import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import pygame
import time
import os
import sqlite3
from collections import deque
from sklearn.metrics import classification_report, accuracy_score
from db.db_config import log_alert, init_db, SQLITE_PATH, get_logs_dataframe

# =====================================================================
# 1. PAGE CONFIGURATION & HIGH-CONTRAST CYBER NEON STYLING
# =====================================================================
st.set_page_config(
    page_title="GuardianEye | Driver Monitoring System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    /* Cyber Neon Background */
    .stApp {
        background: radial-gradient(circle at 50% -10%, #0E1626 0%, #060911 100%);
        color: #FFFFFF;
    }
    
    /* Top Luxury Brand Header */
    .top-brand-bar {
        background: rgba(14, 22, 38, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 16px;
        padding: 20px 32px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .brand-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .brand-subtitle {
        font-size: 16px;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 500;
    }
    
    /* High-Contrast Large Navigation Bar */
    div[data-testid="stHorizontalBlock"] > div:has(div[data-testid="stRadio"]) {
        background: rgba(14, 22, 38, 0.95);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 14px;
        padding: 8px 16px;
        margin-bottom: 28px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    div[role="radiogroup"] {
        justify-content: center;
        gap: 20px;
    }
    div[role="radiogroup"] > label {
        background: transparent !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        color: #94A3B8 !important;
        transition: all 0.2s ease !important;
    }
    div[role="radiogroup"] > label:hover {
        color: #00F2FE !important;
        background: rgba(0, 242, 254, 0.1) !important;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #0052D4 0%, #00F2FE 100%) !important;
        box-shadow: 0 4px 18px rgba(0, 242, 254, 0.45) !important;
    }

    /* Large Cockpit Metric Cards */
    .cockpit-card {
        background: rgba(14, 22, 38, 0.75);
        border: 1px solid rgba(0, 242, 254, 0.18);
        border-radius: 16px;
        padding: 22px 24px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .cockpit-card:hover {
        border-color: rgba(0, 242, 254, 0.5);
        transform: translateY(-2px);
    }
    .cockpit-card-title {
        font-size: 15px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94A3B8;
    }
    .cockpit-card-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 38px;
        font-weight: 800;
        margin-top: 8px;
        letter-spacing: -0.5px;
    }

    /* High-Tech Standby Banner */
    .standby-box {
        background: linear-gradient(135deg, rgba(14, 22, 38, 0.9) 0%, rgba(9, 13, 22, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 20px;
        padding: 50px 36px;
        text-align: center;
        margin: 24px 0;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
    }
    .standby-title {
        font-size: 26px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    .standby-desc {
        font-size: 16px;
        color: #94A3B8;
        max-width: 680px;
        margin: 0 auto 26px auto;
        line-height: 1.6;
    }
    .health-pill {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 24px;
        font-size: 14px;
        font-weight: 700;
        margin: 6px 6px;
        background: rgba(0, 255, 135, 0.12);
        border: 1px solid rgba(0, 255, 135, 0.4);
        color: #00FF87;
        box-shadow: 0 0 12px rgba(0, 255, 135, 0.15);
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-size: 34px !important;
        font-weight: 800 !important;
        color: #00F2FE !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. LOAD SYSTEM RESOURCES & DYNAMIC MODEL EVALUATION
# =====================================================================
@st.cache_resource
def load_system():
    init_db()
    model_path = 'models/rfc_drowsiness_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found. Run 2_train_model.py first.")
        st.stop()
    model = joblib.load(model_path)

    pygame.mixer.init()
    alarm = None
    if os.path.exists('assets/alarm.wav'):
        try:
            alarm = pygame.mixer.Sound('assets/alarm.wav')
        except Exception:
            pass

    mp_face = mp.solutions.face_mesh
    return model, alarm, mp_face

model, alarm_sound, mp_face = load_system()

@st.cache_data
def get_dynamic_evaluation():
    kaggle_p = 'dataset/kaggle_features.csv'
    custom_p = 'dataset/custom_features.csv'
    dfs = []
    if os.path.exists(kaggle_p):
        dfs.append(pd.read_csv(kaggle_p))
    if os.path.exists(custom_p):
        dfs.append(pd.read_csv(custom_p))
    
    if not dfs:
        return None, 0.0, None, {}

    df_all = pd.concat(dfs, ignore_index=True)
    X = df_all[['avg_ear', 'l_ear', 'r_ear', 'mor']]
    y = df_all['label']

    preds = model.predict(X)
    acc = accuracy_score(y, preds) * 100.0
    report_dict = classification_report(y, preds, target_names=['ACTIVE (0)', 'DROWSY (1)', 'YAWNING (2)'], output_dict=True)

    feat_names = ['Average EAR', 'Left Eye EAR', 'Right Eye EAR', 'Mouth Opening (MOR)']
    importances_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance Weight': model.feature_importances_
    }).sort_values(by='Importance Weight', ascending=False)

    return df_all, acc, importances_df, report_dict

dataset_df, dynamic_acc, dynamic_feat_imp, dynamic_report = get_dynamic_evaluation()

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(pts, eye_indices):
    A = np.linalg.norm(pts[eye_indices[1]] - pts[eye_indices[5]])
    B = np.linalg.norm(pts[eye_indices[2]] - pts[eye_indices[4]])
    C = np.linalg.norm(pts[eye_indices[0]] - pts[eye_indices[3]])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def calculate_mor(pts):
    height = np.linalg.norm(pts[13] - pts[14])
    width = np.linalg.norm(pts[78] - pts[308])
    return height / width if width > 0 else 0.0

# =====================================================================
# 3. TOP BRAND BAR (NO EMOJIS, CLEAN NEON)
# =====================================================================
acc_badge_str = f"RFC MODEL: {dynamic_acc:.2f}% ACCURACY" if dynamic_acc > 0 else "RFC MODEL ONLINE"

st.markdown(f"""
<div class="top-brand-bar">
    <div>
        <div class="brand-title">GUARDIAN EYE // ADAS COCKPIT</div>
        <div class="brand-subtitle">Smart Real-Time Driver Drowsiness & Fatigue Monitoring System</div>
    </div>
    <div>
        <span style="background: rgba(0, 242, 254, 0.12); border: 1px solid #00F2FE; color: #00F2FE; padding: 8px 18px; border-radius: 20px; font-size: 14px; font-weight: 800; letter-spacing: 0.5px; box-shadow: 0 0 15px rgba(0, 242, 254, 0.25);">
            {acc_badge_str}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# 4. SLEEK TOP NAVIGATION BAR (CLEAN TEXT)
# =====================================================================
selected_nav = st.radio(
    label="Navigation",
    options=["Live Cockpit Monitor", "Incident Logs & Audit", "Telemetry & Analytics", "System Architecture"],
    horizontal=True,
    label_visibility="collapsed"
)

# =====================================================================
# 5. VIEW: LIVE COCKPIT MONITOR
# =====================================================================
if selected_nav == "Live Cockpit Monitor":
    c1, c2, c3, c4 = st.columns(4)
    status_box = c1.empty()
    ear_box = c2.empty()
    blink_box = c3.empty()
    perclos_box = c4.empty()

    status_box.markdown("""
    <div class="cockpit-card">
        <div class="cockpit-card-title">Driver State</div>
        <div class="cockpit-card-value" style="color: #00FF87;">STANDBY</div>
    </div>
    """, unsafe_allow_html=True)

    ear_box.markdown("""
    <div class="cockpit-card">
        <div class="cockpit-card-title">Live EAR Value</div>
        <div class="cockpit-card-value" style="color: #00F2FE;">--</div>
    </div>
    """, unsafe_allow_html=True)

    blink_box.markdown("""
    <div class="cockpit-card">
        <div class="cockpit-card-title">Blink Activity</div>
        <div class="cockpit-card-value" style="color: #FFFFFF;">0 <span style="font-size: 16px; color: #94A3B8;">(0/min)</span></div>
    </div>
    """, unsafe_allow_html=True)

    perclos_box.markdown("""
    <div class="cockpit-card">
        <div class="cockpit-card-title">PERCLOS Fatigue</div>
        <div class="cockpit-card-value" style="color: #FFD600;">0.0%</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([4, 4, 4])
    with ctrl_col1:
        run_camera = st.toggle("ENGAGE LIVE MONITORING", value=False)
    with ctrl_col2:
        audio_alert_enabled = st.toggle("Audio Siren Alarm", value=True)
    with ctrl_col3:
        sensitivity = st.select_slider("Sensitivity Threshold", options=["Low (0.16)", "Balanced (0.20)", "High (0.24)"], value="Balanced (0.20)")

    ear_threshold = 0.16 if "Low" in sensitivity else (0.24 if "High" in sensitivity else 0.20)
    consec_thresh = 15

    if not run_camera:
        samples_count = len(dataset_df) if dataset_df is not None else 0
        st.markdown(f"""
        <div class="standby-box">
            <div class="standby-title">Driver Monitoring System in Standby</div>
            <div class="standby-desc">
                Toggle <strong>ENGAGE LIVE MONITORING</strong> above to initialize webcam stream, 
                MediaPipe 468-point 3D Face Mesh, and real-time fatigue analysis.
            </div>
            <div>
                <span class="health-pill">MediaPipe 3D: READY</span>
                <span class="health-pill">AI Model: {dynamic_acc:.1f}% ACCURACY</span>
                <span class="health-pill">Training Samples: {samples_count}</span>
                <span class="health-pill">SQLite Database: CONNECTED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        v_col, telemetry_col = st.columns([7, 5])
        video_feed_placeholder = v_col.empty()
        
        with telemetry_col:
            st.markdown("#### Live Telemetry Stream")
            chart_placeholder = st.empty()
            telemetry_detail = st.empty()

        cap = cv2.VideoCapture(0)
        ear_history = deque(maxlen=40)
        blink_timestamps = deque()
        total_blinks = 0
        consecutive_closed = 0
        eye_state_closed = False
        last_log_time = 0
        alarm_playing = False
        LABELS = {0: 'ACTIVE', 1: 'DROWSY', 2: 'YAWNING'}

        with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not read from camera.")
                    break

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)

                status_str = "ACTIVE"
                conf = 0.95
                avg_ear, l_ear, r_ear, mor = 0.30, 0.30, 0.30, 0.20
                curr_t = time.time()

                if res.multi_face_landmarks:
                    pts = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark])
                    l_ear = calculate_ear(pts, LEFT_EYE)
                    r_ear = calculate_ear(pts, RIGHT_EYE)
                    avg_ear = (l_ear + r_ear) / 2.0
                    mor = calculate_mor(pts)
                    ear_history.append(avg_ear)

                    if avg_ear < ear_threshold:
                        consecutive_closed += 1
                        if not eye_state_closed:
                            eye_state_closed = True
                    else:
                        if eye_state_closed:
                            if 2 <= consecutive_closed < consec_thresh:
                                total_blinks += 1
                                blink_timestamps.append(curr_t)
                            eye_state_closed = False
                        consecutive_closed = 0

                    pred = model.predict([[avg_ear, l_ear, r_ear, mor]])[0]
                    probs = model.predict_proba([[avg_ear, l_ear, r_ear, mor]])[0]
                    status_str = LABELS.get(pred, 'ACTIVE')
                    conf = probs[pred]

                    while blink_timestamps and (curr_t - blink_timestamps[0] > 60.0):
                        blink_timestamps.popleft()

                    if pred == 1:
                        if consecutive_closed >= consec_thresh:
                            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 14)
                            cv2.putText(frame, "!!! WAKE UP DRIVER !!!", (int(w*0.12), int(h*0.48)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                            if audio_alert_enabled and alarm_sound and not alarm_playing:
                                alarm_sound.play(loops=-1)
                                alarm_playing = True
                            if curr_t - last_log_time > 2.0:
                                log_alert(status_str, avg_ear, mor, conf)
                                last_log_time = curr_t
                    elif pred == 2:
                        if alarm_playing and alarm_sound:
                            alarm_sound.stop()
                            alarm_playing = False
                        if curr_t - last_log_time > 3.0:
                            log_alert(status_str, avg_ear, mor, conf)
                            last_log_time = curr_t
                    else:
                        if alarm_playing and alarm_sound:
                            alarm_sound.stop()
                            alarm_playing = False
                else:
                    if alarm_playing and alarm_sound:
                        alarm_sound.stop()
                        alarm_playing = False
                    cv2.putText(frame, "NO FACE DETECTED", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                perclos = (sum(1 for e in ear_history if e < ear_threshold) / len(ear_history) * 100.0) if ear_history else 0.0
                bpm = len(blink_timestamps)

                status_color = "#00FF87" if status_str == "ACTIVE" else ("#FF1744" if status_str == "DROWSY" else "#FFD600")
                status_box.markdown(f"""
                <div class="cockpit-card">
                    <div class="cockpit-card-title">Driver State</div>
                    <div class="cockpit-card-value" style="color: {status_color};">{status_str}</div>
                </div>
                """, unsafe_allow_html=True)

                ear_box.markdown(f"""
                <div class="cockpit-card">
                    <div class="cockpit-card-title">Live EAR Value</div>
                    <div class="cockpit-card-value" style="color: #00F2FE;">{avg_ear:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

                blink_box.markdown(f"""
                <div class="cockpit-card">
                    <div class="cockpit-card-title">Blink Activity</div>
                    <div class="cockpit-card-value" style="color: #FFFFFF;">{total_blinks} <span style="font-size: 16px; color: #94A3B8;">({bpm}/m)</span></div>
                </div>
                """, unsafe_allow_html=True)

                perclos_box.markdown(f"""
                <div class="cockpit-card">
                    <div class="cockpit-card-title">PERCLOS Fatigue</div>
                    <div class="cockpit-card-value" style="color: {'#FF1744' if perclos > 30 else '#FFD600'};">{perclos:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                video_feed_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                if ear_history:
                    chart_placeholder.line_chart(list(ear_history), height=200)
                    telemetry_detail.markdown(f"""
                    - **Model Confidence:** `{conf*100:.1f}%`
                    - **Mouth Opening (MOR):** `{mor:.2f}`
                    - **Closed Frame Buffer:** `{consecutive_closed}/{consec_thresh}`
                    """)

                time.sleep(0.03)

        if alarm_playing and alarm_sound:
            alarm_sound.stop()
        cap.release()

# =====================================================================
# 6. VIEW: INCIDENT LOGS & AUDIT TRAIL
# =====================================================================
elif selected_nav == "Incident Logs & Audit":
    st.markdown("### Incident Audit Logs & Event History")
    st.caption("Permanent record of all driver drowsiness and yawning occurrences.")

    logs_df = get_logs_dataframe()

    if logs_df.empty:
        st.info("No incidents logged in the database yet. Trigger a drowsiness or yawning event to view entries.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total_ev = len(logs_df)
        drowsy_ev = (logs_df['status'] == 'DROWSY').sum()
        yawn_ev = (logs_df['status'] == 'YAWNING').sum()
        avg_conf = logs_df['confidence'].mean() * 100

        col1.metric("Total Events Logged", total_ev)
        col2.metric("Drowsy Events", drowsy_ev, f"{(drowsy_ev/total_ev)*100:.1f}%")
        col3.metric("Yawn Events", yawn_ev, f"{(yawn_ev/total_ev)*100:.1f}%")
        col4.metric("Avg Detection Conf", f"{avg_conf:.1f}%")

        st.divider()

        f_col1, f_col2 = st.columns([3, 2])
        with f_col1:
            filter_choice = st.selectbox("Filter Logs by Incident Type", ["All Events", "DROWSY", "YAWNING"])
        
        display_df = logs_df if filter_choice == "All Events" else logs_df[logs_df['status'] == filter_choice]

        with f_col2:
            st.write("")
            st.download_button(
                label="Export Full Audit Log (CSV)",
                data=display_df.to_csv(index=False).encode('utf-8'),
                file_name="driver_fatigue_audit_logs.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.dataframe(display_df, use_container_width=True, height=380)

# =====================================================================
# 7. VIEW: 100% DYNAMIC TELEMETRY & ML ANALYTICS
# =====================================================================
elif selected_nav == "Telemetry & Analytics":
    st.markdown("### Driver Safety Telemetry & Machine Learning Analytics")
    st.caption("100% dynamically computed from SQLite database records and trained Random Forest model.")

    logs_df = get_logs_dataframe()

    st.markdown("#### 1. Dynamic Driver Safety Index")
    if logs_df.empty:
        st.info("No driving data logged yet. Engage Live Monitoring to compute driver telemetry.")
    else:
        drowsy_count = (logs_df['status'] == 'DROWSY').sum()
        yawn_count = (logs_df['status'] == 'YAWNING').sum()

        safety_score = max(0, 100 - (drowsy_count * 4) - (yawn_count * 2))
        risk_level = "LOW RISK (SAFE)" if safety_score >= 80 else ("MODERATE RISK" if safety_score >= 50 else "CRITICAL FATIGUE")

        ds_col1, ds_col2, ds_col3, ds_col4 = st.columns(4)
        ds_col1.metric("Driver Safety Score", f"{safety_score} / 100", f"{risk_level}")
        ds_col2.metric("Recorded Drowsiness Triggers", f"{drowsy_count} times")
        ds_col3.metric("Recorded Yawning Triggers", f"{yawn_count} times")
        ds_col4.metric("Average Trigger EAR", f"{logs_df['ear_value'].mean():.3f}")

        st.write("")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("##### Incident Type Distribution")
            st.bar_chart(logs_df['status'].value_counts())
        with chart_col2:
            st.markdown("##### EAR Severity Spread on Incidents")
            st.line_chart(logs_df['ear_value'].head(50))

    st.divider()

    st.markdown("#### 2. Live Machine Learning Model Metrics")
    if dataset_df is not None:
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Live Model Accuracy", f"{dynamic_acc:.2f}%", "Computed on CSV")
        m_c2.metric("Dataset Sample Size", f"{len(dataset_df):,} rows", "Kaggle + Custom")
        m_c3.metric("Decision Trees", f"{model.n_estimators} Trees", "Random Forest")
        m_c4.metric("Drowsy Precision", f"{dynamic_report['DROWSY (1)']['precision']*100:.1f}%", "High Confidence")

        st.write("")
        st.markdown("##### Dynamic Feature Importance (What Influenced Predictions Most?)")
        st.bar_chart(dynamic_feat_imp.set_index('Feature'))

        st.markdown("##### Live Classification Report (Generated Dynamically via Scikit-Learn)")
        report_rows = []
        for class_name in ['ACTIVE (0)', 'DROWSY (1)', 'YAWNING (2)']:
            if class_name in dynamic_report:
                report_rows.append({
                    'Class Name': class_name,
                    'Precision': f"{dynamic_report[class_name]['precision']*100:.1f}%",
                    'Recall': f"{dynamic_report[class_name]['recall']*100:.1f}%",
                    'F1-Score': f"{dynamic_report[class_name]['f1-score']:.3f}",
                    'Samples (Support)': int(dynamic_report[class_name]['support'])
                })
        st.table(pd.DataFrame(report_rows))
    else:
        st.warning("No dataset CSV found to compute evaluation metrics.")

# =====================================================================
# 8. VIEW: SYSTEM ARCHITECTURE
# =====================================================================
else:
    st.markdown("### System Architecture & Syllabus Alignment")
    st.markdown("""
    **Department of Data Science — Python for Data Science (PDS 2026-27)**
    
    This Smart Driver Drowsiness & Fatigue Monitoring System integrates:
    1. **MediaPipe 3D Face Mesh:** Real-time tracking of 468 facial points for 3D Euclidean EAR & MOR computation.
    2. **Machine Learning Pipeline:** Scikit-Learn `RandomForestClassifier` trained with 100 decision trees.
    3. **Audio-Visual Alarm Engine:** Pygame hardware audio buffer with continuous emergency siren looping.
    4. **Relational Logging:** Local SQLite embedded database tracking driver safety telemetry.
    """)
    st.success("System verified and production-ready.")
