import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
import time
import os
from collections import deque
from db.db_config import log_alert, init_db

# =====================================================================
# MODULE 1 (OOP): FEATURE EXTRACTOR
# Computes EAR and MOR from 3D MediaPipe Facial Landmarks
# =====================================================================
class FeatureExtractor:
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 291, 0, 17, 13, 14, 78, 308]

    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, frame_rgb, width, height):
        res = self.face_mesh.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None
        return np.array([(p.x * width, p.y * height) for p in res.multi_face_landmarks[0].landmark])

    @staticmethod
    def calculate_ear(pts, eye_indices):
        A = np.linalg.norm(pts[eye_indices[1]] - pts[eye_indices[5]])
        B = np.linalg.norm(pts[eye_indices[2]] - pts[eye_indices[4]])
        C = np.linalg.norm(pts[eye_indices[0]] - pts[eye_indices[3]])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    @staticmethod
    def calculate_mor(pts):
        height = np.linalg.norm(pts[13] - pts[14])
        width = np.linalg.norm(pts[78] - pts[308])
        if width == 0:
            return 0.0
        return height / width

    def get_features(self, pts):
        l_ear = self.calculate_ear(pts, self.LEFT_EYE)
        r_ear = self.calculate_ear(pts, self.RIGHT_EYE)
        avg_ear = (l_ear + r_ear) / 2.0
        mor = self.calculate_mor(pts)
        return avg_ear, l_ear, r_ear, mor


# =====================================================================
# MODULE 2 (OOP & DS STATS): BLINK & PERCLOS TRACKER
# Tracks rolling window statistics, blink frequency, and PERCLOS
# =====================================================================
class BlinkAndPerclosTracker:
    def __init__(self, window_size=60, ear_closed_thresh=0.20):
        self.window_size = window_size
        self.ear_closed_thresh = ear_closed_thresh
        self.ear_history = deque(maxlen=window_size)
        self.blink_timestamps = deque()
        
        self.total_blinks = 0
        self.consecutive_closed_frames = 0
        self.eye_state_closed = False

    def update(self, current_ear):
        self.ear_history.append(current_ear)
        current_time = time.time()

        # Clean up blink timestamps older than 60 seconds for BPM
        while self.blink_timestamps and (current_time - self.blink_timestamps[0] > 60.0):
            self.blink_timestamps.popleft()

        # Blink Detection Logic
        if current_ear < self.ear_closed_thresh:
            self.consecutive_closed_frames += 1
            if not self.eye_state_closed:
                self.eye_state_closed = True
        else:
            if self.eye_state_closed:
                # Normal blink duration is typically 2-12 frames (< 0.4 sec)
                if 2 <= self.consecutive_closed_frames < 15:
                    self.total_blinks += 1
                    self.blink_timestamps.append(current_time)
                self.eye_state_closed = False
            self.consecutive_closed_frames = 0

    def get_perclos(self):
        """Percentage of Eye Closure over rolling window"""
        if not self.ear_history:
            return 0.0
        closed_count = sum(1 for e in self.ear_history if e < self.ear_closed_thresh)
        return (closed_count / len(self.ear_history)) * 100.0

    def get_rolling_stats(self):
        """Descriptive statistics: Mean and Variance of EAR (Module 2)"""
        if not self.ear_history:
            return 0.0, 0.0
        arr = np.array(self.ear_history)
        return np.mean(arr), np.std(arr)

    def get_blink_rate_bpm(self):
        """Blinks per minute based on the last 60 seconds"""
        return len(self.blink_timestamps)


# =====================================================================
# MODULE 3 (OOP): DROWSINESS CLASSIFIER
# Wraps trained Random Forest model with safety checks
# =====================================================================
class DrowsinessClassifier:
    LABELS = {0: 'ACTIVE', 1: 'DROWSY', 2: 'YAWNING'}

    def __init__(self, model_path='models/rfc_drowsiness_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[Error] Model {model_path} not found. Run 2_train_model.py first.")
        self.model = joblib.load(model_path)

    def predict(self, avg_ear, l_ear, r_ear, mor):
        features = np.array([[avg_ear, l_ear, r_ear, mor]])
        pred = self.model.predict(features)[0]
        probs = self.model.predict_proba(features)[0]
        status = self.LABELS.get(pred, 'ACTIVE')
        confidence = probs[pred]
        return pred, status, confidence


# =====================================================================
# MODULE 4 (OOP): ALERT & LOGGING MANAGER
# Audio Siren playback and PostgreSQL / SQLite logging
# =====================================================================
class AlertManager:
    def __init__(self, alarm_path='assets/alarm.wav'):
        init_db()
        self.alarm_path = alarm_path
        self.alarm_sound = None
        self.alarm_playing = False
        self.last_log_time = 0

        pygame.mixer.init()
        if os.path.exists(alarm_path):
            try:
                self.alarm_sound = pygame.mixer.Sound(alarm_path)
            except Exception as e:
                print(f"[Audio Warning] {e}")

    def trigger_alarm(self):
        if self.alarm_sound and not self.alarm_playing:
            self.alarm_sound.play(loops=-1)
            self.alarm_playing = True

    def stop_alarm(self):
        if self.alarm_sound and self.alarm_playing:
            self.alarm_sound.stop()
            self.alarm_playing = False

    def log_incident(self, status, ear, mor, conf, min_interval=2.0):
        current_time = time.time()
        if current_time - self.last_log_time >= min_interval:
            log_alert(status, ear, mor, conf)
            self.last_log_time = current_time


# =====================================================================
# MAIN DETECTOR PIPELINE (VideoHandler)
# Integrates all modules into real-time camera loop
# =====================================================================
class VideoHandler:
    CONSEC_THRESH = 15  # Continuous eye closed frames to trigger alarm

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.tracker = BlinkAndPerclosTracker(window_size=60)
        self.classifier = DrowsinessClassifier()
        self.alert_manager = AlertManager()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] Could not open webcam.")
            return

        print("\n=======================================================")
        print("  SMART DRIVER DROWSINESS & FATIGUE MONITORING SYSTEM  ")
        print("=======================================================")
        print("Classes: ACTIVE | DROWSY | YAWNING")
        print("Press 'q' in video window to exit.")
        print("=======================================================\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pts = self.feature_extractor.extract_landmarks(rgb, w, h)

            status_str, conf = 'ACTIVE', 0.95
            color = (0, 255, 0)
            avg_ear, l_ear, r_ear, mor = 0.30, 0.30, 0.30, 0.20

            if pts is not None:
                avg_ear, l_ear, r_ear, mor = self.feature_extractor.get_features(pts)
                self.tracker.update(avg_ear)

                pred, status_str, conf = self.classifier.predict(avg_ear, l_ear, r_ear, mor)

                # Drowsy Condition
                if pred == 1:
                    if self.tracker.consecutive_closed_frames >= self.CONSEC_THRESH:
                        color = (0, 0, 255)  # Red Alert
                        # Red Flashing Border
                        cv2.rectangle(frame, (0, 0), (w, h), color, 12)
                        cv2.putText(frame, '!!! WAKE UP DRIVER !!!', (int(w * 0.12), int(h * 0.45)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                        self.alert_manager.trigger_alarm()
                        self.alert_manager.log_incident(status_str, avg_ear, mor, conf)

                # Yawning Condition
                elif pred == 2:
                    color = (0, 255, 255)  # Yellow
                    self.alert_manager.stop_alarm()
                    self.alert_manager.log_incident(status_str, avg_ear, mor, conf, min_interval=3.0)

                # Active / Safe Condition
                else:
                    color = (0, 255, 0)  # Green
                    self.alert_manager.stop_alarm()

            else:
                self.alert_manager.stop_alarm()
                cv2.putText(frame, 'NO FACE DETECTED', (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Descriptive Statistics & Real-time Metrics
            perclos = self.tracker.get_perclos()
            mean_ear, std_ear = self.tracker.get_rolling_stats()
            blinks = self.tracker.total_blinks
            bpm = self.tracker.get_blink_rate_bpm()
            closed_cnt = self.tracker.consecutive_closed_frames

            # High-Tech HUD Overlay
            # Header Panel
            cv2.rectangle(frame, (15, 12), (540, 140), (25, 25, 25), -1)
            cv2.rectangle(frame, (15, 12), (540, 140), color, 2)

            cv2.putText(frame, f'STATUS: {status_str} ({conf * 100:.1f}%)', (25, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame, f'EAR: {avg_ear:.2f} | MOR: {mor:.2f} | EAR Std: {std_ear:.3f}', (25, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f'Blinks: {blinks} (Rate: {bpm}/min) | PERCLOS: {perclos:.1f}%', (25, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(frame, f'Closed Frames: {closed_cnt}/{self.CONSEC_THRESH}', (25, 124),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

            cv2.imshow('Smart Driver Drowsiness Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.alert_manager.stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        print("[System Stopped] Exited successfully.")

if __name__ == '__main__':
    detector = VideoHandler()
    detector.run()
