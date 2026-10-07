import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os
import pygame

def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def calculate_mor(mouth_landmarks):
    height = np.linalg.norm(mouth_landmarks[13] - mouth_landmarks[14])
    width = np.linalg.norm(mouth_landmarks[78] - mouth_landmarks[308])
    if width == 0:
        return 0.0
    return height / width

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def play_beep():
    try:
        import winsound
        winsound.Beep(1200, 200)
    except Exception:
        pass

def main():
    os.makedirs('dataset', exist_ok=True)
    csv_file = 'dataset/custom_features.csv'

    stages = [
        {"label": 0, "name": "ACTIVE (Keep Eyes Open)", "instruct": "Look normally at camera with eyes OPEN", "color": (0, 255, 0)},
        {"label": 1, "name": "DROWSY (Close Your Eyes)", "instruct": "Keep both eyes completely CLOSED", "color": (0, 0, 255)},
        {"label": 2, "name": "YAWNING (Open Mouth Wide)", "instruct": "YAWN - Open your mouth wide", "color": (0, 255, 255)}
    ]

    cap = cv2.VideoCapture(0)
    mp_face = mp.solutions.face_mesh
    all_collected_data = []

    print("\n=======================================================")
    print("      SMART 3-IN-1 GUIDED DATASET COLLECTOR")
    print("=======================================================")
    print("This script will guide you through 3 quick stages:")
    print("  Stage 1: Open Eyes (Alert)")
    print("  Stage 2: Closed Eyes (Drowsy)")
    print("  Stage 3: Mouth Open (Yawning)")
    print("Each stage takes ~10 seconds. Press 'q' in window to cancel.")
    print("=======================================================\n")

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
        for idx, stage in enumerate(stages):
            print(f"\n[Stage {idx+1}/3] Preparing for: {stage['name']}...")
            play_beep()

            # 4-Second Countdown before recording starts
            start_countdown = time.time()
            while time.time() - start_countdown < 4.0:
                ret, frame = cap.read()
                if not ret: break
                h, w, _ = frame.shape
                rem = 4 - int(time.time() - start_countdown)
                
                # Dark overlay banner
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 140), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                cv2.putText(frame, f"STAGE {idx+1}/3: {stage['name']}", (25, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, stage['color'], 2)
                cv2.putText(frame, f"{stage['instruct']}", (25, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Starting in: {rem}...", (25, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow('Smart Guided Collector', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("[Cancelled by user]")
                    return

            play_beep()
            # Capture 300 frames for this stage
            stage_data = []
            while len(stage_data) < 300:
                ret, frame = cap.read()
                if not ret: break
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)

                avg_ear, mor = 0.0, 0.0
                if res.multi_face_landmarks:
                    pts = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark])
                    l_ear = calculate_ear(pts[LEFT_EYE])
                    r_ear = calculate_ear(pts[RIGHT_EYE])
                    avg_ear = (l_ear + r_ear) / 2.0
                    mor = calculate_mor(pts)

                    stage_data.append([avg_ear, l_ear, r_ear, mor, stage['label']])

                    cv2.putText(frame, f"RECORDING {stage['name']}: {len(stage_data)}/300", 
                                (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage['color'], 2)
                    cv2.putText(frame, f"EAR: {avg_ear:.2f} | MOR: {mor:.2f}", 
                                (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "NO FACE DETECTED! Look at camera.", (25, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow('Smart Guided Collector', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("[Cancelled by user]")
                    return

            all_collected_data.extend(stage_data)
            print(f"[Success] Stage {idx+1} complete: 300 frames captured.")

    cap.release()
    cv2.destroyAllWindows()
    play_beep()

    if all_collected_data:
        df = pd.DataFrame(all_collected_data, columns=['avg_ear', 'l_ear', 'r_ear', 'mor', 'label'])
        header = not os.path.exists(csv_file)
        df.to_csv(csv_file, mode='a', header=header, index=False)
        print("\n=======================================================")
        print(f"[COMPLETED] Total {len(all_collected_data)} custom samples collected!")
        print(f"[Saved] Appended to: {csv_file}")
        print("Next step: Run 'python 2_train_model.py' to train your hybrid model!")
        print("=======================================================\n")

if __name__ == '__main__':
    main()
