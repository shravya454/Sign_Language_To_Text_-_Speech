import os
import sys
import numpy as np
import cv2
import mediapipe as mp

sys.stdout.reconfigure(encoding='utf-8')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

def extract_landmarks(image_path, hands_detector):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    
    # Pick first hand
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    # Normalize relative to wrist (landmark 0)
    base_x, base_y, base_z = landmarks[0], landmarks[1], landmarks[2]
    norm_landmarks = []
    for i in range(0, len(landmarks), 3):
        norm_landmarks.append(landmarks[i] - base_x)
        norm_landmarks.append(landmarks[i+1] - base_y)
        norm_landmarks.append(landmarks[i+2] - base_z)
    
    # Scale normalization
    max_val = max(abs(x) for x in norm_landmarks)
    if max_val > 0:
        norm_landmarks = [x / max_val for x in norm_landmarks]
        
    return norm_landmarks

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.2)
    
    test_dir = 'data/kannada/ಕ'
    files = os.listdir(test_dir)[:10]
    detected = 0
    for f in files:
        fp = os.path.join(test_dir, f)
        lm = extract_landmarks(fp, hands)
        if lm is not None:
            detected += 1
    print(f"Detected landmarks in {detected}/{len(files)} sample images from {test_dir}")
