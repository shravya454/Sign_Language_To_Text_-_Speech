import os
import cv2
import numpy as np
import mediapipe as mp

def cv2_imread_unicode(filepath):
    try:
        with open(filepath, 'rb') as f:
            bytes_data = f.read()
        image_array = np.frombuffer(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.01)

hindi_path = 'data/hindi'
classes = sorted([d for d in os.listdir(hindi_path) if os.path.isdir(os.path.join(hindi_path, d))])

det_classes = 0
for c in classes:
    cp = os.path.join(hindi_path, c)
    files = [f for f in os.listdir(cp) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]
    found = False
    for f in files:
        img = cv2_imread_unicode(os.path.join(cp, f))
        if img is None:
            continue
        # Resize small images to 300x300 and pad with border
        if img.shape[0] < 200 or img.shape[1] < 200:
            img = cv2.resize(img, (300, 300))
        padded = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0,0,0])
        res = hands.process(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            found = True
            break
    if found:
        det_classes += 1

print(f"Detected hands in {det_classes}/{len(classes)} Hindi classes with resize+padding!")
