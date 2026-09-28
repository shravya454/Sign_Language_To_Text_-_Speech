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
    except Exception as e:
        return None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)

with open('inspect_out3.txt', 'w', encoding='utf-8') as f_out:
    for lang in ['kannada', 'hindi', 'tamil']:
        path = os.path.join('data', lang)
        if not os.path.exists(path):
            continue
        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        f_out.write(f"=== {lang}: {len(classes)} classes ===\n")
        c0 = classes[0]
        cp = os.path.join(path, c0)
        files = os.listdir(cp)
        if files:
            img_path = os.path.join(cp, files[0])
            img = cv2_imread_unicode(img_path)
            if img is not None:
                f_out.write(f"Class '{c0}', Image shape: {img.shape}, dtype: {img.dtype}\n")
                
                det_count = 0
                for f in files[:10]:
                    fp = os.path.join(cp, f)
                    im = cv2_imread_unicode(fp)
                    if im is not None:
                        res = hands.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                        if res.multi_hand_landmarks:
                            det_count += 1
                f_out.write(f"MediaPipe Hand Landmark Detection Rate for '{c0}': {det_count}/10\n\n")

print("Test 3 complete!")
