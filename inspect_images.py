import os
import cv2
import numpy as np
import mediapipe as mp

output = []
def log(msg):
    print(msg)
    output.append(str(msg))

data_dir = 'data'
mp_hands = mp.solutions.hands

for lang in ['kannada', 'hindi', 'tamil']:
    lang_path = os.path.join(data_dir, lang)
    if not os.path.exists(lang_path):
        continue
    classes = [d for d in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, d))]
    log(f"Language {lang}: {len(classes)} classes found")
    if not classes:
        continue
    c0 = classes[0]
    cp = os.path.join(lang_path, c0)
    files = [f for f in os.listdir(cp) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        log(f"No image files in {cp}")
        continue
    sample_path = os.path.join(cp, files[0])
    img = cv2.imread(sample_path)
    if img is not None:
        log(f"=== {lang} Class '{c0}' Sample '{files[0]}' ===")
        log(f"Shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
        
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)
        det_count = 0
        for f in files[:10]:
            fp = os.path.join(cp, f)
            im = cv2.imread(fp)
            if im is None:
                continue
            res = hands.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                det_count += 1
        log(f"MediaPipe Detected Hands in {det_count}/10 sample images")

with open('inspect_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))
log("Done writing inspect_out.txt")
