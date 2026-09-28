import os
import sys
import json
import pickle
import numpy as np
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODELS_DIR = 'models'
DATA_DIR = 'data'
os.makedirs(MODELS_DIR, exist_ok=True)

def cv2_imread_unicode(filepath):
    try:
        with open(filepath, 'rb') as f:
            bytes_data = f.read()
        image_array = np.frombuffer(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None

def process_single_image(img_path, label_idx, mp_hands_module):
    img = cv2_imread_unicode(img_path)
    if img is None:
        return None
    
    # Instantiate hands detector inside thread
    hands = mp_hands_module.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hands.close()
    
    if not results.multi_hand_landmarks:
        return None
    
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    base_x, base_y, base_z = landmarks[0], landmarks[1], landmarks[2]
    norm_landmarks = []
    for i in range(0, len(landmarks), 3):
        norm_landmarks.append(landmarks[i] - base_x)
        norm_landmarks.append(landmarks[i+1] - base_y)
        norm_landmarks.append(landmarks[i+2] - base_z)
    
    max_val = max(abs(x) for x in norm_landmarks)
    if max_val > 0:
        norm_landmarks = [x / max_val for x in norm_landmarks]
        
    return norm_landmarks, label_idx

def train_language(language_name, max_samples_per_class=60):
    print(f"\n==========================================", flush=True)
    print(f"  Training Model for Language: {language_name.upper()}", flush=True)
    print(f"==========================================", flush=True)
    
    lang_path = os.path.join(DATA_DIR, language_name)
    if not os.path.exists(lang_path):
        print(f"Error: Path {lang_path} does not exist.", flush=True)
        return

    classes = sorted([d for d in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, d))])
    print(f"Found {len(classes)} classes for {language_name}.", flush=True)
    
    label_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_label = {i: cls_name for i, cls_name in enumerate(classes)}
    
    tasks = []
    mp_hands_module = mp.solutions.hands
    
    for cls_name in classes:
        cls_path = os.path.join(lang_path, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > max_samples_per_class:
            indices = np.linspace(0, len(files) - 1, max_samples_per_class, dtype=int)
            files = [files[i] for i in indices]
        
        for f in files:
            img_path = os.path.join(cls_path, f)
            tasks.append((img_path, label_to_idx[cls_name]))
            
    print(f"Processing {len(tasks)} total images using multi-threading...", flush=True)
    
    X = []
    y = []
    
    # Process tasks in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_image, t[0], t[1], mp_hands_module) for t in tasks]
        for f in futures:
            res = f.result()
            if res is not None:
                feats, lbl = res
                X.append(feats)
                y.append(lbl)
                
    X = np.array(X)
    y = np.array(y)
    print(f"Successfully extracted {len(X)}/{len(tasks)} valid hand landmark features. Shape: {X.shape}", flush=True)
    
    if len(X) == 0:
        print(f"Error: No valid features extracted for {language_name}.", flush=True)
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    print(f"Training Random Forest classifier for {language_name}...", flush=True)
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"--> [SUCCESS] Accuracy for {language_name.upper()}: {acc * 100:.2f}%", flush=True)
    
    model_path = os.path.join(MODELS_DIR, f"{language_name}_model.pkl")
    labels_path = os.path.join(MODELS_DIR, f"{language_name}_classes.json")
    
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': classes, 'idx_to_label': idx_to_label}, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {language_name} model to '{model_path}'.", flush=True)

if __name__ == '__main__':
    train_language('kannada', max_samples_per_class=60)
    train_language('hindi', max_samples_per_class=50)
    train_language('tamil', max_samples_per_class=30)
    print("\n[ALL DONE] All models trained successfully!", flush=True)
