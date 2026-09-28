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

def process_img(args):
    img_path, label_str, mp_hands_module = args
    img = cv2_imread_unicode(img_path)
    if img is None:
        return None
    
    hands = mp_hands_module.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.05)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        # Try border padding
        padded = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0,0,0])
        results = hands.process(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))

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
        
    return norm_landmarks, label_str

def train_language(language_name, max_samples_per_class=40):
    print(f"\n==========================================", flush=True)
    print(f"  Fast Training Model for: {language_name.upper()}", flush=True)
    print(f"==========================================", flush=True)
    
    lang_path = os.path.join(DATA_DIR, language_name)
    if not os.path.exists(lang_path):
        print(f"Error: Path {lang_path} does not exist.", flush=True)
        return

    classes = sorted([d for d in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, d))])
    print(f"Found {len(classes)} classes for {language_name}.", flush=True)
    
    mp_hands_module = mp.solutions.hands
    
    tasks = []
    for cls_name in classes:
        cls_path = os.path.join(lang_path, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > max_samples_per_class:
            indices = np.linspace(0, len(files) - 1, max_samples_per_class, dtype=int)
            files = [files[i] for i in indices]
        
        for f in files:
            tasks.append((os.path.join(cls_path, f), cls_name, mp_hands_module))
            
    print(f"Processing {len(tasks)} images in parallel...", flush=True)
    
    raw_samples = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_img, tasks)
        for res in results:
            if res is not None:
                raw_samples.append(res)

    print(f"Extracted {len(raw_samples)}/{len(tasks)} valid landmark features.", flush=True)
    
    # Filter classes with at least 2 samples to allow train/test split
    class_counts = {}
    for feats, lbl in raw_samples:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
        
    valid_classes = sorted([lbl for lbl, cnt in class_counts.items() if cnt >= 2])
    print(f"Active classes with >=2 samples: {len(valid_classes)} / {len(classes)}", flush=True)
    
    label_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
    idx_to_label = {i: cls_name for i, cls_name in enumerate(valid_classes)}
    
    X = []
    y = []
    for feats, lbl in raw_samples:
        if lbl in label_to_idx:
            X.append(feats)
            y.append(label_to_idx[lbl])
            
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print(f"Error: No valid features extracted for {language_name}.", flush=True)
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    print(f"Training Random Forest classifier...", flush=True)
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"--> [SUCCESS] Accuracy for {language_name.upper()}: {acc * 100:.2f}%", flush=True)
    
    model_path = os.path.join(MODELS_DIR, f"{language_name}_model.pkl")
    labels_path = os.path.join(MODELS_DIR, f"{language_name}_classes.json")

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': valid_classes, 'idx_to_label': idx_to_label}, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {language_name} model to '{model_path}'.", flush=True)

if __name__ == '__main__':
    # Kill previous task-136 first if running
    train_language('kannada', max_samples_per_class=40)
    train_language('hindi', max_samples_per_class=40)
    train_language('tamil', max_samples_per_class=20)
    print("\n[ALL DONE] All language models trained successfully!", flush=True)
