import os
import sys
import json
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from detector import compute_81_features

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

def train_language(language_name, max_samples_per_class=60):
    print(f"\n==========================================", flush=True)
    print(f"  Training Model for: {language_name.upper()}", flush=True)
    print(f"==========================================", flush=True)
    
    lang_path = os.path.join(DATA_DIR, language_name)
    if not os.path.exists(lang_path):
        print(f"Error: Path {lang_path} does not exist.", flush=True)
        return

    classes = sorted([d for d in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, d))])
    print(f"Found {len(classes)} classes for {language_name}.", flush=True)
    
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.01)
    
    X = []
    y = []
    valid_classes = []
    class_sample_counts = {}

    for idx, cls_name in enumerate(classes):
        cls_path = os.path.join(lang_path, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > max_samples_per_class:
            indices = np.linspace(0, len(files) - 1, max_samples_per_class, dtype=int)
            files = [files[i] for i in indices]
            
        cls_count = 0
        for f in files:
            img_path = os.path.join(cls_path, f)
            img = cv2_imread_unicode(img_path)
            if img is None:
                continue
            
            if img.shape[0] < 200 or img.shape[1] < 200:
                img = cv2.resize(img, (300, 300))
            padded = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[0,0,0])

            img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(img_rgb)
            if not results.multi_hand_landmarks:
                continue
            
            hand_landmarks = results.multi_hand_landmarks[0]
            feats = compute_81_features(hand_landmarks)
            
            X.append(feats)
            if cls_name not in valid_classes:
                valid_classes.append(cls_name)
            y.append(valid_classes.index(cls_name))
            cls_count += 1
            
        if cls_count > 0:
            class_sample_counts[cls_name] = cls_count
            
        if (idx + 1) % 10 == 0 or (idx + 1) == len(classes):
            print(f"Processed {idx+1}/{len(classes)} classes. Samples extracted: {len(X)}", flush=True)

    hands_detector.close()
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print(f"Error: No valid features extracted for {language_name}.", flush=True)
        return

    print(f"\nFinal dataset for {language_name.upper()}: X={X.shape}, y={y.shape}, Classes={len(valid_classes)}", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    print(f"Training Random Forest classifier (150 trees)...", flush=True)
    clf = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"--> [SUCCESS] Accuracy for {language_name.upper()}: {acc * 100:.2f}%", flush=True)
    
    model_path = os.path.join(MODELS_DIR, f"{language_name}_model.pkl")
    labels_path = os.path.join(MODELS_DIR, f"{language_name}_classes.json")

    idx_to_label = {i: cls_name for i, cls_name in enumerate(valid_classes)}

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': valid_classes, 'idx_to_label': idx_to_label}, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {language_name} model to '{model_path}'.", flush=True)

if __name__ == '__main__':
    train_language('kannada', max_samples_per_class=60)
    train_language('hindi', max_samples_per_class=60)
    train_language('tamil', max_samples_per_class=30)
    print("\n[ALL DONE] Models trained successfully with 81-dim features!", flush=True)
