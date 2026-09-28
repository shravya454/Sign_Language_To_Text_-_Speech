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

def extract_landmark_features(img, hands_detector):
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    
    # If not found, try with zero padding border
    if not results.multi_hand_landmarks:
        padded = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0,0,0])
        results = hands_detector.process(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        
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
        
    return norm_landmarks

def train_language(language_name, max_samples_per_class=40):
    print(f"\n==========================================", flush=True)
    print(f"  Training Model for Language: {language_name.upper()}", flush=True)
    print(f"==========================================", flush=True)
    
    lang_path = os.path.join(DATA_DIR, language_name)
    if not os.path.exists(lang_path):
        print(f"Error: Path {lang_path} does not exist.", flush=True)
        return

    classes = sorted([d for d in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, d))])
    print(f"Found {len(classes)} class folders for {language_name}.", flush=True)
    
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.05)
    
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
            feats = extract_landmark_features(img, hands_detector)
            if feats is not None:
                X.append(feats)
                y.append(len(valid_classes) if cls_name not in class_sample_counts else valid_classes.index(cls_name))
                cls_count += 1
                
        if cls_count > 0:
            if cls_name not in class_sample_counts:
                valid_classes.append(cls_name)
            class_sample_counts[cls_name] = class_sample_counts.get(cls_name, 0) + cls_count
            
        if (idx + 1) % 10 == 0 or (idx + 1) == len(classes):
            print(f"Processed {idx+1}/{len(classes)} classes. Active classes: {len(valid_classes)}, total samples: {len(X)}", flush=True)

    hands_detector.close()
    
    X = np.array(X)
    y = np.array(y)
    print(f"\nFinal Dataset for {language_name.upper()}: X={X.shape}, y={y.shape}, Classes={len(valid_classes)}", flush=True)
    
    if len(X) == 0:
        print(f"Error: No valid features extracted for {language_name}.", flush=True)
        return

    # Check minimum class counts
    counts = np.bincount(y)
    min_count = np.min(counts) if len(counts) > 0 else 0
    use_stratify = y if min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=use_stratify)
    
    print(f"Training Random Forest classifier for {language_name}...", flush=True)
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"--> [SUCCESS] Accuracy for {language_name.upper()}: {acc * 100:.2f}%", flush=True)
    
    model_path = os.path.join(MODELS_DIR, f"{language_name}_model.pkl")
    labels_path = os.path.join(MODELS_DIR, f"{language_name}_classes.json")
    
    label_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
    idx_to_label = {i: cls_name for i, cls_name in enumerate(valid_classes)}

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': valid_classes, 'idx_to_label': idx_to_label}, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {language_name} model to '{model_path}'.", flush=True)

if __name__ == '__main__':
    train_language('kannada', max_samples_per_class=40)
    train_language('hindi', max_samples_per_class=40)
    train_language('tamil', max_samples_per_class=20)
    print("\n[ALL DONE] All models trained successfully!", flush=True)
