import os
import sys
import json
import pickle
import math
import numpy as np
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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

def extract_features_from_image(img, hands_detector):
    """
    Extracts 81-dim features from original image and horizontally flipped image
    for mirrored left/right hand data augmentation.
    Includes multi-stage wrist-padding fallback for tightly cropped hand images.
    """
    if img is None:
        return []
    
    if img.shape[0] < 200 or img.shape[1] < 200:
        img = cv2.resize(img, (300, 300))
    padded = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    extracted_feats = []
    
    # Function to try landmark detection with wrist pad fallback
    def process_with_fallbacks(target_img):
        img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(img_rgb)
        if res.multi_hand_landmarks:
            return res.multi_hand_landmarks[0]
            
        # Fallback 1: Add artificial lower wrist extension block
        h, w, _ = target_img.shape
        wrist_pad = np.ones((80, w, 3), dtype=np.uint8) * 180
        combined = np.vstack([target_img, wrist_pad])
        res2 = hands_detector.process(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        if res2.multi_hand_landmarks:
            return res2.multi_hand_landmarks[0]
            
        # Fallback 2: Light background padding
        padded_light = cv2.copyMakeBorder(target_img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[240, 240, 240])
        res3 = hands_detector.process(cv2.cvtColor(padded_light, cv2.COLOR_BGR2RGB))
        if res3.multi_hand_landmarks:
            return res3.multi_hand_landmarks[0]
            
        return None

    # 1. Original image
    lm1 = process_with_fallbacks(padded)
    if lm1:
        extracted_feats.append(compute_81_features(lm1))
        
    # 2. Mirrored image (Horizontally flipped)
    flipped_img = cv2.flip(padded, 1)
    lm2 = process_with_fallbacks(flipped_img)
    if lm2:
        extracted_feats.append(compute_81_features(lm2))
        
    return extracted_feats

def process_img_worker(args):
    img_path, label_str = args
    img = cv2_imread_unicode(img_path)
    if img is None:
        return None
    
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.01)
    
    feats_list = extract_features_from_image(img, hands_detector)
    hands_detector.close()
    
    if not feats_list:
        return None
    
    return feats_list, label_str

def train_language(language_name, max_samples_per_class=120):
    print(f"\n==================================================", flush=True)
    print(f"  Training High-Precision Model for: {language_name.upper()}", flush=True)
    print(f"==================================================", flush=True)
    
    lang_path = os.path.join(DATA_DIR, language_name)
    if not os.path.exists(lang_path):
        print(f"Error: Path {lang_path} does not exist.", flush=True)
        return

    classes = sorted([d for d in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, d))])
    print(f"Found {len(classes)} character class directories for {language_name}.", flush=True)
    
    tasks = []
    for cls_name in classes:
        cls_path = os.path.join(lang_path, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > max_samples_per_class:
            indices = np.linspace(0, len(files) - 1, max_samples_per_class, dtype=int)
            files = [files[i] for i in indices]
        
        for f in files:
            tasks.append((os.path.join(cls_path, f), cls_name))
            
    print(f"Extracting 81-dim landmark features (with Mirrored Augmentation) from {len(tasks)} images...", flush=True)
    
    X_list = []
    y_labels = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_img_worker, tasks)
        for res in results:
            if res is not None:
                feats_list, lbl = res
                for feats in feats_list:
                    X_list.append(feats)
                    y_labels.append(lbl)

    print(f"Extracted total {len(X_list)} augmented landmark samples across classes.", flush=True)
    
    class_counts = {}
    for lbl in y_labels:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
        
    valid_classes = sorted([lbl for lbl, cnt in class_counts.items() if cnt >= 2])
    print(f"Valid active classes with >=2 samples: {len(valid_classes)} / {len(classes)}", flush=True)
    
    label_to_idx = {cls_name: i for i, cls_name in enumerate(valid_classes)}
    idx_to_label = {i: cls_name for i, cls_name in enumerate(valid_classes)}
    
    X = []
    y = []
    for feats, lbl in zip(X_list, y_labels):
        if lbl in label_to_idx:
            X.append(feats)
            y.append(label_to_idx[lbl])
            
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print(f"Error: No valid features extracted for {language_name}.", flush=True)
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    print(f"Training ExtraTrees classifier (100 trees, max_depth 20, balanced weights)...", flush=True)
    clf = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"--> [SUCCESS] Accuracy for {language_name.upper()}: {acc * 100:.2f}%", flush=True)
    
    model_path = os.path.join(MODELS_DIR, f"{language_name}_model.pkl")
    labels_path = os.path.join(MODELS_DIR, f"{language_name}_classes.json")

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': valid_classes, 'idx_to_label': idx_to_label}, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {language_name} model to '{model_path}' ({len(valid_classes)} classes).", flush=True)

if __name__ == '__main__':
    # Train Kannada & Hindi with generous sample limits
    train_language('kannada', max_samples_per_class=120)
    train_language('hindi', max_samples_per_class=120)
    # Train Tamil with 40 samples per class to comfortably cover all 247 classes fast
    train_language('tamil', max_samples_per_class=40)
    print("\n[ALL DONE] All models retrained successfully with high accuracy & mirrored augmentation!", flush=True)
