import os
import sys
import json
import pickle
import math
import random
import numpy as np
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from detector import compute_multihand_features

sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODELS_DIR = 'models'
DATA_DIR = 'data/tamil'
os.makedirs(MODELS_DIR, exist_ok=True)

# The 31 Authentic Base Alphabet Classes for Tamil Fingerspelling
BASE_CLASSES = [
    # 12 Independent Vowels (Uyir Ezhuthukkal)
    'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ',
    # 1 Ayutha Ezhuthu
    'ஃ',
    # 18 Base Consonants (Mei Ezhuthukkal - Base bodies)
    'க', 'ங', 'ச', 'ஞ', 'ட', 'ண', 'த', 'ந', 'ன', 'ப', 'ம', 'ய', 'ர', 'ற', 'ல', 'ள', 'ழ', 'வ'
]

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
    Extracts 186-dim dual-hand features from image with subtle landmark jitter augmentation.
    """
    if img is None:
        return []
    
    extracted_feats = []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(img_rgb)
    
    if res.multi_hand_landmarks:
        # 1. Base clean sample
        base_feat = compute_multihand_features(res.multi_hand_landmarks)
        extracted_feats.append(base_feat)
        
        # 2. Subtle landmark jitter augmentation (simulates webcam trembling & slight hand shift)
        if len(res.multi_hand_landmarks) >= 2:
            jittered_hands = []
            for h in res.multi_hand_landmarks:
                pts = [[lm.x + np.random.normal(0, 0.003), 
                        lm.y + np.random.normal(0, 0.003), 
                        lm.z + np.random.normal(0, 0.003)] for lm in h.landmark]
                jittered_hands.append(pts)
            extracted_feats.append(compute_multihand_features(jittered_hands))
            
    return extracted_feats

def process_img_worker(args):
    img_path, label_str = args
    img = cv2_imread_unicode(img_path)
    if img is None:
        return None
    
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.1
    )
    
    feats_list = extract_features_from_image(img, hands_detector)
    hands_detector.close()
    
    if not feats_list:
        return None
    
    return feats_list, label_str

def main(samples_per_class=120):
    print("=" * 60)
    print("  TRAINING TAMIL DUAL-HAND BASE ALPHABET MODEL (31 CLASSES)")
    print("  ARCHITECTURE: 186-DIM DUAL-HAND GEOMETRIC & SPATIAL VECTORS")
    print("=" * 60)
    
    tasks = []
    for cls_name in BASE_CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"Warning: Directory not found: {cls_dir}")
            continue
            
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > samples_per_class:
            indices = np.linspace(0, len(files) - 1, samples_per_class, dtype=int)
            files = [files[i] for i in indices]
            
        for f in files:
            tasks.append((os.path.join(cls_dir, f), cls_name))
            
    print(f"Total images scheduled for feature extraction: {len(tasks)}")
    print("Extracting 186-dim dual-hand landmark features with augmentation...")
    
    X_list = []
    y_labels = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_img_worker, tasks)
        for res in results:
            if res is not None:
                feats_list, lbl = res
                for feats in feats_list:
                    if len(feats) == 186:
                        X_list.append(feats)
                        y_labels.append(lbl)
                    
    print(f"Successfully extracted {len(X_list)} dual-hand landmark samples.")
    
    # Verify class sample counts
    counts = {}
    for lbl in y_labels:
        counts[lbl] = counts.get(lbl, 0) + 1
        
    valid_classes = sorted([c for c in BASE_CLASSES if counts.get(c, 0) >= 15])
    print(f"Classes with sufficient samples: {len(valid_classes)} / {len(BASE_CLASSES)}")
    for c in valid_classes:
        print(f"  Class '{c}': {counts.get(c, 0)} samples")
    
    label_to_idx = {c: i for i, c in enumerate(valid_classes)}
    idx_to_label = {i: c for i, c in enumerate(valid_classes)}
    
    X = np.array([f for f, l in zip(X_list, y_labels) if l in label_to_idx], dtype=np.float32)
    y = np.array([label_to_idx[l] for l in y_labels if l in label_to_idx], dtype=np.int32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"\nTraining ExtraTreesClassifier (150 estimators) on {len(X_train)} samples with {X.shape[1]} features...")
    clf = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n==================================================")
    print(f"  [SUCCESS] Test Accuracy: {acc * 100:.2f}%")
    print(f"==================================================")
    
    # Save the new dual-hand model & classes
    model_path = os.path.join(MODELS_DIR, 'tamil_model.pkl')
    labels_path = os.path.join(MODELS_DIR, 'tamil_classes.json')
    
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': valid_classes, 'idx_to_label': idx_to_label}, f, ensure_ascii=False, indent=2)
        
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Saved new Dual-Hand Tamil model to '{model_path}' ({size_mb:.2f} MB).")
    print(f"Saved class mappings to '{labels_path}' ({len(valid_classes)} classes).")

if __name__ == '__main__':
    main(samples_per_class=120)
