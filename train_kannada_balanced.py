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

sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODELS_DIR = 'models'
DATA_DIR = 'data/kannada'
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

def compute_85_features_from_coords(coords_21x3):
    """
    Computes 85-dim features directly from a 21x3 array of landmark coordinates.
    coords_21x3: list or array of shape (21, 3) representing (x, y, z) for 21 keypoints.
    """
    lms = coords_21x3
    x0, y0, z0 = lms[0]
    x9, y9, z9 = lms[9]
    
    palm_dist = math.sqrt((x9 - x0)**2 + (y9 - y0)**2 + (z9 - z0)**2)
    if palm_dist < 1e-5:
        palm_dist = 1.0

    # 1. 63 Normalized Relative 3D coordinates
    norm_coords = []
    for i in range(21):
        norm_coords.append((lms[i][0] - x0) / palm_dist)
        norm_coords.append((lms[i][1] - y0) / palm_dist)
        norm_coords.append((lms[i][2] - z0) / palm_dist)

    def dist3d(i, j):
        return math.sqrt((lms[i][0]-lms[j][0])**2 + (lms[i][1]-lms[j][1])**2 + (lms[i][2]-lms[j][2])**2) / palm_dist

    # 2. 5 Fingertip-to-Wrist normalized distances
    tips = [4, 8, 12, 16, 20]
    tip_to_wrist = [dist3d(t, 0) for t in tips]

    # 3. 5 Adjacent fingertip distances
    adj_pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20)]
    adj_dists = [dist3d(p1, p2) for p1, p2 in adj_pairs]

    # 4. 8 Fingertip to Knuckle distances
    mcp_pairs = [(4, 5), (4, 9), (4, 13), (4, 17), (8, 2), (12, 5), (16, 9), (20, 13)]
    mcp_dists = [dist3d(p1, p2) for p1, p2 in mcp_pairs]

    base_81 = norm_coords + tip_to_wrist + adj_dists + mcp_dists

    # 5. Hand tilt angle in XY plane (Wrist to Middle MCP)
    dx = x9 - x0
    dy = y9 - y0
    dz = z9 - z0
    d_xy = math.sqrt(dx**2 + dy**2)
    tilt_xy = math.atan2(dy, dx) / math.pi

    # 6. Hand tilt elevation angle in Z
    tilt_z = math.atan2(dz, d_xy + 1e-6) / (math.pi / 2.0)

    # 7. Index finger curl ratio: dist(8, 5) / sum of segments
    def raw_dist3d(i, j):
        return math.sqrt((lms[i][0]-lms[j][0])**2 + (lms[i][1]-lms[j][1])**2 + (lms[i][2]-lms[j][2])**2)

    seg_idx = raw_dist3d(5, 6) + raw_dist3d(6, 7) + raw_dist3d(7, 8)
    curl_idx = raw_dist3d(8, 5) / (seg_idx + 1e-5)

    # 8. Middle finger curl ratio: dist(12, 9) / sum of segments
    seg_mid = raw_dist3d(9, 10) + raw_dist3d(10, 11) + raw_dist3d(11, 12)
    curl_mid = raw_dist3d(12, 9) / (seg_mid + 1e-5)

    return base_81 + [tilt_xy, tilt_z, curl_idx, curl_mid]

def augment_landmarks(coords_21x3, num_augments=3):
    """
    Generates synthetic landmark samples via:
    - 3D Gaussian coordinate jitter (sigma = 0.008)
    - Scale jitter (+/- 8%)
    - 2D rotation jitter (+/- 5 deg, +/- 10 deg)
    """
    augmented = []
    lms = np.array(coords_21x3, dtype=np.float32)
    x0, y0, z0 = lms[0]

    for _ in range(num_augments):
        aug = lms.copy()
        
        # 1. Scale jitter (+/- 8%)
        scale = np.random.uniform(0.92, 1.08)
        aug = lms[0] + (aug - lms[0]) * scale
        
        # 2. 2D rotation jitter around wrist (-10 to +10 degrees)
        angle_deg = np.random.uniform(-10.0, 10.0)
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        dx = aug[:, 0] - x0
        dy = aug[:, 1] - y0
        aug[:, 0] = x0 + dx * cos_a - dy * sin_a
        aug[:, 1] = y0 + dx * sin_a + dy * cos_a
        
        # 3. 3D Gaussian coordinate jitter (sigma = 0.008)
        noise = np.random.normal(0, 0.008, aug.shape)
        # Keep wrist anchor relatively stable
        noise[0] *= 0.2
        aug += noise
        
        augmented.append(aug.tolist())
        
    return augmented

def process_image(args):
    img_path, label = args
    img = cv2_imread_unicode(img_path)
    if img is None:
        return None
        
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.1
    )
    
    # 1. Direct detection
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)
    
    lm_list = None
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        lm_list = [[p.x, p.y, p.z] for p in lm.landmark]
    else:
        # Fallback padding with border
        padded = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        res2 = hands_detector.process(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        if res2.multi_hand_landmarks:
            lm = res2.multi_hand_landmarks[0]
            lm_list = [[p.x, p.y, p.z] for p in lm.landmark]
            
    hands_detector.close()
    if lm_list is None:
        return None
        
    return lm_list, label

def main():
    print("==========================================================")
    print("  KANNADA MODEL RETRAINING: 85-DIM ORIENTATION & AUGMENTATION")
    print("==========================================================")

    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Total Kannada classes found: {len(classes)}")

    # Target: 50 images per class for balanced extraction, then augment
    tasks = []
    for c in classes:
        p = os.path.join(DATA_DIR, c)
        files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > 50:
            indices = np.linspace(0, len(files) - 1, 50, dtype=int)
            files = [files[i] for i in indices]
        for f in files:
            tasks.append((os.path.join(p, f), c))

    print(f"Extracting landmarks from {len(tasks)} images with thread pool...")
    raw_samples_by_class = {c: [] for c in classes}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_image, tasks)
        for r in results:
            if r is not None:
                lm_list, lbl = r
                raw_samples_by_class[lbl].append(lm_list)

    print("\n--- Raw Extraction Counts & Synthetic Augmentation ---")
    X_all = []
    y_all = []

    confused_targets = {'ಅಃ', 'ಋ', 'ಒ', 'ಈ', 'ದ', 'ಓ', 'ಉ', 'ಅಂ', 'ಐ'}

    for c in classes:
        raw_list = raw_samples_by_class[c]
        count = len(raw_list)
        if count == 0:
            print(f"Warning: class {c} has 0 samples extracted!")
            continue

        # Add raw 85-dim features
        for lm in raw_list:
            X_all.append(compute_85_features_from_coords(lm))
            y_all.append(c)

        # Augmentation: if confused class or count < 60, add more synthetic samples
        num_aug = 3 if c in confused_targets else 1
        for lm in raw_list:
            augmented_lms = augment_landmarks(lm, num_augments=num_aug)
            for aug in augmented_lms:
                X_all.append(compute_85_features_from_coords(aug))
                y_all.append(c)

        total_class_samples = len([y for y in y_all if y == c])
        print(f"Class '{c}': raw={count} -> total with augmentation={total_class_samples}")

    X = np.array(X_all, dtype=np.float32)
    y_labels = y_all

    unique_classes = sorted(list(set(y_labels)))
    label_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    idx_to_label = {i: cls_name for i, cls_name in enumerate(unique_classes)}
    y = np.array([label_to_idx[l] for l in y_labels], dtype=np.int32)

    print(f"\nTotal Dataset Shape: X={X.shape} (85 features), y={y.shape} across {len(unique_classes)} classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining ExtraTreesClassifier (85-dim features, 100 trees, max_depth=22)...")
    clf = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=22,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"--> Train Accuracy: {train_acc * 100:.2f}%")
    print(f"--> Test Accuracy:  {test_acc * 100:.2f}%")

    model_path = os.path.join(MODELS_DIR, 'kannada_model.pkl')
    label_path = os.path.join(MODELS_DIR, 'kannada_classes.json')

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': unique_classes, 'idx_to_label': idx_to_label}, f, ensure_ascii=False, indent=2)

    print(f"Saved Kannada model to '{model_path}' ({os.path.getsize(model_path)/(1024*1024):.1f} MB).")
    print(f"Saved classes to '{label_path}'.")

if __name__ == '__main__':
    main()
