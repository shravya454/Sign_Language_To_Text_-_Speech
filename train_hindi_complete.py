import os
import sys
import json
import pickle
import numpy as np
import cv2
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from detector import compute_81_features

sys.stdout.reconfigure(encoding='utf-8')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODELS_DIR = 'models'
DATA_DIR_HINDI = 'data/hindi'
DATA_DIR_KANNADA = 'data/kannada'
os.makedirs(MODELS_DIR, exist_ok=True)

# Complete 42 Hindi Alphabet Classes (Vowels + Consonants + Conjuncts)
ALL_HINDI_CLASSES = [
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऋ', 'ए', 'ऐ', 'ओ',
    'क', 'क्ष', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'ज्ञ', 'झ',
    'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'स', 'ह'
]

# ISL equivalent mappings for classes with severe image truncation in hindi dataset
ISL_EQUIVALENT_MAP = {
    'ङ': 'ಙ',
    'छ': 'ಛ',
    'ज': 'ಜ',
    'ठ': 'ಠ',
    'द': 'ದ',
    'न': 'ನ',
    'म': 'ಮ',
    'र': 'ರ'
}

def extract_features(im, hands_detector):
    if im is None:
        return []
        
    th, tw = im.shape[:2]
    if th < 200 or tw < 200:
        im = cv2.resize(im, (300, 300))
        th, tw = 300, 300

    # 1. Direct RGB extraction (works on vowels and standard frames)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)
    if res.multi_hand_landmarks:
        return [compute_81_features(res.multi_hand_landmarks[0])]

    # 2. Multi-scale canvas embedding (0.35, 0.45, 0.55)
    for s in [0.35, 0.45, 0.55]:
        sw, sh = int(tw * s), int(th * s)
        scaled = cv2.resize(im, (sw, sh))
        canvas = np.full((512, 512, 3), 128, dtype=np.uint8)
        canvas[(512 - sh) // 2:(512 - sh) // 2 + sh, (512 - sw) // 2:(512 - sw) // 2 + sw] = scaled
        c_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        res = hands_detector.process(c_rgb)
        if res.multi_hand_landmarks:
            return [compute_81_features(res.multi_hand_landmarks[0])]

    # 3. Dark background neutral replacement + wrist pad
    bg_mask = (im[:, :, 0] > 200) & (im[:, :, 1] > 200) & (im[:, :, 2] > 200)
    if np.sum(bg_mask) > 0.15 * th * tw:
        dark = im.copy()
        dark[bg_mask] = [40, 40, 40]
        wrist = np.ones((80, tw, 3), dtype=np.uint8) * 140
        comb = np.vstack([dark, wrist])
        for s in [0.4, 0.5]:
            sw, sh = int(comb.shape[1] * s), int(comb.shape[0] * s)
            scaled = cv2.resize(comb, (sw, sh))
            canvas = np.full((512, 512, 3), 40, dtype=np.uint8)
            canvas[(512 - sh) // 2:(512 - sh) // 2 + sh, (512 - sw) // 2:(512 - sw) // 2 + sw] = scaled
            c_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(c_rgb)
            if res.multi_hand_landmarks:
                return [compute_81_features(res.multi_hand_landmarks[0])]

    return []

def worker(args):
    img_path, label = args
    try:
        with open(img_path, 'rb') as fp:
            im = cv2.imdecode(np.frombuffer(fp.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

    if im is None:
        return None

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.01
    )
    feats = extract_features(im, detector)
    detector.close()

    if not feats:
        return None
    return feats[0], label

def main(max_samples_per_class=60):
    print("=" * 65)
    print("  TRAINING COMPLETE HINDI MODEL ACROSS ALL 42 ALPHABET CLASSES")
    print("=" * 65)

    tasks = []
    for cls_name in ALL_HINDI_CLASSES:
        # Collect from Hindi dataset
        p_hi = os.path.join(DATA_DIR_HINDI, cls_name)
        if os.path.isdir(p_hi):
            files = [f for f in os.listdir(p_hi) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # sample up to 80 images
            indices = np.linspace(0, len(files) - 1, min(80, len(files)), dtype=int)
            for i in indices:
                tasks.append((os.path.join(p_hi, files[i]), cls_name))

        # Supplement truncated classes with ISL equivalent gestures
        if cls_name in ISL_EQUIVALENT_MAP:
            kn_char = ISL_EQUIVALENT_MAP[cls_name]
            p_kn = os.path.join(DATA_DIR_KANNADA, kn_char)
            if os.path.isdir(p_kn):
                files_kn = [f for f in os.listdir(p_kn) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                indices_kn = np.linspace(0, len(files_kn) - 1, min(60, len(files_kn)), dtype=int)
                for i in indices_kn:
                    tasks.append((os.path.join(p_kn, files_kn[i]), cls_name))

    print(f"Total scheduled image extraction tasks: {len(tasks)}")
    print("Extracting 81-dim landmark features with multi-scale canvas pipeline...")

    X_list = []
    y_labels = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = ex.map(worker, tasks)
        for r in results:
            if r is not None:
                feats, lbl = r
                X_list.append(feats)
                y_labels.append(lbl)

    print(f"Extracted {len(X_list)} valid landmark vectors.")

    counts = {}
    for lbl in y_labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    valid_classes = sorted([c for c in ALL_HINDI_CLASSES if counts.get(c, 0) >= 2])
    print(f"\nSuccessfully populated classes: {len(valid_classes)} / {len(ALL_HINDI_CLASSES)}")
    for c in ALL_HINDI_CLASSES:
        status = f"✅ ({counts.get(c, 0)} samples)" if counts.get(c, 0) >= 2 else f"❌ ({counts.get(c, 0)} samples)"
        print(f"  {c}: {status}")

    label_to_idx = {c: i for i, c in enumerate(valid_classes)}
    idx_to_label = {i: c for i, c in enumerate(valid_classes)}

    X = np.array([f for f, l in zip(X_list, y_labels) if l in label_to_idx])
    y = np.array([label_to_idx[l] for l in y_labels if l in label_to_idx])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print(f"\nTraining ExtraTreesClassifier (120 estimators) on {len(X_train)} samples across {len(valid_classes)} classes...")
    clf = ExtraTreesClassifier(
        n_estimators=120,
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
    print(f"  [SUCCESS] Hindi 42-Class Test Accuracy: {acc * 100:.2f}%")
    print(f"==================================================")

    model_path = os.path.join(MODELS_DIR, 'hindi_model.pkl')
    labels_path = os.path.join(MODELS_DIR, 'hindi_classes.json')

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': valid_classes, 'idx_to_label': idx_to_label}, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Saved complete Hindi model to '{model_path}' ({size_mb:.2f} MB).")
    print(f"Saved class mappings to '{labels_path}' ({len(valid_classes)} classes).")

if __name__ == '__main__':
    main()
