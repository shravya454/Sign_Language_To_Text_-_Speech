import os
import sys
import json
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

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
    except Exception as e:
        return None

def extract_landmark_features(img, hands_detector):
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    # Normalize relative to wrist (landmark 0)
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

def train_language_model(language_name, max_samples_per_class=100):
    print(f"\n==========================================")
    print(f"  Training Model for Language: {language_name.upper()}")
    print(f"==========================================")
    
    lang_path = os.path.join(DATA_DIR, language_name)
    if not os.path.exists(lang_path):
        print(f"Error: Path {lang_path} does not exist.")
        return
    
    classes = sorted([d for d in os.listdir(lang_path) if os.path.isdir(os.path.join(lang_path, d))])
    print(f"Found {len(classes)} classes for {language_name}.")
    
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)
    
    X = []
    y = []
    
    label_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_label = {i: cls_name for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_path = os.path.join(lang_path, cls_name)
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Subsample if files exceed max_samples_per_class
        if len(files) > max_samples_per_class:
            # Deterministic subset for reproducibility
            indices = np.linspace(0, len(files) - 1, max_samples_per_class, dtype=int)
            files = [files[i] for i in indices]
            
        success_count = 0
        for f in files:
            img_path = os.path.join(cls_path, f)
            img = cv2_imread_unicode(img_path)
            feats = extract_landmark_features(img, hands_detector)
            if feats is not None:
                X.append(feats)
                y.append(label_to_idx[cls_name])
                success_count += 1
                
        print(f"Class '{cls_name}': Extracted {success_count}/{len(files)} landmark samples.")
    
    hands_detector.close()
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTotal extracted dataset shape: X={X.shape}, y={y.shape}")
    if len(X) == 0:
        print(f"Error: No features extracted for {language_name}.")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    print(f"Training Random Forest Classifier on {len(X_train)} training samples...")
    clf = RandomForestClassifier(n_estimators=120, max_depth=25, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"--> Test Accuracy for {language_name.upper()}: {acc * 100:.2f}%")
    
    # Save model and mapping
    model_path = os.path.join(MODELS_DIR, f"{language_name}_model.pkl")
    labels_path = os.path.join(MODELS_DIR, f"{language_name}_classes.json")
    
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
        
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({'classes': classes, 'idx_to_label': idx_to_label}, f, ensure_ascii=False, indent=2)
        
    print(f"Saved model to '{model_path}' and classes to '{labels_path}'.")

if __name__ == '__main__':
    train_language_model('kannada', max_samples_per_class=100)
    train_language_model('hindi', max_samples_per_class=80)
    train_language_model('tamil', max_samples_per_class=50)
    print("\nAll models trained and saved successfully!")
