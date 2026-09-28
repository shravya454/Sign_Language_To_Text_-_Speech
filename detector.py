"""
================================================================================
MODULE: detector.py
PURPOSE: Computer Vision Hand Landmark Extraction & ML Model Inference Pipeline
TECHNOLOGIES: OpenCV, MediaPipe Hands, Scikit-Learn Classifiers
================================================================================
EXPLANATION FOR PRESENTATION:
1. Input Video Stream: Captures real-time camera frames at 30 FPS using OpenCV.
2. MediaPipe Hand Tracking: Extracts 21 3D hand keypoints (wrist, knuckles, fingertips).
3. 81-Dimensional Feature Vector:
   - 63 relative 3D coordinates normalized by palm size (wrist to middle MCP).
   - 5 fingertip-to-wrist normalized distances (Thumb, Index, Middle, Ring, Pinky).
   - 5 adjacent fingertip distances.
   - 8 fingertip-to-knuckle key structural distances.
4. Scale & Rotation Invariance: Dividing all distances by palm size makes features 
   100% invariant to hand size, distance from camera, and zoom level.
5. Temporal Confidence Smoothing: Aggregates prediction scores across a sliding window 
   buffer of recent video frames to eliminate flickering and noise.
================================================================================
"""

import os
import json
import pickle
import math
import numpy as np
import cv2
import mediapipe as mp

# ------------------------------------------------------------------------------
# 1. 81-Dimensional Scale-Invariant Geometric Feature Extractor Function
# ------------------------------------------------------------------------------
def compute_81_features(hand_landmarks):
    """
    Computes 81 scale-invariant geometric & structural distance features 
    from MediaPipe hand landmarks.
    
    Returns: 81-element list of floating point feature values.
    """
    if hasattr(hand_landmarks, 'landmark'):
        lms = []
        for lm in hand_landmarks.landmark:
            lms.extend([lm.x, lm.y, lm.z])
    elif isinstance(hand_landmarks, (list, np.ndarray)) and len(hand_landmarks) == 21:
        lms = []
        for pt in hand_landmarks:
            lms.extend([pt[0], pt[1], pt[2]])
    else:
        lms = list(hand_landmarks)
    
    # Landmark 0: Wrist Base Point
    x0, y0, z0 = lms[0], lms[1], lms[2]
    # Landmark 9: Middle Finger MCP Knuckle Joint
    x9, y9, z9 = lms[27], lms[28], lms[29]
    
    # Palm scale normalization factor = 3D distance between Wrist and Middle MCP
    palm_dist = math.sqrt((x9 - x0)**2 + (y9 - y0)**2 + (z9 - z0)**2)
    if palm_dist < 1e-5:
        palm_dist = 1.0

    # Feature Group 1: 63 Relative 3D Coordinates normalized by Palm Size
    norm_coords = []
    for i in range(0, 63, 3):
        norm_coords.append((lms[i] - x0) / palm_dist)
        norm_coords.append((lms[i+1] - y0) / palm_dist)
        norm_coords.append((lms[i+2] - z0) / palm_dist)
        
    def get_pt(idx):
        return (lms[idx*3], lms[idx*3+1], lms[idx*3+2])

    def dist3d(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2) / palm_dist

    # Feature Group 2: 5 Fingertip-to-Wrist Normalized Distances
    wrist = get_pt(0)
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky Tips
    tip_to_wrist = [dist3d(get_pt(t), wrist) for t in tips]
    
    # Feature Group 3: 5 Adjacent Fingertip Spacing Distances
    adj_pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20)]
    adj_dists = [dist3d(get_pt(p1), get_pt(p2)) for p1, p2 in adj_pairs]
    
    # Feature Group 4: 8 Fingertip to Knuckle Key Structural Distances
    mcp_pairs = [(4, 5), (4, 9), (4, 13), (4, 17), (8, 2), (12, 5), (16, 9), (20, 13)]
    mcp_dists = [dist3d(get_pt(p1), get_pt(p2)) for p1, p2 in mcp_pairs]

    # Combine all groups into final 81-dimensional vector
    features = norm_coords + tip_to_wrist + adj_dists + mcp_dists
    return features

# ------------------------------------------------------------------------------
# 2. 85-Dimensional Feature Extractor (Orientation + Finger Curl Ratios)
# ------------------------------------------------------------------------------
def compute_85_features(hand_landmarks):
    """
    Computes 85 scale-invariant features:
    - 81 base geometric features
    - 2 hand tilt orientation angles (XY plane tilt and Z elevation angle)
    - 2 finger curl ratios (Index and Middle fingers)
    
    Returns: 85-element list of floating point feature values.
    """
    base_81 = compute_81_features(hand_landmarks)

    if hasattr(hand_landmarks, 'landmark'):
        coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    elif isinstance(hand_landmarks, (list, np.ndarray)) and len(hand_landmarks) == 21:
        coords = [(pt[0], pt[1], pt[2]) for pt in hand_landmarks]
    else:
        coords = [(hand_landmarks[i], hand_landmarks[i+1], hand_landmarks[i+2]) for i in range(0, 63, 3)]

    # Point 0: Wrist Base, Point 9: Middle Finger MCP
    x0, y0, z0 = coords[0]
    x9, y9, z9 = coords[9]

    dx = x9 - x0
    dy = y9 - y0
    dz = z9 - z0
    d_xy = math.sqrt(dx**2 + dy**2)

    # Orientation Feature 82: Hand tilt angle in camera image XY plane [-1.0, 1.0]
    tilt_xy = math.atan2(dy, dx) / math.pi

    # Orientation Feature 83: Hand tilt elevation angle in Z [-1.0, 1.0]
    tilt_z = math.atan2(dz, d_xy + 1e-6) / (math.pi / 2.0)

    # Helper for Euclidean distance between landmark points
    def raw_dist3d(i, j):
        return math.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2 + (coords[i][2]-coords[j][2])**2)

    # Structural Feature 84: Index finger curl ratio (tip 8 to MCP 5 / segment length)
    seg_idx = raw_dist3d(5, 6) + raw_dist3d(6, 7) + raw_dist3d(7, 8)
    curl_idx = raw_dist3d(8, 5) / (seg_idx + 1e-5)

    # Structural Feature 85: Middle finger curl ratio (tip 12 to MCP 9 / segment length)
    seg_mid = raw_dist3d(9, 10) + raw_dist3d(10, 11) + raw_dist3d(11, 12)
    curl_mid = raw_dist3d(12, 9) / (seg_mid + 1e-5)

    return base_81 + [tilt_xy, tilt_z, curl_idx, curl_mid]

# ------------------------------------------------------------------------------
# 3. 186-Dimensional Dual-Hand Geometric & Spatial Feature Extractor (Tamil)
# ------------------------------------------------------------------------------
def compute_multihand_features(landmarks_list):
    """
    Computes 186-dimensional scale-invariant geometric & spatial relation features
    for two hands (essential for Tamil sign language fingerspelling):
    - Hand 1 (Screen-Left Hand): 85 scale-invariant features
    - Hand 2 (Screen-Right Hand): 85 scale-invariant features
    - Inter-hand spatial & structural relations (16 scale-invariant features):
        1. has_two_hands flag (1.0 or 0.0)
        2-4. relative wrist displacement vector (dx, dy, dz) normalized by palm scale
        5. wrist-to-wrist 3D Euclidean distance normalized by palm scale
        6-10. fingertip-to-fingertip distances (Thumb, Index, Middle, Ring, Pinky)
        11. Index 1 tip to Wrist 2 distance
        12. Index 2 tip to Wrist 1 distance
        13. MCP knuckle-to-knuckle distance (joint 9 to 9)
        14. Thumb 1 tip to Index 2 tip distance
        15. Index 1 tip to Thumb 2 tip distance
        16. Relative palm size ratio (p1 / p2)
    
    Total: 85 + 85 + 16 = 186 features.
    """
    if not landmarks_list:
        return [0.0] * 186
        
    def to_coords(h):
        if hasattr(h, 'landmark'):
            return [[lm.x, lm.y, lm.z] for lm in h.landmark]
        elif isinstance(h, (list, np.ndarray)) and len(h) == 21:
            return [[pt[0], pt[1], pt[2]] for pt in h]
        elif isinstance(h, (list, np.ndarray)) and len(h) == 63:
            return [[h[i], h[i+1], h[i+2]] for i in range(0, 63, 3)]
        return [[pt[0], pt[1], pt[2]] for pt in h]

    hands_coords = [to_coords(h) for h in landmarks_list if h is not None]
    if not hands_coords:
        return [0.0] * 186

    # Sort hands by screen-X coordinate of wrist (landmark 0)
    sorted_hands = sorted(hands_coords, key=lambda c: c[0][0])
    
    if len(sorted_hands) >= 2:
        c1 = sorted_hands[0]
        c2 = sorted_hands[1]
        f1 = compute_85_features(c1)
        f2 = compute_85_features(c2)
        has_two = 1.0
        
        # Reference palm scale (average palm distance of both hands)
        p1 = math.sqrt((c1[9][0]-c1[0][0])**2 + (c1[9][1]-c1[0][1])**2 + (c1[9][2]-c1[0][2])**2)
        p2 = math.sqrt((c2[9][0]-c2[0][0])**2 + (c2[9][1]-c2[0][1])**2 + (c2[9][2]-c2[0][2])**2)
        scale = (p1 + p2) / 2.0
        if scale < 1e-4:
            scale = 1.0
            
        dx = (c2[0][0] - c1[0][0]) / scale
        dy = (c2[0][1] - c1[0][1]) / scale
        dz = (c2[0][2] - c1[0][2]) / scale
        wrist_dist = math.sqrt(dx**2 + dy**2 + dz**2)
        
        tips = [4, 8, 12, 16, 20]
        tip_dists = [
            math.sqrt((c1[t][0]-c2[t][0])**2 + (c1[t][1]-c2[t][1])**2 + (c1[t][2]-c2[t][2])**2) / scale
            for t in tips
        ]
        
        cross_d1 = math.sqrt((c1[8][0]-c2[0][0])**2 + (c1[8][1]-c2[0][1])**2 + (c1[8][2]-c2[0][2])**2) / scale
        cross_d2 = math.sqrt((c2[8][0]-c1[0][0])**2 + (c2[8][1]-c1[0][1])**2 + (c2[8][2]-c1[0][2])**2) / scale
        mcp_dist = math.sqrt((c1[9][0]-c2[9][0])**2 + (c1[9][1]-c2[9][1])**2 + (c1[9][2]-c2[9][2])**2) / scale
        thumb1_idx2 = math.sqrt((c1[4][0]-c2[8][0])**2 + (c1[4][1]-c2[8][1])**2 + (c1[4][2]-c2[8][2])**2) / scale
        idx1_thumb2 = math.sqrt((c1[8][0]-c2[4][0])**2 + (c1[8][1]-c2[4][1])**2 + (c1[8][2]-c2[4][2])**2) / scale
        palm_ratio = p1 / (p2 + 1e-5)
        
        inter = [
            has_two, dx, dy, dz, wrist_dist,
            tip_dists[0], tip_dists[1], tip_dists[2], tip_dists[3], tip_dists[4],
            cross_d1, cross_d2, mcp_dist, thumb1_idx2, idx1_thumb2, palm_ratio
        ]
    else:
        # Exactly 1 hand detected
        c1 = sorted_hands[0]
        f_single = compute_85_features(c1)
        wx = c1[0][0]
        if wx >= 0.5:
            f1 = [0.0] * 85
            f2 = f_single
        else:
            f1 = f_single
            f2 = [0.0] * 85
        inter = [0.0] * 16

    feat = f1 + f2 + inter
    return feat

# ------------------------------------------------------------------------------
# 4. HandSignDetector Wrapper Class
# ------------------------------------------------------------------------------
class HandSignDetector:
    """
    Wraps MediaPipe Hands landmark detector and multi-language ML classifier models.
    Supports Kannada (85-dim), Hindi (81-dim), and Tamil (186-dim Dual-Hand) models.
    """
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}     # Loaded ML classifier models per language
        self.classes = {}    # Class label mappings per language
        
        # Initialize MediaPipe Hands solution with dual-hand support
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,      # Optimized for fast video stream tracking
            max_num_hands=2,               # Multi-hand recognition (supports single-hand & dual-hand)
            model_complexity=0,            # High-speed lightweight model for CPU real-time detection
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load trained language model classifiers (.pkl & .json)
        self._load_models()
        
        # Sliding window prediction buffer for confidence-weighted smoothing
        self.prediction_buffer = []
        self.buffer_capacity = 8
        self.no_hand_count = 0
        self.num_detected_hands = 0
        self.current_language = None
        
        # Landmark Smoothing Caches:
        # Single-hand EMA cache (Kannada, Hindi)
        self.last_landmark_coords = None
        # Dual-hand EMA cache (Tamil): [left_hand_coords, right_hand_coords]
        self.last_tamil_coords = [None, None]

    def _load_models(self):
        """Loads trained machine learning models for Kannada, Hindi, and Tamil."""
        for lang in ['kannada', 'hindi', 'tamil']:
            model_path = os.path.join(self.models_dir, f"{lang}_model.pkl")
            label_path = os.path.join(self.models_dir, f"{lang}_classes.json")
            
            if os.path.exists(model_path) and os.path.exists(label_path):
                try:
                    with open(model_path, 'rb') as f:
                        clf = pickle.load(f)
                    with open(label_path, 'r', encoding='utf-8') as f:
                        lbl_data = json.load(f)
                    self.models[lang] = clf
                    idx_map = {int(k) if k.isdigit() else k: v for k, v in lbl_data['idx_to_label'].items()}
                    self.classes[lang] = idx_map
                    n_feats = getattr(clf, 'n_features_in_', 81)
                    print(f"[MODEL LOADED] {lang.upper()}: {len(idx_map)} classes active ({n_feats}-dim features).")
                except Exception as e:
                    print(f"Error loading {lang} model: {e}")

    def extract_landmarks(self, frame, draw=True, language=None):
        """
        Extracts scale-invariant feature vector from OpenCV BGR camera frame.
        - For Tamil: Extracts 186-dim dual-hand geometric & spatial features.
        - For Kannada & Hindi: Extracts 85-dim single-hand features from active hand.
        Applies Exponential Moving Average (EMA) landmark smoothing and occlusion recovery.
        Optionally renders 21-keypoint MediaPipe skeleton overlay for ALL detected hands.
        
        Returns: (features_vector, annotated_frame)
        """
        if frame is None:
            return None, frame

        target_lang = (language or self.current_language or 'kannada').lower()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        annotated_frame = frame.copy()
        
        if not results.multi_hand_landmarks:
            self.no_hand_count += 1
            self.num_detected_hands = 0
            if self.no_hand_count >= 5:
                self.prediction_buffer.clear()
                self.last_landmark_coords = None
                self.last_tamil_coords = [None, None]
                return None, annotated_frame
            # 2-frame occlusion bridge: If hand was detected recently, reuse smoothed coords
            if self.no_hand_count <= 2:
                if target_lang == 'tamil' and (self.last_tamil_coords[0] is not None or self.last_tamil_coords[1] is not None):
                    valid_coords = [c for c in self.last_tamil_coords if c is not None]
                    features = compute_multihand_features(valid_coords)
                    return features, annotated_frame
                elif self.last_landmark_coords is not None:
                    features = compute_85_features(self.last_landmark_coords)
                    return features, annotated_frame
            return None, annotated_frame

        self.no_hand_count = 0
        self.num_detected_hands = len(results.multi_hand_landmarks)

        if draw:
            # Draw MediaPipe hand skeleton landmarks and connections for ALL detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        if target_lang == 'tamil':
            # ------------------------------------------------------------------
            # Dual-Hand Processing Pipeline for Tamil
            # ------------------------------------------------------------------
            def get_wrist_x(h):
                return h.landmark[0].x
                
            sorted_hands = sorted(results.multi_hand_landmarks, key=get_wrist_x)
            smoothed_hands = []
            
            if len(sorted_hands) >= 2:
                for idx in [0, 1]:
                    curr_c = [[lm.x, lm.y, lm.z] for lm in sorted_hands[idx].landmark]
                    prev_c = self.last_tamil_coords[idx]
                    if prev_c is not None:
                        smoothed = [
                            [0.65 * c[0] + 0.35 * p[0], 0.65 * c[1] + 0.35 * p[1], 0.65 * c[2] + 0.35 * p[2]]
                            for c, p in zip(curr_c, prev_c)
                        ]
                    else:
                        smoothed = curr_c
                    self.last_tamil_coords[idx] = smoothed
                    smoothed_hands.append(smoothed)
            else:
                curr_c = [[lm.x, lm.y, lm.z] for lm in sorted_hands[0].landmark]
                wx = curr_c[0][0]
                slot_idx = 1 if wx >= 0.5 else 0
                prev_c = self.last_tamil_coords[slot_idx]
                if prev_c is not None:
                    smoothed = [
                        [0.65 * c[0] + 0.35 * p[0], 0.65 * c[1] + 0.35 * p[1], 0.65 * c[2] + 0.35 * p[2]]
                        for c, p in zip(curr_c, prev_c)
                    ]
                else:
                    smoothed = curr_c
                self.last_tamil_coords[slot_idx] = smoothed
                self.last_tamil_coords[1 - slot_idx] = None
                smoothed_hands.append(smoothed)

            features = compute_multihand_features(smoothed_hands)
            return features, annotated_frame
        else:
            # ------------------------------------------------------------------
            # Single-Hand Processing Pipeline (Kannada & Hindi)
            # ------------------------------------------------------------------
            raw_landmarks = results.multi_hand_landmarks[0]
            curr_coords = [[lm.x, lm.y, lm.z] for lm in raw_landmarks.landmark]
            
            # 2-frame Exponential Moving Average (EMA) smoothing: alpha = 0.65
            if self.last_landmark_coords is not None:
                smoothed = [
                    [0.65 * c[0] + 0.35 * p[0], 0.65 * c[1] + 0.35 * p[1], 0.65 * c[2] + 0.35 * p[2]]
                    for c, p in zip(curr_coords, self.last_landmark_coords)
                ]
                self.last_landmark_coords = smoothed
            else:
                self.last_landmark_coords = curr_coords

            features = compute_85_features(self.last_landmark_coords)
            return features, annotated_frame

    def clear_buffer(self):
        """Resets the sliding prediction window and coordinate smoothing caches."""
        self.prediction_buffer.clear()
        self.last_landmark_coords = None
        self.last_tamil_coords = [None, None]

    def predict(self, language, features_vector, smooth=True):
        """
        Performs model inference and applies confidence-weighted temporal smoothing.
        Dynamically adapts to model input dimension (81-dim, 85-dim, or 186-dim).
        
        Returns: (best_predicted_character, confidence_score, top_3_probabilities)
        """
        language = language.lower()
        if language not in self.models:
            return None, 0.0, []

        if self.current_language != language:
            self.prediction_buffer.clear()
            self.current_language = language

        clf = self.models[language]
        class_map = self.classes[language]
        
        n_features = getattr(clf, 'n_features_in_', len(features_vector))
        feat_in = features_vector[:n_features]
        X_in = np.array([feat_in])
        
        try:
            probs = clf.predict_proba(X_in)[0]
        except Exception as e:
            return None, 0.0, []

        top_3_indices = np.argsort(probs)[::-1][:3]
        top_pred_idx = top_3_indices[0]
        
        pred_label = class_map.get(top_pred_idx, class_map.get(str(top_pred_idx), "Unknown"))
        confidence = float(probs[top_pred_idx])
        
        top_3 = []
        for idx in top_3_indices:
            lbl = class_map.get(idx, class_map.get(str(idx), "Unknown"))
            p = float(probs[idx])
            top_3.append((lbl, p))

        if not smooth:
            return pred_label, confidence, top_3

        # Add to sliding window prediction buffer
        self.prediction_buffer.append((pred_label, confidence))
        if len(self.prediction_buffer) > self.buffer_capacity:
            self.prediction_buffer.pop(0)

        # Weighted label aggregation: sum confidence scores across window
        conf_sums = {}
        counts = {}
        for lbl, c in self.prediction_buffer:
            conf_sums[lbl] = conf_sums.get(lbl, 0.0) + c
            counts[lbl] = counts.get(lbl, 0) + 1
            
        best_label = max(conf_sums.items(), key=lambda x: x[1])[0]
        smoothed_conf = conf_sums[best_label] / counts[best_label]

        return best_label, smoothed_conf, top_3
