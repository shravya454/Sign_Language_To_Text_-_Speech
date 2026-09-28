"""
================================================================================
MAIN APPLICATION: app.py
PROJECT: Multilingual Sign Language to Text & Speech System
SUPPORTED LANGUAGES: Kannada (ಕನ್ನಡ), Hindi (हिंदी), Tamil (தமிழ்)
================================================================================
EXPLANATION FOR PRESENTATION:
1. Multilingual Streamlit Interface:
   - Language selector dropdown in sidebar dynamically switches character script, 
     Abugida composition engine rules, and Text-to-Speech audio synthesizer.
2. Real-Time Dynamic Streamlit Architecture:
   - Streamlit `st.empty()` placeholders are initialized BEFORE entering the webcam loop.
   - Every camera frame processes 81-dimensional hand landmarks and instantly updates 
     the Live Detected Sign card, Confidence %, Top Probabilities, and Assembled Text box.
3. Continuous Hands-Free Auto Detection:
   - When a user holds a gesture steadily for ~0.6 seconds, the system automatically 
     appends the character into the sentence buffer and merges consonants with vowel matras 
     (e.g., ಕ + ಆ -> ಕಾ, क + आ -> का, க + ஆ -> கா) without requiring button clicks or photos.
4. IndicBART Error Correction & Speech:
   - Corrects spelling/syntactic errors and synthesizes natural audio using native voices.
================================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure UTF-8 encoding for Windows terminals & Streamlit
sys.stdout.reconfigure(encoding='utf-8')

from detector import HandSignDetector
from composer import ScriptComposer, VOWEL_TO_MATRA, CONSONANTS
from corrector import IndicTextCorrector
from tts import TextToSpeechEngine

# ------------------------------------------------------------------------------
# 1. Page Configuration & Modern Dark Slate CSS Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Multilingual Sign Language to Text & Speech",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        padding: 1.6rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px -5px rgba(67, 56, 202, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #c7d2fe;
        font-size: 1.05rem;
        margin: 0;
    }
    .stat-card {
        background: #151d30;
        border: 1px solid #23304a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .detected-char {
        font-size: 3.8rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 0.3rem 0;
        line-height: 1;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    .comp-box {
        background: #090d16;
        border-left: 4px solid #6366f1;
        padding: 1.2rem;
        border-radius: 10px;
        font-family: monospace;
        font-size: 1.4rem;
        color: #a5f3fc;
        margin: 0.8rem 0;
        min-height: 70px;
        display: flex;
        align-items: center;
    }
    .comp-box-corrected {
        border-left-color: #10b981 !important;
        color: #6ee7b7 !important;
    }
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Session State Initialization
# ------------------------------------------------------------------------------
if 'detector' not in st.session_state:
    with st.spinner("Initializing Hand Sign Detector and ML Models..."):
        st.session_state.detector = HandSignDetector()

if 'corrector' not in st.session_state:
    st.session_state.corrector = IndicTextCorrector(use_indicbart=False)

if 'tts' not in st.session_state:
    st.session_state.tts = TextToSpeechEngine()

if 'composer' not in st.session_state:
    st.session_state.composer = ScriptComposer(language='kannada')

if 'current_detected_sign' not in st.session_state:
    st.session_state.current_detected_sign = "-"

if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0

if 'top_3_preds' not in st.session_state:
    st.session_state.top_3_preds = []

if 'corrected_sentence' not in st.session_state:
    st.session_state.corrected_sentence = ""

if 'auto_commit_enabled' not in st.session_state:
    st.session_state.auto_commit_enabled = True

if 'auto_correct_enabled' not in st.session_state:
    st.session_state.auto_correct_enabled = True

def get_corrected_sentence(raw_text, lang, force_refine=False):
    if not raw_text or not raw_text.strip():
        return ""
    if st.session_state.get('auto_correct_enabled', True) or force_refine:
        return st.session_state.corrector.correct_text(raw_text, language=lang, force_refine=force_refine)
    return raw_text

if 'stability_frames' not in st.session_state:
    st.session_state.stability_frames = 5  # ~0.35s hold duration

if 'hold_tracker' not in st.session_state:
    st.session_state.hold_tracker = {'sign': None, 'count': 0}

if 'cooldown_counter' not in st.session_state:
    st.session_state.cooldown_counter = 0

if 'last_committed_notice' not in st.session_state:
    st.session_state.last_committed_notice = ""

# ------------------------------------------------------------------------------
# 3. Sidebar Navigation & Multilingual Settings
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/sign-language.png", width=65)
    st.title("Control Center")
    
    # Multilingual Selection Dropdown (Kannada, Hindi, Tamil)
    selected_language = st.selectbox(
        "🌐 Target Language",
        options=['Kannada', 'Hindi', 'Tamil'],
        index=0,
        help="Switches character script, Brahmic composition rules, and TTS synthesis."
    )
    lang_key = selected_language.lower()
    
    if st.session_state.composer.language != lang_key:
        st.session_state.composer.set_language(lang_key)

    st.markdown("---")
    st.subheader("⚡ Continuous Auto-Detection")
    st.session_state.auto_commit_enabled = st.toggle(
        "✨ Hands-Free Auto-Detection",
        value=True,
        help="Automatically appends detected hand sign to the sentence as soon as you show it."
    )
    
    hold_sec = st.slider(
        "⏱️ Hold Sensitivity (Seconds)",
        min_value=0.2,
        max_value=1.2,
        value=0.4,
        step=0.1,
        help="Duration to hold gesture to trigger continuous auto-add."
    )
    st.session_state.stability_frames = max(3, int(hold_sec * 12))

    st.markdown("---")
    st.subheader("✍️ NLP Error Correction")
    st.session_state.auto_correct_enabled = st.toggle(
        "Auto-Correct Spelling & Conjuncts",
        value=True,
        help="Applies Akshara-level NLP spelling and fingerspelling conjunct correction. Turn off if you wish to keep raw detected signs exactly as signed without any modification."
    )

    with st.expander("📖 Custom Dictionary / Names"):
        trie_obj = st.session_state.corrector.tries.get(lang_key)
        n_words = len(trie_obj.words) if trie_obj else 0
        st.caption(f"Loaded: **{n_words}** {selected_language} words & popular names.")
        new_word_input = st.text_input("Add custom name or word:", placeholder="e.g. ದೀಪ್ತಿ, ಶ್ರಾವ್ಯ...", key="custom_word_input")
        if st.button("➕ Add to Dictionary", use_container_width=True):
            if new_word_input and new_word_input.strip():
                if st.session_state.corrector.add_custom_word(new_word_input.strip(), language=lang_key):
                    st.success(f"Added '{new_word_input.strip()}' to {selected_language} dictionary!")
                    st.rerun()

    st.markdown("---")
    st.subheader("🤖 Model Status")
    loaded_models = st.session_state.detector.models
    if lang_key in loaded_models:
        num_classes = len(st.session_state.detector.classes[lang_key])
        n_feats = getattr(st.session_state.detector.models[lang_key], 'n_features_in_', 81)
        st.success(f"✅ {selected_language} Model Active")
        if lang_key == 'tamil':
            st.caption(f"Supports {num_classes} gesture classes (Dual-Hand {n_feats}-Dim Features).")
        else:
            st.caption(f"Supports {num_classes} gesture classes ({n_feats}-Dim Features).")
    else:
        st.warning(f"⚠️ {selected_language} Model loading...")

    st.markdown("---")
    input_mode = st.radio(
        "📷 Input Mode",
        options=["🟢 Live Real-Time Feed", "📸 Single Photo Snapshot", "📁 Upload Image File"],
        index=0
    )

# ------------------------------------------------------------------------------
# 4. Main Header Banner
# ------------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🤟 Multilingual Real-Time Sign Language Translator</h1>
    <p>Hands-Free Real-Time Detection • Brahmic Abugida Composition • IndicBART NLP • AI Speech Synthesis</p>
</div>
""", unsafe_allow_html=True)

# Helper function to render character card HTML
def render_char_card(char, lang_name):
    return f"""
    <div class="stat-card">
        <small style="color: #94a3b8;">Live Detected Sign</small>
        <div class="detected-char">{char}</div>
        <span style="color: #38bdf8;">Script: {lang_name}</span>
    </div>
    """

# Helper function to render confidence card HTML
def render_conf_card(conf_val, lang_name="kannada"):
    conf_pct = conf_val * 100
    color = "#4ade80" if conf_val >= 0.70 else "#f59e0b" if conf_val >= 0.45 else "#f87171"
    feat_desc = "186-Dim Dual-Hand Spatial Features" if lang_name.lower() == 'tamil' else "Scale-Invariant Geometric Features"
    return f"""
    <div class="stat-card">
        <small style="color: #94a3b8;">Detection Confidence</small>
        <h2 style="color: {color}; margin: 0.5rem 0;">{conf_pct:.1f}%</h2>
        <small style="color: #94a3b8;">{feat_desc}</small>
    </div>
    """

def render_sentence_box(text):
    disp = text if text else "<i>(Show signs to camera - sentence will build automatically)</i>"
    return f'<div class="comp-box">{disp}</div>'

def render_corrected_box(text):
    disp = text if text else "<i>(Refinement pending)</i>"
    return f'<div class="comp-box comp-box-corrected">{disp}</div>'

# ------------------------------------------------------------------------------
# 5. Dynamic Layout Setup with Streamlit Placeholders
# ------------------------------------------------------------------------------
col_left, col_right = st.columns([1.15, 0.85])

with col_left:
    st.subheader("📹 Real-Time Gesture Input Stream")
    frame_placeholder = st.empty()

with col_right:
    st.subheader("📊 Live Recognition Analytics")
    c1, c2 = st.columns(2)
    with c1:
        char_card_ph = st.empty()
    with c2:
        conf_card_ph = st.empty()
        
    top3_ph = st.empty()
    progress_ph = st.empty()
    toast_ph = st.empty()

char_card_ph.markdown(render_char_card(st.session_state.current_detected_sign, selected_language), unsafe_allow_html=True)
conf_card_ph.markdown(render_conf_card(st.session_state.current_confidence, selected_language), unsafe_allow_html=True)

st.markdown("---")

# Assembly Display Placeholders
st.subheader("🧩 Brahmic Abugida Syllable & Word Assembly Engine")

col_out1, col_out2 = st.columns(2)

with col_out1:
    st.markdown("##### 📝 Live Assembled Sentence (Raw Signs)")
    sentence_box_ph = st.empty()

with col_out2:
    st.markdown("##### ✨ NLP Corrected Sentence")
    corrected_box_ph = st.empty()

current_sentence = st.session_state.composer.compose_sentence()
sentence_box_ph.markdown(render_sentence_box(current_sentence), unsafe_allow_html=True)
corrected_box_ph.markdown(render_corrected_box(st.session_state.corrected_sentence or current_sentence), unsafe_allow_html=True)

st.write("##### 🔬 Abugida Composition Rules & History")
trace_ph = st.empty()
trace_ph.info(f"💡 **Composition Trace:** {st.session_state.composer.explain_composition()}")

# ------------------------------------------------------------------------------
# 6. Action Controls & Speech Buttons (ALWAYS VISIBLE & ACCESSIBLE)
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("⚙️ Sentence Controls & Audio Output")

col_btn0, col_btn_virama, col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(7)

with col_btn0:
    cur_sign = st.session_state.current_detected_sign
    btn_label = f"➕ Add Sign '{cur_sign}'" if cur_sign != "-" else "➕ Add Sign"
    if st.button(btn_label, use_container_width=True, type="primary"):
        if cur_sign and cur_sign != "-":
            st.session_state.composer.add_token(cur_sign)
            st.session_state.last_committed_notice = cur_sign
            raw_s = st.session_state.composer.compose_sentence()
            st.session_state.corrected_sentence = get_corrected_sentence(raw_s, lang_key)
            st.rerun()

with col_btn_virama:
    virama_map = {'tamil': '் (Pulli)', 'hindi': '् (Halant)', 'kannada': '್ (Virama)'}
    virama_char_map = {'tamil': '்', 'hindi': '्', 'kannada': '್'}
    v_text = virama_map.get(lang_key, '்')
    if st.button(f"• {v_text}", use_container_width=True, help="Attach Pulli/Virama to form half-consonant"):
        st.session_state.composer.add_token(virama_char_map.get(lang_key, '்'))
        raw_s = st.session_state.composer.compose_sentence()
        st.session_state.corrected_sentence = get_corrected_sentence(raw_s, lang_key)
        st.rerun()

with col_btn1:
    if st.button("␣ Add Space", use_container_width=True):
        st.session_state.composer.add_space()
        raw_s = st.session_state.composer.compose_sentence()
        st.session_state.corrected_sentence = get_corrected_sentence(raw_s, lang_key)
        st.rerun()

with col_btn2:
    if st.button("⌫ Backspace", use_container_width=True):
        st.session_state.composer.backspace()
        raw_s = st.session_state.composer.compose_sentence()
        st.session_state.corrected_sentence = get_corrected_sentence(raw_s, lang_key)
        st.rerun()

with col_btn3:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.composer.clear()
        st.session_state.corrected_sentence = ""
        st.rerun()

with col_btn4:
    if st.button("✨ Refine IndicBART", use_container_width=True, help="Refine sentence with NLP / IndicBART"):
        raw_text = st.session_state.composer.compose_sentence()
        if raw_text.strip():
            with st.spinner("✨ Refining sentence with NLP..."):
                corrected = get_corrected_sentence(raw_text, lang_key, force_refine=True)
                st.session_state.corrected_sentence = corrected
            st.rerun()

with col_btn5:
    if st.button("🔊 Speak Sentence", use_container_width=True):
        raw_text = st.session_state.composer.compose_sentence()
        text_to_speak = st.session_state.corrected_sentence if st.session_state.corrected_sentence else raw_text
        if text_to_speak.strip():
            with st.spinner("🔊 Synthesizing speech..."):
                audio_path = st.session_state.tts.synthesize(text_to_speak, language=lang_key)
            if audio_path and os.path.exists(audio_path):
                # Browser HTML5 autoplay only (prevents 2-sound echo issue)
                with open(audio_path, 'rb') as f:
                    st.audio(f.read(), format='audio/mp3', autoplay=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# 7. Real-Time Camera Stream / Input Mode Execution
# ------------------------------------------------------------------------------
if input_mode == "🟢 Live Real-Time Feed":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        frame_placeholder.error("❌ Unable to access webcam. Please check camera device permissions.")
    else:
        stop_button = st.button("⏹️ Stop Live Feed")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                frame_placeholder.error("Failed to read frame from webcam.")
                break
            
            # Flip horizontally for natural camera mirror display
            frame = cv2.flip(frame, 1)
            
            # Extract features & render MediaPipe skeleton (supports single-hand and dual-hand)
            features, annotated_frame = st.session_state.detector.extract_landmarks(frame, draw=True, language=lang_key)
            
            h, w, _ = annotated_frame.shape
            num_hands = getattr(st.session_state.detector, 'num_detected_hands', 0)
            
            if features is not None:
                pred_label, confidence, top_3 = st.session_state.detector.predict(lang_key, features)
                
                if pred_label and confidence >= 0.35:
                    st.session_state.current_detected_sign = pred_label
                    st.session_state.current_confidence = confidence
                    st.session_state.top_3_preds = top_3
                    
                    # Dynamically update UI cards LIVE!
                    char_card_ph.markdown(render_char_card(pred_label, selected_language), unsafe_allow_html=True)
                    conf_card_ph.markdown(render_conf_card(confidence, selected_language), unsafe_allow_html=True)
                    
                    if top_3:
                        with top3_ph.container():
                            st.write("##### 🎯 Top Probabilities")
                            for char, p in top_3:
                                st.progress(float(p), text=f"Character **'{char}'**: {p*100:.1f}%")

                    # Hold Tracker for Continuous Auto-Commit
                    ht = st.session_state.hold_tracker
                    if ht['sign'] == pred_label:
                        ht['count'] += 1
                    else:
                        ht['sign'] = pred_label
                        ht['count'] = 1

                    target_frames = st.session_state.stability_frames
                    progress_ratio = min(1.0, ht['count'] / target_frames)
                    
                    # OpenCV HUD Overlay
                    cv2.rectangle(annotated_frame, (10, 10), (w - 10, 80), (15, 23, 42), -1)
                    cv2.rectangle(annotated_frame, (10, 10), (w - 10, 80), (99, 102, 241), 2)
                    
                    hand_info = f" | Hands: {num_hands}/2" if lang_key == 'tamil' else f" | Hands: {num_hands}"
                    txt_label = f"Sign: {pred_label} ({confidence*100:.0f}%){hand_info}"
                    cv2.putText(annotated_frame, txt_label, (25, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 189, 248), 2, cv2.LINE_AA)
                    
                    bar_w = int((w - 50) * progress_ratio)
                    bar_color = (16, 185, 129) if progress_ratio >= 1.0 else (245, 158, 11)
                    cv2.rectangle(annotated_frame, (25, 60), (25 + bar_w, 70), bar_color, -1)

                    if lang_key == 'tamil' and num_hands < 2:
                        progress_ph.progress(progress_ratio, text=f"⚡ Hold steady (💡 Note: Most Tamil signs use both hands): {int(progress_ratio*100)}%")
                    else:
                        progress_ph.progress(progress_ratio, text=f"⚡ Hold to Auto-Add: {int(progress_ratio*100)}%")

                    # Auto-Commit Trigger: Appends sign into sentence buffer
                    if (st.session_state.auto_commit_enabled and 
                        ht['count'] >= target_frames and 
                        st.session_state.cooldown_counter == 0):
                        
                        st.session_state.composer.add_token(pred_label)
                        st.session_state.last_committed_notice = pred_label
                        st.session_state.cooldown_counter = 8  # Fast ~0.4 sec cooldown
                        ht['count'] = 0
                        
                        # Update Sentence & Trace Placeholders LIVE!
                        new_sent = st.session_state.composer.compose_sentence()
                        corr_sent = get_corrected_sentence(new_sent, lang_key)
                        st.session_state.corrected_sentence = corr_sent
                        sentence_box_ph.markdown(render_sentence_box(new_sent), unsafe_allow_html=True)
                        corrected_box_ph.markdown(render_corrected_box(corr_sent or new_sent), unsafe_allow_html=True)
                        trace_ph.info(f"💡 **Composition Trace:** {st.session_state.composer.explain_composition()}")
                        
                        toast_ph.success(f"✨ Auto-Added sign **'{pred_label}'** to sentence buffer!")
                        
                        cv2.rectangle(annotated_frame, (0, 0), (w, h), (0, 255, 0), 10)
                        cv2.putText(annotated_frame, f"ADDED: {pred_label}", (w//2 - 120, h//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
                else:
                    st.session_state.hold_tracker['count'] = 0
                    if lang_key == 'tamil' and num_hands == 1:
                        progress_ph.caption("✋ 1 hand detected — Show both hands for Tamil signs...")
                    else:
                        progress_ph.caption("Show sign steadily to detect...")
            else:
                st.session_state.hold_tracker['count'] = 0
                if lang_key == 'tamil':
                    no_hand_txt = "Show Both Hands for Tamil Signs"
                    wait_txt = "Waiting for Tamil sign (use both hands)..."
                else:
                    no_hand_txt = "No Hand Detected - Show Sign"
                    wait_txt = "Waiting for hand gesture..."
                cv2.putText(annotated_frame, no_hand_txt, (25, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (148, 163, 184), 2, cv2.LINE_AA)
                progress_ph.caption(wait_txt)

            if st.session_state.cooldown_counter > 0:
                st.session_state.cooldown_counter -= 1

            # Update live video frame display
            frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            time.sleep(0.005)

        cap.release()

elif input_mode == "📸 Single Photo Snapshot":
    img_file_buffer = st.camera_input("Take Photo Snapshot")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        features, annotated_frame = st.session_state.detector.extract_landmarks(cv_img, draw=True, language=lang_key)
        frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if features is not None:
            pred_label, confidence, top_3 = st.session_state.detector.predict(lang_key, features, smooth=False)
            if pred_label:
                st.session_state.current_detected_sign = pred_label
                st.session_state.current_confidence = confidence
                st.session_state.top_3_preds = top_3
                
                char_card_ph.markdown(render_char_card(pred_label, selected_language), unsafe_allow_html=True)
                conf_card_ph.markdown(render_conf_card(confidence, selected_language), unsafe_allow_html=True)
                
                if st.session_state.auto_commit_enabled:
                    st.session_state.composer.add_token(pred_label)
                    new_sent = st.session_state.composer.compose_sentence()
                    corr_sent = get_corrected_sentence(new_sent, lang_key)
                    st.session_state.corrected_sentence = corr_sent
                    sentence_box_ph.markdown(render_sentence_box(new_sent), unsafe_allow_html=True)
                    corrected_box_ph.markdown(render_corrected_box(corr_sent or new_sent), unsafe_allow_html=True)

else:
    uploaded_file = st.file_uploader("Upload sign image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        features, annotated_frame = st.session_state.detector.extract_landmarks(cv_img, draw=True, language=lang_key)
        frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if features is not None:
            pred_label, confidence, top_3 = st.session_state.detector.predict(lang_key, features, smooth=False)
            if pred_label:
                st.session_state.current_detected_sign = pred_label
                st.session_state.current_confidence = confidence
                st.session_state.top_3_preds = top_3
                
                char_card_ph.markdown(render_char_card(pred_label, selected_language), unsafe_allow_html=True)
                conf_card_ph.markdown(render_conf_card(confidence, selected_language), unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 8. Multilingual Abugida Composition Examples Expander
# ------------------------------------------------------------------------------
with st.expander("📚 View Preset Brahmic Composition Examples (Kannada / Hindi / Tamil)"):
    st.markdown("""
    #### 1. Kannada Demonstration
    - **Phrase**: `ನಾವು ನಾಳೆ ಶಾಲೆಗೆ ಹೋಗುತ್ತೇವೆ.`
    - **Step-by-step**:
      - `ನ + ಆ` = `ನಾ` | `ವ + ಉ` = `ವು` => **ನಾವು**
      - `ನ + ಆ` = `ನಾ` | `ಳ + ಎ` = `ಳೆ` => **ನಾಳೆ**
      - `ಶ + ಆ` = `ಶಾ` | `ಲ + ಎ` = `ಲೆ` | `ಗ + ಎ` = `ಗೆ` => **ಶಾಲೆಗೆ**
      - `ಹ + ಓ` = `ಹೋ` | `ಗ + ಉ` = `ಗು` | `ತ + ತ + ಏ` = `ತ್ತೇ` | `ವ + ಎ` = `ವೆ` => **ಹೋಗುತ್ತೇವೆ**

    #### 2. Hindi Demonstration (Devanagari)
    - **Phrase**: `हम कल स्कूल जाएंगे।`
    - **Step-by-step**:
      - `ह + म` => **हम**
      - `क + ल` => **कल**
      - `स + ् + क + ू + ल` => **स्कूल**
      - `ज + ा + ए + ं + ग + े` => **जाएंगे**

    #### 3. Tamil Demonstration (Tamil Uyir-Mei)
    - **Phrase**: `நாங்கள் நாளை பள்ளிக்குச் செல்வோம்.`
    - **Step-by-step**:
      - `ந + ஆ` = `நா` | `ங + ்` = `ங்` | `க + ள + ்` = `கள்` => **நாங்கள்**
      - `ந + ஆ` = `நா` | `ள + ஐ` = `ளை` => **நாளை**
      - `ப + ள + ் + ள + ி` = `பள்ளி` | `க + ு` = `கு` => **பள்ளிக்கு**
    """)
