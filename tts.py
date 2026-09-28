"""
================================================================================
MODULE: tts.py
PURPOSE: Text-to-Speech (TTS) Synthesis Engine for Indian Languages
SUPPORTED LANGUAGES: Kannada (kn), Hindi (hi), Tamil (ta)
================================================================================
EXPLANATION FOR PRESENTATION:
1. Speech Synthesis Pipeline:
   - Takes error-corrected script sentence as input string.
   - Generates high-fidelity native Indian voice audio using gTTS (Google Text-to-Speech)
     or AI4Bharat IndicTTS APIs.
   - Plays output audio asynchronously using Pygame Mixer audio stream.
================================================================================
"""

import os
import time
import tempfile
import logging
from gtts import gTTS
import pygame

# Initialize Pygame Mixer for smooth audio playback
try:
    pygame.mixer.init()
except Exception as e:
    print(f"Pygame mixer init warning: {e}")

# Language ISO Codes mapping
LANGUAGE_CODES = {
    'kannada': 'kn',
    'hindi': 'hi',
    'tamil': 'ta'
}

class TextToSpeechEngine:
    """
    Text-to-Speech Engine synthesizing audio files for Kannada, Hindi, and Tamil.
    Includes in-memory speech caching for fast instant playback.
    """
    def __init__(self):
        self.output_dir = tempfile.gettempdir()
        self._cache = {}

    def synthesize(self, text, language='kannada'):
        """
        Synthesizes text string into MP3 audio file for specified target language.
        Uses cached audio files when available for zero-latency repeat speech.
        
        Returns: Filepath of generated audio MP3 file or None.
        """
        if not text or not text.strip():
            return None

        clean_text = text.strip()
        lang_code = LANGUAGE_CODES.get(language.lower(), 'kn')
        cache_key = (clean_text, lang_code)

        if cache_key in self._cache and os.path.exists(self._cache[cache_key]):
            return self._cache[cache_key]

        timestamp = int(time.time() * 1000)
        output_file = os.path.join(self.output_dir, f"sign_tts_{lang_code}_{timestamp}.mp3")

        try:
            # Generate high-quality native Indic voice audio
            tts = gTTS(text=clean_text, lang=lang_code, slow=False)
            tts.save(output_file)
            self._cache[cache_key] = output_file
            return output_file
        except Exception as e:
            print(f"gTTS error: {e}")
            try:
                import pyttsx3
                engine = pyttsx3.init()
                output_wav = os.path.join(self.output_dir, f"sign_tts_{timestamp}.wav")
                engine.save_to_file(clean_text, output_wav)
                engine.runAndWait()
                self._cache[cache_key] = output_wav
                return output_wav
            except Exception as ex:
                print(f"pyttsx3 error: {ex}")
                return None

    def play(self, audio_filepath):
        """Plays audio file using Pygame mixer."""
        if not audio_filepath or not os.path.exists(audio_filepath):
            return False

        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.load(audio_filepath)
                pygame.mixer.music.play()
            return True
        except Exception as e:
            print(f"Error playing audio with pygame: {e}")
            return False

def speak_text(text, language='kannada'):
    """Utility wrapper to synthesize and play audio immediately."""
    engine = TextToSpeechEngine()
    file_path = engine.synthesize(text, language)
    if file_path:
        engine.play(file_path)
    return file_path
