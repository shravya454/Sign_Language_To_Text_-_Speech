import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

print("Testing imports and modules...", flush=True)

from composer import ScriptComposer, compose_syllables
from corrector import IndicTextCorrector
from tts import TextToSpeechEngine

# Test Composer
print("\n--- Testing Composer ---", flush=True)
composer = ScriptComposer(language='kannada')
composer.add_token('ನ')
composer.add_token('ಆ')
composer.add_token('ವ')
composer.add_token('ಉ')
sentence = composer.compose_sentence()
print(f"Kannada Composition: 'ನ + ಆ + ವ + ಉ' => '{sentence}'")
assert sentence == 'ನಾವು', f"Expected 'ನಾವು', got '{sentence}'"
print("Explanation:", composer.explain_composition())

# Test Hindi Composer
composer.set_language('hindi')
composer.add_token('क')
composer.add_token('आ')
sentence_hi = composer.compose_sentence()
print(f"Hindi Composition: 'क + आ' => '{sentence_hi}'")

# Test Tamil Composer
composer.set_language('tamil')
composer.add_token('க')
composer.add_token('ஆ')
sentence_ta = composer.compose_sentence()
print(f"Tamil Composition: 'க + ஆ' => '{sentence_ta}'")

# Test Corrector
print("\n--- Testing Corrector ---", flush=True)
corrector = IndicTextCorrector(use_indicbart=False)
corrected_kn = corrector.correct_text("ನ ಆ ವ ಉ", language='kannada')
print(f"Correction for 'ನ ಆ ವ ಉ' => '{corrected_kn}'")

# Regression Tests: Ensure correct signs / valid words are never corrupted
print("\n--- Running Regression Tests for NLP Corrector ---", flush=True)
assert corrector.correct_text("ಶಾ", language='kannada') == "ಶಾ", "Single akshara 'ಶಾ' was corrupted!"
assert corrector.correct_text("ಮರ", language='kannada') == "ಮರ", "Word 'ಮರ' was corrupted to something else!"
assert corrector.correct_text("ನಮ", language='kannada') == "ನಮ", "Word 'ನಮ' was corrupted!"
assert corrector.correct_text("जल", language='hindi') == "जल", "Hindi 'जल' was corrupted to 'कल'!"
assert corrector.correct_text("घर", language='hindi') == "घर", "Hindi 'घर' was corrupted!"
assert corrector.correct_text("தமிழ்", language='tamil') == "தமிழ்", "Tamil 'தமிழ்' was corrupted!"
assert corrector.correct_text("ಕನನಡ", language='kannada') == "ಕನ್ನಡ", "Geminate 'ಕನನಡ' didn't resolve to 'ಕನ್ನಡ'!"
assert corrector.correct_text("ಧನಯವಾದ", language='kannada') == "ಧನ್ಯವಾದ", "Virama 'ಧನಯವಾದ' didn't resolve to 'ಧನ್ಯವಾದ'!"
assert corrector.correct_text("ಭಾಷಣ", language='kannada') == "ಭಾಷಣ", "'ಭಾಷಣ' substring was corrupted to 'ಭಾಷೆಣ'!"
assert corrector.correct_text("ನನ್ನ ಭಾಷ ಕನ್ನಡ", language='kannada') == "ನನ್ನ ಭಾಷೆ ಕನ್ನಡ", "Full phrase was not corrected!"

# Name & Matra slip corrections (e.g. Deepthi, Pooja)
print("\n--- Testing Name & Matra Slip Corrections ---", flush=True)
assert corrector.correct_text("ದಿಪ್ತೀ", language='kannada') == "ದೀಪ್ತಿ", f"Expected 'ದೀಪ್ತಿ', got '{corrector.correct_text('ದಿಪ್ತೀ', 'kannada')}'"
assert corrector.correct_text("ಪುಜಾ", language='kannada') == "ಪೂಜಾ", f"Expected 'ಪೂಜಾ', got '{corrector.correct_text('ಪುಜಾ', 'kannada')}'"
assert corrector.correct_text("ದಿಪಕ್", language='kannada') == "ದೀಪಕ್", f"Expected 'ದೀಪಕ್', got '{corrector.correct_text('ದಿಪಕ್', 'kannada')}'"
assert corrector.correct_text("दिप्ती", language='hindi') == "दीप्ति", f"Expected 'दीप्ति', got '{corrector.correct_text('दिप्ती', 'hindi')}'"
assert corrector.correct_text("திப்தீ", language='tamil') == "தீப்தி", f"Expected 'தீப்தி', got '{corrector.correct_text('திப்தீ', 'tamil')}'"
print("[ALL REGRESSION & NAME CORRECTION TESTS PASSED!]", flush=True)

# Test TTS
print("\n--- Testing TTS Engine ---", flush=True)
tts = TextToSpeechEngine()
audio_file = tts.synthesize("ನಮಸ್ಕಾರ", language='kannada')
print(f"TTS Audio file generated: {audio_file}")
assert audio_file is not None and os.path.exists(audio_file), "TTS synthesis failed"

print("\n[ALL MODULE TESTS PASSED SUCCESSFULLY!]", flush=True)
