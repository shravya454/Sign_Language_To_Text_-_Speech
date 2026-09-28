"""
================================================================================
MODULE: composer.py
PURPOSE: Brahmic Abugida Script Syllable & Word Composition Engine
SUPPORTED LANGUAGES: Kannada (ಕನ್ನಡ), Hindi (हिंदी), Tamil (தமிழ்)
================================================================================
EXPLANATION FOR PRESENTATION:
1. Indian languages use Abugida writing systems where consonants carry an 
   inherent short vowel 'a' (e.g. 'ಕ' / 'क' / 'க').
2. When a vowel modifier (matra) is added after a consonant (e.g., 'ಆ' / 'आ' / 'ஆ'), 
   the base consonant combines with the matra to form a full syllable:
   - Kannada:  ಕ (Ka) + ಆ (Aa) = ಕಾ (Kaa)
   - Hindi:    क (Ka) + आ (Aa) = का (Kaa)
   - Tamil:    க (Ka) + ஆ (Aa) = கா (Kaa)
3. Conjunct consonants (halant / virama) combine consecutive consonants:
   - Kannada:  ತ + ್ + ತ = ತ್ತ
   - Hindi:    त + ् + त = त्त
   - Tamil:    த + ் + த = த்த
================================================================================
"""

import re

# ------------------------------------------------------------------------------
# 1. Vowels to Matra (Vowel Sign) Mapping Table for Kannada, Hindi & Tamil
# ------------------------------------------------------------------------------
# Independent vowels (e.g., 'ಆ') are mapped to their corresponding dependent matras (e.g., 'ಾ').
VOWEL_TO_MATRA = {
    'kannada': {
        'ಅ': '',      # Inherited base vowel 'a' (no matra symbol needed)
        'ಆ': 'ಾ',     # U+0CBE (aa matra)
        'ಇ': 'ಿ',     # U+0CBF (i matra)
        'ಈ': 'ೀ',     # U+0CC0 (ee matra)
        'ಉ': 'ು',     # U+0CC1 (u matra)
        'ಊ': 'ೂ',     # U+0CC2 (oo matra)
        'ಋ': 'ೃ',     # U+0CC3 (ru matra)
        'ಎ': 'ೆ',     # U+0CC6 (e matra)
        'ಏ': 'ೇ',     # U+0CC7 (ee matra)
        'ಐ': 'ೈ',     # U+0CC8 (ai matra)
        'ಒ': 'ೊ',     # U+0CCA (o matra)
        'ಓ': 'ೋ',     # U+0CCB (oo matra)
        'ಔ': 'ೌ',     # U+0CCC (au matra)
        'ಅಂ': 'ಂ',    # U+0C82 (anusvara)
        'ಅಃ': 'ಃ',    # U+0C83 (visarga)
    },
    'hindi': {
        'अ': '',      # Inherited base vowel 'a'
        'आ': 'ा',     # U+093E (aa matra)
        'इ': 'ि',     # U+093F (i matra)
        'ई': 'ी',     # U+0940 (ee matra)
        'उ': 'ु',     # U+0941 (u matra)
        'ऊ': 'ू',     # U+0942 (oo matra)
        'ऋ': 'ृ',     # U+0943 (ru matra)
        'ए': 'े',     # U+0947 (e matra)
        'ऐ': 'ै',     # U+0948 (ai matra)
        'ओ': 'ो',     # U+094B (o matra)
        'औ': 'ौ',     # U+094C (au matra)
        'अं': 'ं',    # U+0902 (anusvara)
        'अः': 'ः',    # U+0903 (visarga)
    },
    'tamil': {
        'அ': '',      # Inherited base vowel 'a'
        'ஆ': 'ா',     # U+0BBE (aa matra)
        'இ': 'ி',     # U+0BBF (i matra)
        'ஈ': 'ீ',     # U+0BC0 (ee matra)
        'உ': 'ு',     # U+0BC1 (u matra)
        'ஊ': 'ூ',     # U+0BC2 (oo matra)
        'எ': 'ெ',     # U+0BC6 (e matra)
        'ஏ': 'ே',     # U+0BC7 (ee matra)
        'ஐ': 'ை',     # U+0BC8 (ai matra)
        'ஒ': 'ொ',     # U+0BCA (o matra)
        'ஓ': 'ோ',     # U+0BCB (oo matra)
        'ஔ': 'ௌ',    # U+0BCC (au matra)
    }
}

# ------------------------------------------------------------------------------
# 2. Consonant Character Sets per Language Script
# ------------------------------------------------------------------------------
CONSONANTS = {
    'kannada': set('ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲಳವಶಷಸಹ'),
    'hindi': set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षज्ञ'),
    'tamil': set('கஙசஞடணதநபமயரலவழளறன')
}

# ------------------------------------------------------------------------------
# 3. Virama / Halant Characters for Conjunct Formation
# ------------------------------------------------------------------------------
VIRAMA = {
    'kannada': '್',  # U+0CCD (Kannada Halant)
    'hindi': '्',    # U+094D (Devanagari Halant)
    'tamil': '்'     # U+0BCD (Tamil Pulli)
}

VIRAMA_TRIGGERS = {
    '್', '्', '்', 
    'virama', 'halant', 'pulli', 
    'VIRAMA', 'HALANT', 'PULLI', 
    'halant_sign', 'virama_sign', 'pulli_sign'
}

# ------------------------------------------------------------------------------
# 4. ScriptComposer Class Implementation
# ------------------------------------------------------------------------------
class ScriptComposer:
    """
    Main dynamic Abugida engine that buffers incoming hand sign tokens
    and applies language-specific vowel-consonant and selective conjunct composition rules.
    """
    def __init__(self, language='kannada'):
        self.language = language.lower()
        self.tokens = []           # Holds current composed token sequence
        self.history_steps = []    # Stores step-by-step composition trace for presentation

    def set_language(self, language):
        """Switches target script language and resets assembly buffer."""
        self.language = language.lower()
        self.clear()

    def clear(self):
        """Clears all accumulated tokens and composition history."""
        self.tokens = []
        self.history_steps = []

    def add_token(self, token):
        """
        Appends a newly detected sign token into the assembly stream.
        Applies Brahmic Abugida rules:
        1. Explicit Virama/Halant/Pulli: Attaches virama sign to preceding consonant to enable conjunct formation.
        2. Vowel/Matra Addition: Merges vowel matra with preceding consonant/syllable (e.g., ಕ + ಆ -> ಕಾ).
        3. Selective Conjunct Formation: ONLY combines consonants into an ottakshara/conjunct if 
           an explicit Virama/Halant sign is present (e.g., ತ + ್ + ತ + ಏ -> ತ್ತೇ).
        4. Default Consonant Separation: Consecutive consonants remain separate by default (e.g., ಸ + ಮ + ಯ -> ಸಮಯ).
        """
        if not token:
            return
        
        token = str(token).strip()
        if not token:
            return

        lang_vowels = VOWEL_TO_MATRA.get(self.language, {})
        lang_consonants = CONSONANTS.get(self.language, set())
        virama_char = VIRAMA.get(self.language, '')
        all_viramas = {'್', '्', '்'}

        # ----------------------------------------------------------------------
        # Rule 1: Explicit Virama / Halant / Pulli Trigger Token
        # ----------------------------------------------------------------------
        if token in VIRAMA_TRIGGERS or token in all_viramas:
            if self.tokens:
                last_token = self.tokens[-1]
                if not any(last_token.endswith(v) for v in all_viramas):
                    combined = last_token + virama_char
                    self.tokens[-1] = combined
                    self.history_steps.append(f"{last_token} + [Halant/Virama] = {combined}")
                    return
            else:
                self.tokens.append(virama_char)
                self.history_steps.append(f"Added Halant/Virama: {virama_char}")
                return

        # Collect all dependent matra characters across all languages
        all_matras = set(m for lang in VOWEL_TO_MATRA.values() for m in lang.values() if m)

        # ----------------------------------------------------------------------
        # Rule 2: Vowel / Matra Addition (Consonant + Vowel = Syllable)
        # ----------------------------------------------------------------------
        if token in lang_vowels:
            matra = lang_vowels.get(token, '')
            if self.tokens:
                last_token = self.tokens[-1]
                # If last token ends with a virama (e.g. ತ್), strip virama and add matra (e.g. ತ್ + ಏ -> ತ್ತೇ)
                if any(last_token.endswith(v) for v in all_viramas):
                    base = last_token[:-1]
                    combined = base + matra if matra else base
                    step_msg = f"{last_token} + {token} = {combined}"
                    self.tokens[-1] = combined
                    self.history_steps.append(step_msg)
                    return
                # Combine matra with last_token ONLY if last_token is a base consonant without an existing matra
                elif len(last_token) == 1 and last_token in lang_consonants:
                    if matra == '':
                        step_msg = f"{last_token} + {token} = {last_token}"
                    else:
                        combined = last_token + matra
                        step_msg = f"{last_token} + {token} = {combined}"
                        self.tokens[-1] = combined
                    self.history_steps.append(step_msg)
                    return

        # ----------------------------------------------------------------------
        # Rule 3: Selective Conjunct Formation
        # ----------------------------------------------------------------------
        if self.tokens and (token in lang_consonants or any(c in lang_consonants for c in token)):
            last_token = self.tokens[-1]
            last_char = last_token[-1] if last_token else ''
            
            # Sub-rule 3A: Explicit Virama present (e.g., ತ್ + ತ -> ತ್ತ)
            if any(last_token.endswith(v) for v in all_viramas):
                conjunct = last_token + token
                step_msg = f"{last_token} + {token} = {conjunct}"
                self.tokens[-1] = conjunct
                self.history_steps.append(step_msg)
                return
            
            # Sub-rule 3B: Identical Consecutive Consonants (Gemination, e.g., ನ + ನ -> ನ್ನ, म + म -> म्म, ள + ள -> ள்ள)
            elif (last_char in lang_consonants and token in lang_consonants) and (last_char == token or last_token == token):
                conjunct = last_token + virama_char + token
                step_msg = f"{last_token} + {token} (Identical Duplicate) = {conjunct}"
                self.tokens[-1] = conjunct
                self.history_steps.append(step_msg)
                return

        # ----------------------------------------------------------------------
        # Rule 4: Default Behavior for Consecutive Consonants
        # Keeps consecutive consonants separate (e.g., ಸ + ಮ + ಯ -> ಸಮಯ)
        # ----------------------------------------------------------------------
        self.tokens.append(token)
        self.history_steps.append(f"Added token: {token}")

    def add_space(self):
        """Appends a word boundary space."""
        self.tokens.append(' ')
        self.history_steps.append("Space added")

    def backspace(self):
        """Deletes the last assembled token from buffer."""
        if self.tokens:
            popped = self.tokens.pop()
            self.history_steps.append(f"Removed: {popped}")

    def compose_sentence(self):
        """Combines tokens into a clean formatted string sentence."""
        raw_text = "".join(self.tokens)
        clean_text = re.sub(r'\s+', ' ', raw_text)
        return clean_text

    def explain_composition(self):
        """Returns recent composition steps for visual presentation explanation."""
        if not self.history_steps:
            return "No tokens detected yet."
        return " | ".join(self.history_steps[-6:])

# ------------------------------------------------------------------------------
# 5. Helper Function for Rule Testing
# ------------------------------------------------------------------------------
def compose_syllables(char_list, language='kannada'):
    """Utility wrapper to compose a list of raw sign characters into text."""
    composer = ScriptComposer(language=language)
    for c in char_list:
        if c == ' ':
            composer.add_space()
        else:
            composer.add_token(c)
    return composer.compose_sentence()
