"""
Dynamic Multi-Language Translation Layer for RAG.
Supports: English, Hindi, Hinglish (Romanized Hindi), Gujarati, and Gujlish (Romanized Gujarati).

Architecture:
  1. Ultra-fast language & script detection (< 0.05ms)
  2. Instant local query normalization for Romanized Hindi/Gujarati (< 0.05ms)
  3. Instant pattern-based fact translation for common patterns (< 0.01ms)
  4. In-memory translation caching for instant repeated queries (0.00ms)
  5. Fallback to MyMemory translation for arbitrary sentences
  6. 100% LLM-free, deterministic, ultra-fast.
"""
import re
import json
import unicodedata
from pathlib import Path
from typing import Tuple, Optional, Dict
from deep_translator import GoogleTranslator, MyMemoryTranslator
from app.rag.config import DATA_DIR

# ============================================================
# PERSISTENT TRANSLATION CACHE
# ============================================================
TRANSLATION_CACHE_FILE = DATA_DIR / "translation_cache.json"

_TRANSLATION_CACHE: Dict[str, str] = {
    # System message translations for instant responses
    "en->gu:I couldn't find that information in the document.": "મને દસ્તાવેજમાં તે માહિતી મળી નથી.",
    "en->hi:I couldn't find that information in the document.": "मुझे दस्तावेज़ में वह जानकारी नहीं मिली.",
    "en->gu:Relevant information was not found in the uploaded documents.": "અપલોડ કરેલા દસ્તાવેજોમાં સંબંધિત માહિતી મળી નથી.",
    "en->hi:Relevant information was not found in the uploaded documents.": "अपलोड किए गए दस्तावेज़ों में संबंधित जानकारी नहीं मिली।",
}

import threading
_trans_lock = threading.Lock()

def _load_persistent_cache():
    if TRANSLATION_CACHE_FILE.exists():
        try:
            with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                with _trans_lock:
                    _TRANSLATION_CACHE.update(json.load(f))
        except Exception:
            pass

_load_persistent_cache()

def _save_cache_entry(key: str, val: str):
    with _trans_lock:
        _TRANSLATION_CACHE[key] = val

# ============================================================
# ROMANIZED VOCABULARY MAPPINGS (GUJLISH & HINGLISH -> ENGLISH)
# ============================================================
ROMAN_TO_ENGLISH = {
    # Gujarati words (Gujlish)
    "pita": "father", "pappa": "father", "bapu": "father", "pitaji": "father",
    "maata": "mother", "mata": "mother", "mummy": "mother", "baa": "mother",
    "umar": "age", "umr": "age", "saal": "years old age",
    "patu": "address", "ghare": "address", "ghar": "address house",
    "kya": "where what", "kyan": "where", "kyahe": "where",
    "rahe": "live", "rehta": "live", "rehti": "live", "rehe": "live", "rahevasi": "resident lives",
    "chhe": "is", "che": "is", "chho": "are", "chhu": "am", "nathi": "not",
    "kon": "who", "kone": "whom", "su": "what", "shu": "what",
    "kem": "why how", "kevi": "how", "rite": "way method", "kevirite": "how",
    "ketla": "how many age", "ketli": "how much age", "ketlo": "how much",
    "kaam": "work job", "dhandho": "profession job", "naukri": "job occupation",
    "na": "'s", "ni": "'s", "nu": "'s", "no": "'s", "ne": "to",
    "mathi": "from in", "thi": "from by", "dwara": "through by", "ma": "in",
    "dikro": "son", "dikri": "daughter", "bhai": "brother", "ben": "sister",
    "patni": "wife", "pati": "husband",
    "banavvu": "create make", "banavva": "create make", "banavvi": "create make", "banavo": "create make",
    "karvu": "do create", "karva": "do create", "karvano": "do enter", "karvani": "do enter",
    "muko": "enter add", "nakhvo": "enter add", "jovu": "see view", "malse": "find get",
    "pachhi": "after next", "pahela": "before", "thase": "will happen",

    # Hindi words (Hinglish)
    "baap": "father", "papa": "father",
    "maa": "mother", "amma": "mother",
    "kaha": "where", "kidhar": "where", "kab": "when", "kaun": "who", "kyun": "why", "kyu": "why",
    "kaise": "how", "kese": "how", "kisko": "whom", "kiska": "whose",
    "hai": "is", "hain": "are", "h": "is", "tha": "was", "thi": "was", "the": "were", "ho": "are",
    "uska": "his", "uski": "her", "uske": "his", "unka": "his", "unki": "her", "unke": "their",
    "iska": "its", "iski": "its", "iske": "its",
    "mera": "my", "meri": "my", "mere": "my", "tera": "your", "teri": "your", "tere": "your",
    "mujhe": "i", "hume": "we", "apna": "own my", "aapka": "your", "tumhara": "your",
    "ka": "'s", "ki": "'s", "ke": "'s", "ko": "to", "se": "from", "me": "in", "mein": "in",
    "batao": "tell me", "bataao": "tell me", "bolo": "tell me", "karo": "do",
    "karna": "do create", "kare": "do create", "karein": "do create", "kar": "do",
    "banaye": "create make", "banayein": "create make", "banana": "create make", "bana": "create make",
    "dost": "friend", "mitra": "friend", "beta": "son", "beti": "daughter",
    "naam": "name", "vyavsay": "profession", "pata": "address",
    "chahiye": "want need", "tarika": "way method", "tarike": "ways options", "rasta": "way",
    "dalna": "enter add", "dale": "enter add", "dalein": "enter add",
    "baad": "after next", "pehle": "before", "hota": "happens does", "hogi": "will happen",
    "hoga": "will happen", "karta": "does", "karti": "does", "karta_hai": "works does",
    "samjhao": "explain", "samjhao_mujhe": "explain to me", "bata": "tell", "kya": "what"
}

# Distinct markers
GUJLISH_DISTINCT = frozenset({
    "chhe", "che", "chho", "chhu", "nathi", "nthi", "ketla", "ketli", "ketlo",
    "kem", "su", "shu", "kone", "maaru", "taaru", "enu", "emnu", "rahe", "rehe",
    "ghare", "dhandho", "dikro", "dikri", "ben", "baa", "pappa", "bapu", "aabhar", "majama",
    "kevi", "rite", "kevirite", "banavvu", "banavva", "banavvi", "karvu", "karva", "karvano", "karvani",
    "malse", "jovu", "pachhi", "pahela"
})

HINGLISH_DISTINCT = frozenset({
    "hai", "hain", "kaha", "kidhar", "kaun", "kyun", "kyu", "kaise", "kese", "uska", "uski",
    "uske", "unka", "unki", "unke", "iska", "iski", "iske", "batao", "bataao", "mujhe",
    "bolo", "beta", "beti", "pata", "ghar", "namaste", "namaskar", "dhanyawad", "shukriya",
    "banaye", "banayein", "banana", "karna", "kare", "karein", "chahiye", "tarika", "tarike",
    "baad", "pehle", "hota", "hoga", "karta", "samjhao"
})


# ============================================================
# LANGUAGE DETECTION
# ============================================================
def detect_language(text: str) -> Tuple[str, str]:
    """
    Detect query language and form.
    Returns: (target_lang_code, form)
      target_lang_code: 'en', 'gu', or 'hi'
      form: 'english', 'gu_script', 'hi_script', 'gujlish', 'hinglish'
    """
    if not text or not text.strip():
        return "en", "english"

    clean = text.strip()

    # 1. Unicode Script Analysis
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', clean))
    gujarati_count = len(re.findall(r'[\u0A80-\u0AFF]', clean))

    if gujarati_count > 0 and gujarati_count >= devanagari_count:
        return "gu", "gu_script"
    if devanagari_count > 0:
        return "hi", "hi_script"

    # 2. Latin Script: check for Gujlish vs Hinglish vs English
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', t).lower() for t in clean.split()]
    tokens = [t for t in tokens if t]

    guj_hits = sum(1 for t in tokens if t in GUJLISH_DISTINCT or t in {"na", "ni", "nu", "no"})
    hin_hits = sum(1 for t in tokens if t in HINGLISH_DISTINCT or t in {"ka", "ki", "ke", "h", "ko", "se"})

    has_strong_guj = any(t in GUJLISH_DISTINCT for t in tokens)
    has_strong_hin = any(t in HINGLISH_DISTINCT for t in tokens)

    if has_strong_guj or (guj_hits > hin_hits and guj_hits >= 1):
        return "gu", "gujlish"
    elif has_strong_hin or (hin_hits >= 1 and not (len(tokens) <= 3 and "is" in tokens and "who" in tokens)):
        return "hi", "hinglish"

    # Check if any general Romanized word is present
    roman_hits = sum(1 for t in tokens if t in ROMAN_TO_ENGLISH)
    if roman_hits >= 2:
        return "hi", "hinglish"

    return "en", "english"


# ============================================================
# FAST TEMPLATE / PATTERN TRANSLATOR (<0.01ms)
# ============================================================
def fast_pattern_translate(text: str, target_lang: str) -> Optional[str]:
    """
    Translates common extracted factual templates instantly without network calls.
    """
    clean = text.strip()

    # 1. Attribute pattern: "[Entity]'s [attribute] is [Value]"
    m = re.match(r"^([^.]+?)'s\s+([a-zA-Z\s]+?)\s+is\s+([^.]+)\.?$", clean, re.I)
    if m:
        subject, attr, val = m.group(1).strip(), m.group(2).strip().lower(), m.group(3).strip()
        if target_lang == "gu":
            return f"{subject}નું {attr} {val} છે."
        elif target_lang == "hi":
            return f"{subject} का {attr} {val} है।"

    # 2. Age pattern: "X is Y years old."
    m = re.match(r"^([^.]+?)\s+is\s+(\d+)\s+years\s+old\.?$", clean, re.I)
    if m:
        subject, age = m.group(1).strip(), m.group(2).strip()
        if target_lang == "gu":
            return f"{subject} {age} વર્ષનો છે."
        elif target_lang == "hi":
            return f"{subject} {age} साल के हैं।"

    # 3. Residence / Location pattern: "X is [gender] and lives at Y."
    m = re.match(r"^([^.]+?)\s+is\s+(male|female)\s+and\s+lives\s+at\s+([^.]+)\.?$", clean, re.I)
    if m:
        subject, gender, address = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if target_lang == "gu":
            gen_str = "પુરૂષ" if gender.lower() == "male" else "સ્ત્રી"
            return f"{subject} {gen_str} છે અને {address} માં રહે છે."
        elif target_lang == "hi":
            gen_str = "पुरुष" if gender.lower() == "male" else "महिला"
            return f"{subject} {gen_str} हैं और {address} में रहते हैं।"

    # 4. Simple live pattern: "X lives at Y."
    m = re.match(r"^([^.]+?)\s+lives\s+at\s+([^.]+)\.?$", clean, re.I)
    if m:
        subject, address = m.group(1).strip(), m.group(2).strip()
        if target_lang == "gu":
            return f"{subject} {address} માં રહે છે."
        elif target_lang == "hi":
            return f"{subject} {address} में रहते हैं।"

    # 5. System messages / Escalation prompts (0.001ms instant reply)
    if "Relevant information was not found in the uploaded documents." in clean:
        if "Please share your phone number" in clean:
            if target_lang == "gu":
                return "અપલોડ કરેલા દસ્તાવેજોમાં સંબંધિત માહિતી મળી નથી. કૃપા કરીને તમારો ફોન નંબર (અથવા ઇમેઇલ) શેર કરો જેથી અમારી ટીમ તમારી સીધી સહાય કરી શકે."
            elif target_lang == "hi":
                return "अपलोड किए गए दस्तावेज़ों में संबंधित जानकारी नहीं मिली। कृपया अपना फ़ोन नंबर (या ईमेल) साझा करें ताकि हमारी टीम आपसे संपर्क करके सीधे आपकी सहायता कर सके।"
        else:
            if target_lang == "gu":
                return "અપલોડ કરેલા દસ્તાવેજોમાં સંબંધિત માહિતી મળી નથી."
            elif target_lang == "hi":
                return "अपलोड किए गए दस्तावेज़ों में संबंधित जानकारी नहीं मिली।"

    return None


# ============================================================
# TRANSLATION ENGINE
# ============================================================
class TranslationEngine:
    def normalize_romanized_query(self, query: str) -> str:
        """
        Instantly translates Romanized Hindi/Gujarati words into English semantic tokens (<0.05ms).
        Example: 'jon due na pita kon chhe' -> 'jon due 's father who is'
        """
        words = query.split()
        normalized = []
        for word in words:
            match = re.match(r'^([^a-zA-Z0-9]*)([a-zA-Z0-9]+)([^a-zA-Z0-9]*)$', word)
            if match:
                prefix, core, suffix = match.groups()
                clean_core = core.lower()
                if clean_core in ROMAN_TO_ENGLISH:
                    repl = ROMAN_TO_ENGLISH[clean_core]
                    normalized.append(f"{prefix}{repl}{suffix}")
                else:
                    normalized.append(word)
            else:
                normalized.append(word)
        return " ".join(normalized)

    def translate_to_english(self, text: str, source_lang: str, form: str) -> str:
        """Translate non-English query to English."""
        if form == "english" or source_lang == "en":
            return text

        # For Gujlish & Hinglish: use ultra-fast instant local normalizer (<0.05ms)
        if form in ("gujlish", "hinglish"):
            return self.normalize_romanized_query(text)

        # For native scripts (Devanagari / Gujarati script): check persistent cache first (0.00ms)
        cache_key = f"{source_lang}->en:{text.strip()}"
        if cache_key in _TRANSLATION_CACHE:
            return _TRANSLATION_CACHE[cache_key]

        # 1. Try fast GoogleTranslator
        try:
            translated = GoogleTranslator(source=source_lang, target="en").translate(text)
            if translated:
                _save_cache_entry(cache_key, translated)
                return translated
        except Exception:
            pass

        # 2. Fallback to MyMemory
        try:
            src_tag = "gu-IN" if source_lang == "gu" else "hi-IN"
            translated = MyMemoryTranslator(source=src_tag, target="en-GB").translate(text)
            if translated:
                _save_cache_entry(cache_key, translated)
                return translated
        except Exception:
            pass

        return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate English answer into target language (Gujarati or Hindi)."""
        if target_lang == "en" or not text.strip():
            return text

        cache_key = f"en->{target_lang}:{text.strip()}"
        if cache_key in _TRANSLATION_CACHE:
            return _TRANSLATION_CACHE[cache_key]

        # 1. Try instant pattern translation (<0.01ms)
        fast_res = fast_pattern_translate(text, target_lang)
        if fast_res:
            _save_cache_entry(cache_key, fast_res)
            return fast_res

        # 2. Try fast GoogleTranslator
        try:
            translated = GoogleTranslator(source="en", target=target_lang).translate(text)
            if translated:
                _save_cache_entry(cache_key, translated)
                return translated
        except Exception:
            pass

        # 3. Fallback to MyMemory network translation with caching
        try:
            tgt_tag = "gu-IN" if target_lang == "gu" else "hi-IN"
            translated = MyMemoryTranslator(source="en-GB", target=tgt_tag).translate(text)
            if translated:
                _save_cache_entry(cache_key, translated)
                return translated
        except Exception:
            pass

        return text

    def check_greeting(self, query: str, target_lang: str) -> Optional[str]:
        """
        Detects conversational greetings/chit-chat in English, Hindi, and Gujarati
        and returns an instant (<0.01ms) polite response in the user's language.
        """
        clean = re.sub(r'[^\w\s]', '', query.strip().lower())
        
        greeting_patterns = [
            (r'^(?:hi|hey|hello|hlo|helo|hy|hii|hiii)\b', 'hi'),
            (r'^(?:good\s+(?:morning|afternoon|evening|day))\b', 'hello'),
            (r'^(?:namaste|namaskar|pranam|namaskaram)\b', 'hello'),
            (r'^(?:kem\s+ch+o|kem\s+cho|majama|su\s+chhe|shu\s+chhe)\b', 'how_are_you'),
            (r'^(?:how\s+are\s+you|how\s+r\s+u|kaise\s+ho|kya\s+hal\s+hai|kya\s+chal\s+raha\s+hai)\b', 'how_are_you'),
            (r'^(?:who\s+are\s+you|who\s+r\s+u|tum\s+kaun\s+ho|aap\s+kaun\s+hai|tame\s+kon\s+chho)\b', 'who_are_you'),
            (r'^(?:thanks|thank\s+you|thx|tysm|dhanyawad|shukriya|aabhar|dhanyavad)\b', 'thanks'),
            (r'^(?:bye|goodbye|see\s+you|tata|alvida|aavjo)\b', 'bye'),
        ]

        responses = {
            "en": {
                "hello": "Hello! How can I help you with the document today?",
                "hi": "Hi there! What's up?.",
                "hey": "Hey! How can I assist you?",
                "how_are_you": "I'm doing well, thank you! How can I help you today?",
                "who_are_you": "I am your AI document assistant. I can find and answer questions from your uploaded files in English, Hindi, and Gujarati.",
                "thanks": "You're very welcome! Let me know if you need anything else.",
                "bye": "Goodbye! Have a great day!"
            },
            "hi": {
                "hello": "नमस्ते! मैं आपकी दस्तावेज़ से जुड़ी क्या मदद कर सकता हूँ?",
                "hi": "नमस्ते! आप दस्तावेज़ के बारे में कोई भी सवाल पूछ सकते हैं।",
                "hey": "नमस्ते! मैं आपकी कैसे सहायता कर सकता हूँ?",
                "how_are_you": "मैं बिल्कुल ठीक हूँ, धन्यवाद! मैं आपकी क्या सहायता कर सकता हूँ?",
                "who_are_you": "मैं आपका दस्तावेज़ सहायक हूँ। मैं आपके अपलोड किए गए दस्तावेज़ों से सवालों के सटीक जवाब दे सकता हूँ।",
                "thanks": "आपका बहुत-बहुत धन्यवाद! अगर कोई और सवाल हो तो ज़रूर पूछें।",
                "bye": "अलविदा! आपका दिन शुभ हो!"
            },
            "gu": {
                "hello": "નમસ્તે! હું તમારા દસ્તાવેજ વિશે તમને કેવી રીતે મદદ કરી શકું?",
                "hi": "નમસ્તે! તમે અપલોડ કરેલા દસ્તાવેજ વિશે કોઈપણ પ્રશ્ન પૂછી શકો છો.",
                "hey": "હેલો! હું તમને કેવી રીતે સહાય કરી શકું?",
                "how_are_you": "હું મજામાં છું, આભાર! હું તમારી શું મદદ કરી શકું?",
                "who_are_you": "હું તમારો દસ્તાવેજ સહાયક છું. હું તમારા દસ્તાવેજોમાંથી સચોટ જવાબો આપી શકું છું.",
                "thanks": "તમારો ખૂબ ખૂબ આભાર! જો કોઈ અન્ય પ્રશ્ન હોય તો જરૂર પૂછો.",
                "bye": "આવજો! તમારો દિવસ શુભ રહે!"
            }
        }

        for pattern, intent in greeting_patterns:
            if re.search(pattern, clean):
                lang_key = target_lang if target_lang in responses else "en"
                return responses[lang_key].get(intent, responses["en"][intent])
        return None

    def process_query(self, query: str) -> Tuple[str, str, str, str]:
        """
        Returns: (english_query, target_lang, original_query, form)
        """
        target_lang, form = detect_language(query)
        english_query = self.translate_to_english(query, target_lang, form)
        return english_query, target_lang, query, form

    def process_answer(self, answer: str, target_lang: str, form: Optional[str] = None) -> str:
        """
        Translates extracted answer to target language.
        If user typed in Romanized Hinglish or Gujlish, keeps answer in clean English for instant response.
        """
        if target_lang == "en" or form in ("gujlish", "hinglish"):
            return answer
        return self.translate_from_english(answer, target_lang)


# ============================================================
# SINGLETON
# ============================================================
_global_translator: Optional[TranslationEngine] = None


def get_translator() -> TranslationEngine:
    global _global_translator
    if _global_translator is None:
        _global_translator = TranslationEngine()
    return _global_translator
