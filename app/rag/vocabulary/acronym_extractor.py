import re
from typing import Dict, List, Tuple, Set, Optional

# Minor words that may not contribute letters to an acronym
STOP_WORDS = {"of", "and", "the", "for", "in", "to", "on", "with", "at", "by", "from"}

class AcronymExtractor:
    """
    Domain-agnostic extractor for acronyms and abbreviations directly from document text.
    Extracts patterns:
      1. Full Name (ACRONYM) -> e.g. "Single Sign-On (SSO)"
      2. ACRONYM (Full Name) -> e.g. "SLA (Service Level Agreement)"
    Verifies capital letter alignment to ensure high precision without false matches.
    """

    @classmethod
    def verify_alignment(cls, phrase: str, acronym: str) -> bool:
        """
        Verifies that the acronym characters align with the initial characters
        of the words in the full phrase.
        """
        acr = acronym.upper().strip()
        if len(acr) < 2 or len(acr) > 8:
            return False

        words = [w.strip() for w in re.split(r'[\s\-]+', phrase) if w.strip()]
        if not words:
            return False

        # Filter out minor words if they don't match the current acronym character
        sig_words = [w for w in words if w.lower() not in STOP_WORDS]
        if not sig_words:
            sig_words = words

        # Check 1: First letters of significant words
        sig_initials = "".join(w[0].upper() for w in sig_words if w)
        if sig_initials == acr:
            return True

        # Check 2: First letters of all words
        all_initials = "".join(w[0].upper() for w in words if w)
        if all_initials == acr:
            return True

        # Check 3: Acronym characters appear as ordered subsequence of word initials
        acr_idx = 0
        for w in words:
            if acr_idx < len(acr) and w[0].upper() == acr[acr_idx]:
                acr_idx += 1
        if acr_idx == len(acr):
            return True

        return False

    @classmethod
    def find_aligned_phrase(cls, candidate_text: str, acronym: str) -> Optional[str]:
        """
        Extracts the minimal contiguous suffix of candidate_text that aligns with acronym.
        """
        acr = acronym.upper().strip()
        words = [w.strip() for w in re.split(r'[\s\-]+', candidate_text) if w.strip()]
        if not words or len(acr) < 2:
            return None

        # Try suffixes of length from len(acr) to min(len(words), len(acr) + 3)
        max_words = min(len(words), len(acr) + 4)
        for span_len in range(len(acr), max_words + 1):
            sub_words = words[-span_len:]
            sig_words = [w for w in sub_words if w.lower() not in STOP_WORDS]
            if not sig_words:
                continue

            sig_initials = "".join(w[0].upper() for w in sig_words if w)
            if sig_initials == acr:
                # Find where sub_words[0] begins in candidate_text
                first_word = sub_words[0]
                idx = candidate_text.rfind(first_word)
                if idx != -1:
                    return candidate_text[idx:].strip()

            all_initials = "".join(w[0].upper() for w in sub_words if w)
            if all_initials == acr:
                first_word = sub_words[0]
                idx = candidate_text.rfind(first_word)
                if idx != -1:
                    return candidate_text[idx:].strip()

        return None

    @classmethod
    def extract_from_text(cls, text: str) -> Dict[str, str]:
        """
        Extracts verified acronym mappings from text.
        Returns mapping: { "ACRONYM": "Full Phrase" }
        """
        mappings: Dict[str, str] = {}
        if not text:
            return mappings

        # Pattern 1: Full Phrase (ACRONYM)
        # e.g. "Customer Relationship Management (CRM)", "Single Sign-On (SSO)"
        p1 = re.finditer(r'([A-Za-z0-9\s\-]{2,60})\s*\(([A-Z0-9]{2,8})\)', text)
        for m in p1:
            raw_phrase = m.group(1).strip()
            acronym = m.group(2).strip()
            aligned = cls.find_aligned_phrase(raw_phrase, acronym)
            if aligned:
                mappings[acronym.upper()] = aligned

        # Pattern 2: ACRONYM (Full Phrase)
        # e.g. "SLA (Service Level Agreement)"
        p2 = re.finditer(r'\b([A-Z0-9]{2,8})\s*\(([A-Za-z0-9\s\-]{2,60})\)', text)
        for m in p2:
            acronym = m.group(1).strip()
            raw_phrase = m.group(2).strip()
            # For pattern 2, check if raw_phrase aligns from the beginning
            acr = acronym.upper()
            words = [w.strip() for w in re.split(r'[\s\-]+', raw_phrase) if w.strip()]
            sig_words = [w for w in words if w.lower() not in STOP_WORDS]
            if "".join(w[0].upper() for w in sig_words) == acr or "".join(w[0].upper() for w in words) == acr:
                mappings[acr] = raw_phrase

        return mappings
