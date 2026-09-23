import re
from typing import Dict, Any, Optional

class UniversalConversationalSynthesizer:
    """
    Universal High-Performance Synthesizer (< 1ms).
    Transforms raw document extractions and messy OCR fragments into
    short, sweet, crystal-clear answers without markdown asterisks (**).
    100% universal algorithmic logic — zero hardcoded Q&A, zero document-specific dictionaries.
    """

    def synthesize(self, query: str, raw_answer: str, chunk_text: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        if not raw_answer or raw_answer.strip() == "I couldn't find that information in the document.":
            return raw_answer

        t = raw_answer.strip()

        # 1. Strip ALL markdown asterisks (**bold**, *italic*)
        t = t.replace('**', '').replace('*', '')

        # 2. Clean metadata headers, page labels, Q:/A: markers
        t = re.sub(r'^(?:A\s*:|Answer\s*:)\s*', '', t, flags=re.I)
        t = re.sub(r'\b\d+(\.\d+)*\s*(?:Page\s+Header\s+Title|Header\s+Title|Title|Section|Heading)\s*:?\s*', '', t, flags=re.I)
        t = re.sub(r'\b\d+(\.\d+)+\b', '', t) # remove stray section numbers like 11.11, 2.1
        t = re.sub(r'\(\d+\)', '', t) # remove stray counts

        # Clean weird OCR dot artifacts between words (e.g. "Enter. Name." -> "Enter Name.")
        t = re.sub(r'\b([A-Za-z]+)\s*\.\s*([A-Za-z]+)\b', r'\1 \2', t)

        # 3. Universal Multi-Step Detection (handles missing OCR '1.' after 'Steps:')
        # Check if text has "Steps: <Step 1 text> 2. <Step 2 text>..."
        t_normalized_steps = t
        if re.search(r'Steps\s*:\s*[^\d\s]', t, re.I):
            t_normalized_steps = re.sub(r'(Steps\s*:\s*)([^\d\s])', r'\1 1. \2', t, flags=re.I)

        parts = re.split(r'(?:^|\s+)(?=\d+[\.\)]\s+)', t_normalized_steps)
        if len(parts) >= 3 and any(k in t.lower() for k in ["step", "download", "upload", "click", "enter", "select", "submit", "fill"]):
            intro = parts[0].strip()
            # Clean intro: extract navigation arrow from intro if present
            node = r'[\w\+\(\)\-]+(?:\s+[\w\+\(\)\-]+)?'
            nav_match = re.search(rf'({node}(?:\s*(?:→|->|>|»)\s*{node})+)', intro)
            prefix = ""
            if nav_match:
                clean_nav = re.sub(r'\(\s*(\+?\s*[A-Za-z0-9_\s]+)\s*\)', r'\1', nav_match.group(1))
                clean_nav = re.sub(r'\s*(?:→|->|>|»)\s*', ' → ', clean_nav).strip()
                clean_nav = re.sub(r'\b([A-Za-z]+)\s+([A-Za-z]+)\s+\1\b', r'\1 \2', clean_nav)
                clean_nav = re.sub(r'^(?:Click\s+)+', '', clean_nav, flags=re.I)
                prefix = f"Click {clean_nav}:\n"

            cleaned_steps = []
            for p in parts[1:]:
                step_str = p.strip(" \t\r\n,-•●")
                step_str = re.sub(r'\s{2,}', ' ', step_str)
                step_str = re.sub(r'\s*\.\s*$', '', step_str)
                if step_str:
                    cleaned_steps.append(step_str)

            if len(cleaned_steps) >= 2:
                return prefix + "\n".join(cleaned_steps)

        # Clean button parentheses before regex matching: (+ Deal) -> + Deal
        t = re.sub(r'\(\s*(\+?\s*[A-Za-z0-9_\s]+)\s*\)', r'\1', t)

        # 4. Universal Bounded Navigation Path Detection
        node = r'(?:Click\s+)?[\w\+\-]+(?:\s+[\w\+\-]+)?'
        nav_pattern = rf'(?:(?:Navigation\s+Path\s*:?|Path\s*:?|Go\s+to\s*:?)\s*)?({node}(?:\s*(?:→|->|>|»)\s*{node})+)'
        m = re.search(nav_pattern, t)
        if m:
            nav_raw = m.group(1).strip()
            rest_raw = t[m.end():].strip()
            return self._extract_clean_procedural_action(nav_raw, rest_raw)

        # 5. Clean Bulleted Lists (● or •)
        if '●' in t or '•' in t:
            parts = [p.strip(" \t\r\n.,-") for p in re.split(r'[●•]', t) if len(p.strip()) >= 3]
            if len(parts) >= 2:
                return "\n".join(f"- {p}" for p in parts)

        # 6. Clean regular sentences / OCR fragments
        t = re.sub(r'^[•\-\*●\d\.\)\],;]+\s*', '', t).strip()
        t = re.sub(r'\s{2,}', ' ', t)
        t = re.sub(r'\s*([,\.:;])\s*', r'\1 ', t)
        t = re.sub(r'\s*\.\s*,\s*', '. ', t).strip()
        t = re.sub(r'\.{2,}', '.', t)
        if t and not t.endswith(('.', '!', '?')):
            t += '.'

        return t[0].upper() + t[1:] if t else ""

    def _extract_clean_procedural_action(self, nav_raw: str, rest_raw: str) -> str:
        # Clean nav: remove parentheses around buttons (+ Deal) -> + Deal
        nav = re.sub(r'\(\s*(\+?\s*[A-Za-z0-9_\s]+)\s*\)', r'\1', nav_raw)
        nav = re.sub(r'\s*(?:→|->|>|»)\s*', ' → ', nav).strip()

        # Clean duplicate trailing word (e.g. "Import Lead Import" -> "Import Lead")
        nav = re.sub(r'\b([A-Za-z]+)\s+([A-Za-z]+)\s+\1\b', r'\1 \2', nav)
        # Clean trailing prepositions accidentally captured in nav
        nav = re.sub(r'\s+(?:to|for|in|at|by)$', '', nav, flags=re.I).strip()

        rest = rest_raw.strip()

        def format_res(act: str) -> str:
            full = f"Go to {nav} {act}".strip()
            full = re.sub(r'\bto\s+to\b', 'to', full, flags=re.I)
            if not full.endswith(('.', '!', '?')):
                full += '.'
            return full

        # Priority A: "is used to: <action>"
        m_used = re.search(r'is\s+used\s+to\s*:?\s*([^\.\n●•]+)', rest, re.I)
        if m_used:
            act = m_used.group(1).strip()
            act = re.sub(r'^[•\-\*●\d\.\)\],;:\s]+', '', act)
            if act:
                return format_res(f"to {act[0].lower() + act[1:]}")

        # Priority B: "Opens / Creates / Allows <screen/form>"
        m_open = re.search(r'(?:Opens?|Creates?|Allows?)\s+([^\.\n●•]+)', rest, re.I)
        if m_open:
            act = m_open.group(1).strip()
            act = re.sub(r'^[•\-\*●\d\.\)\],;:\s]+', '', act)
            act = re.split(r'\.\s+|After\s+submission', act, flags=re.I)[0].strip()
            if act.lower().startswith("the "):
                return format_res(f"to open {act}")
            elif act:
                return format_res(f"to open the {act}")

        # Priority C: "to create / to add / to enter <entity>"
        m_to = re.search(r'to\s+(create|add|open|log|enter)\s+([^\.\n●•]+)', rest, re.I)
        if m_to:
            verb = m_to.group(1).lower()
            remainder = m_to.group(2).strip()
            remainder = re.split(r'\.\s+|After\s+submission', remainder, flags=re.I)[0].strip()
            return format_res(f"to {verb} {remainder}")

        # Priority D: Simple action clause (e.g. "Enter Name", "Log a Call")
        first_clause = re.split(r'[●•\.\n]', rest)[0].strip(" \t\r\n.,-")
        if first_clause and len(first_clause.split()) <= 5:
            m_verb = re.match(r'^(enter|add|fill|click|select|submit|log)\b\s*(.*)', first_clause, re.I)
            if m_verb:
                verb = m_verb.group(1).lower()
                rem = m_verb.group(2).strip()
                return format_res(f"to {verb} {rem}")

        return f"Go to {nav}."

_global_synthesizer = None

def get_synthesizer() -> UniversalConversationalSynthesizer:
    global _global_synthesizer
    if _global_synthesizer is None:
        _global_synthesizer = UniversalConversationalSynthesizer()
    return _global_synthesizer
