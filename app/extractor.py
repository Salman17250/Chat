import re
import numpy as np
from typing import List, Dict, Any, Optional
from app.config import CONFIDENCE_THRESHOLD
from app.embeddings import get_embedding_engine, get_cross_encoder_engine
from app.chunking import split_into_sentences

QUERY_STOP_WORDS = {
    "who", "is", "a", "an", "the", "tell", "me", "what", "where", "when", "why", "how",
    "which", "whose", "whom", "can", "you", "about", "give", "find", "show", "please",
    "of", "in", "to", "at", "for", "from", "by", "with", "on", "he", "his", "she", "her",
    "they", "their", "it", "its", "ka", "ki", "ke", "kya", "hai", "kaha", "kab", "kitna",
    "kitni", "h", "ko", "se", "unka", "uski", "uske", "details", "information", "info"
}

def clean_formatted_text(text: str) -> str:
    """
    Cleans document metadata artifacts and formats navigation paths into clear, natural instructions.
    Extracts the core procedural essence without trailing bullet noise or secondary artifacts.
    """
    t = text.strip()

    # 0. Clean any trailing or leading Q: / A: markers
    t = re.sub(r'^(?:A\s*:|Answer\s*:)\s*', '', t, flags=re.I)
    t = re.split(r'\n\s*(?:Q\s*:|Question\s*:|\d+\.\s+Q\s*:)', t, flags=re.I)[0].strip()

    # 1. Clean document section/header artifacts
    t = re.sub(r'\b\d+(\.\d+)*\s+(?:Page\s+Header\s+Title|Header\s+Title|Title|Section|Heading)\s*:?\s*', '', t, flags=re.I)
    t = re.sub(r'\b\d+(\.\d+)+\b', '', t) # remove stray 2.1, 3.1.2
    t = re.sub(r'\(\d+\)', '', t) # remove count numbers like (1000), (7) in headers

    # 2. Extract Navigation Path and following actions
    m = re.search(r'(?:(?:Navigation\s+Path\s*:?|Path\s*:?|Go\s+to\s*:?)\s*)([A-Za-z0-9\s_\+\(\)]+(?:\s*(?:→|->|>|»)\s*[A-Za-z0-9\s_\+\(\)]+)+)(.*)', t, re.I | re.DOTALL)
    if m:
        raw_nav = m.group(1).strip()
        rest = m.group(2).strip()
        
        # Trim words like "When clicked" accidentally captured in raw_nav
        raw_nav = re.sub(r'\s+When\s+clicked\s*:?.*$', '', raw_nav, flags=re.I).strip()
        nav_chain = re.sub(r'\s*(?:→|->|>|»)\s*', ' → ', raw_nav)
        
        # Check if rest begins with "→ Click ...", or "Creates ...", or "When clicked:"
        action_match = re.search(r'^(?:\s*→\s*)?(?:Click\s+(?:on\s+)?[\+A-Za-z0-9_\s]+?)(?=(?:When\s+clicked|Creates|Opens|Allows|\.|\n|$))', rest, re.I)
        if action_match:
            click_part = action_match.group(0).strip(' → ')
            nav_chain += f" → {click_part}"
            rest = rest[action_match.end():].strip()
            
        # Clean rest: remove "When clicked:", "screen is used to:", leading bullets, etc.
        rest_clean = re.sub(r'^(?:When\s+clicked\s*:?|Opens\s*:?|Creates\s*:?|Allows\s*:?|The\s+[A-Za-z0-9_\s]+\s+screen\s+is\s+used\s+to\s*:?|is\s+used\s+to\s*:?)\s*', '', rest, flags=re.I).strip()
        rest_clean = re.sub(r'^[•\-\*●\d\.\)\],;:\s]+', '', rest_clean).strip()
        
        # Take only the first sentence or bullet
        if '●' in rest_clean or '•' in rest_clean:
            rest_clean = re.split(r'[●•]', rest_clean)[0].strip()
        if '.' in rest_clean:
            rest_clean = rest_clean.split('.')[0].strip()
            
        # Clean trailing artifacts
        rest_clean = re.sub(r'(?:After\s+submit|Note|Tip|Notice)\s*:.*$', '', rest_clean, flags=re.I).strip()
        rest_clean = re.sub(r'[,;:\s]+$', '', rest_clean).strip()
        
        # Format the clean procedural answer
        if rest_clean:
            # If rest_clean starts with a verb, format as "to <verb>"
            m_verb = re.match(r'^(?:to\s+)?(add|create|open|log|enter|manage|view)\b\s*(.*)', rest_clean, re.I)
            if m_verb:
                verb = m_verb.group(1).lower()
                rem = m_verb.group(2).strip()
                return f"Go to {nav_chain} to {verb} {rem}."
            return f"Go to {nav_chain}. {rest_clean[0].upper() + rest_clean[1:]}."
        else:
            return f"Go to {nav_chain}."

    # 3. Clean bullet points & bracket noise for regular text
    t = re.sub(r'^[•\-\*●\d\.\)\],;]+\s*', '', t).strip()
    t = re.sub(r'\s*[●•]\s*', ', ', t)
    t = re.sub(r'\s{2,}', ' ', t)

    # Final cleanup of double punctuation & spaces
    t = re.sub(r'\s*([,\.:;])\s*', r'\1 ', t)
    t = re.sub(r'\s*:\s*:\s*', ': ', t)
    t = re.sub(r'\s{2,}', ' ', t).strip()
    t = re.sub(r'\s*\.\s*,\s*', '. ', t)
    if t and not t.endswith(('.', '!', '?')):
        t += '.'

    return t[0].upper() + t[1:] if t else ""

def extract_query_targets(query: str) -> List[str]:
    tokens = re.findall(r'[a-zA-Z0-9_\u0900-\u097F\u0A80-\u0AFF]+', query.lower())
    non_stop = [t for t in tokens if t not in QUERY_STOP_WORDS and len(t) >= 2]
    return non_stop if non_stop else tokens

def extract_subject_entity(text: str) -> str:
    """
    Extracts the main subject/entity from text or header (e.g. 'Jon Due').
    """
    first_line = text.splitlines()[0] if text else ""
    clean_line = re.sub(r'[-–—:].*$', '', first_line).strip()
    words = clean_line.split()
    if 1 <= len(words) <= 4:
        return clean_line
    match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+))\b', text)
    if match:
        return match.group(1)
    return ""

def split_clauses(sentence: str) -> List[str]:
    """
    Splits compound sentences into sub-clauses for fine-grained factual extraction.
    Preserves navigation paths and procedural step sequences intact.
    """
    if any(arrow in sentence for arrow in ["→", "->", ">", "»"]):
        return [sentence]
    raw_clauses = re.split(r'[,;]\s+(?:and\s+|but\s+|or\s+|while\s+)?', sentence)
    clauses = [c.strip() for c in raw_clauses if len(c.strip().split()) >= 3]
    return clauses if len(clauses) > 1 else [sentence]

def resolve_pronouns(text: str, subject: str) -> str:
    if not subject:
        return text
    text = re.sub(r'^(?:His|Her|Its|Their)\s+', f"{subject}'s ", text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:He|She|It|They)\s+', f"{subject} ", text, flags=re.IGNORECASE)
    return text


QA_BLOCK_REGEX = re.compile(
    r'(?:^|\n|\b)(?:Q\s*:|Question\s*:|\d+\.\s+Q\s*:)\s*(.*?)(?:\s*(?:A\s*:|Answer\s*:)\s*)(.*?)(?=(?:\s*(?:Q\s*:|Question\s*:|\d+\.\s+Q\s*:)|$))',
    re.DOTALL | re.IGNORECASE
)

class ExtractiveQAEngine:
    def __init__(self):
        self.bi_embedder = get_embedding_engine()
        self.cross_encoder = get_cross_encoder_engine()

    def extract_answer(
        self,
        query: str,
        retrieved_candidates: List[Dict[str, Any]],
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ) -> Dict[str, Any]:
        if not retrieved_candidates:
            return {
                "success": True,
                "answer": "I couldn't find that information in the document.",
                "filename": None,
                "confidence": 0.0,
                "sources": []
            }

        query_targets = extract_query_targets(query)
        target_set = set(t.lower() for t in query_targets)
        is_procedural_query = any(w in query.lower() for w in ["how", "create", "log", "add", "path", "steps", "navigate", "access", "where"])

        candidate_units = []

        for cand in retrieved_candidates:
            chunk_text = cand.get("text", "")
            page = cand.get("page", 1)
            filename = cand.get("filename", "")
            chunk_idx = cand.get("chunk_index", 0)
            subject = extract_subject_entity(chunk_text)

            # 1. Check for Q&A blocks (Q: ... A: ...)
            qa_matches = list(QA_BLOCK_REGEX.finditer(chunk_text))

            for m in qa_matches:
                q_text = " ".join(m.group(1).split()).strip()
                a_text = " ".join(m.group(2).split()).strip()
                a_clean = re.sub(r'^(?:A\s*:|Answer\s*:)\s*', '', a_text, flags=re.I).strip()
                if q_text and a_clean:
                    candidate_units.append({
                        "type": "qa_pair",
                        "question_text": q_text,
                        "eval_text": f"Question: {q_text}\nAnswer: {a_clean}",
                        "answer_text": clean_formatted_text(a_clean),
                        "sentence": clean_formatted_text(a_clean),
                        "subject": subject,
                        "page": page,
                        "filename": filename,
                        "chunk_index": chunk_idx,
                        "chunk_text": chunk_text
                    })

            # 1b. Procedural Navigation Units (Navigation Path + Action)
            nav_matches = list(re.finditer(r'(?:(?:Navigation\s+Path\s*:?|Path\s*:?|Go\s+to\s*:?)\s*)?([A-Za-z0-9\s_\+\(\)]+(?:\s*(?:→|->|>|»)\s*[A-Za-z0-9\s_\+\(\)]+)+)', chunk_text))
            for nm in nav_matches:
                start = max(0, nm.start() - 60)
                end = min(len(chunk_text), nm.end() + 150)
                snippet = chunk_text[start:end].replace('\n', ' ').strip()
                clean_p = clean_formatted_text(snippet)
                if clean_p and len(clean_p.split()) >= 4:
                    candidate_units.append({
                        "type": "procedure",
                        "eval_text": snippet,
                        "answer_text": clean_p,
                        "sentence": clean_p,
                        "subject": subject,
                        "page": page,
                        "filename": filename,
                        "chunk_index": chunk_idx,
                        "chunk_text": chunk_text
                    })

            # 2. Individual Sentences / Blocks
            raw_sents = cand.get("sentences", [])
            if not raw_sents:
                raw_sents = split_into_sentences(chunk_text)

            for s_idx, s in enumerate(raw_sents):
                s_clean = s.strip()
                s_words = s_clean.split()
                # Filter out micro-fragments / standalone headings without verbs
                if len(s_words) <= 3 and not any(a in s_clean for a in ["→", "->", ">", "is", "are", "was", "has", "have", "name", "old", "at", "by", "born"]):
                    continue

                # Check if s_clean is a Q&A pair block itself
                inner_qa = QA_BLOCK_REGEX.match(s_clean)
                if inner_qa:
                    iq = " ".join(inner_qa.group(1).split()).strip()
                    ia = " ".join(inner_qa.group(2).split()).strip()
                    ia_clean = re.sub(r'^(?:A\s*:|Answer\s*:)\s*', '', ia, flags=re.I).strip()
                    candidate_units.append({
                        "type": "qa_pair",
                        "question_text": iq,
                        "eval_text": f"Question: {iq}\nAnswer: {ia_clean}",
                        "answer_text": clean_formatted_text(ia_clean),
                        "sentence": clean_formatted_text(ia_clean),
                        "subject": subject,
                        "page": page,
                        "filename": filename,
                        "chunk_index": chunk_idx,
                        "chunk_text": chunk_text
                    })
                    continue

                clean_disp = clean_formatted_text(s_clean)
                clean_disp = re.sub(r'^(?:A\s*:|Answer\s*:)\s*', '', clean_disp, flags=re.I).strip()

                if len(clean_disp.split()) >= 4 or any(a in clean_disp for a in ["→", "->", ">", "is", "are", "was"]):
                    candidate_units.append({
                        "type": "sentence",
                        "eval_text": s_clean,
                        "answer_text": clean_disp,
                        "sentence": clean_disp,
                        "subject": subject,
                        "page": page,
                        "filename": filename,
                        "chunk_index": chunk_idx,
                        "chunk_text": chunk_text
                    })

            # 3. Sliding 2-Sentence Context Window (only for non-QA documents)
            if not qa_matches and len(raw_sents) >= 2:
                for s_i in range(len(raw_sents) - 1):
                    pair_text = f"{raw_sents[s_i]} {raw_sents[s_i+1]}".strip()
                    if len(pair_text.split()) >= 6:
                        pair_disp = clean_formatted_text(pair_text)
                        candidate_units.append({
                            "type": "window",
                            "eval_text": pair_text,
                            "answer_text": pair_disp,
                            "sentence": pair_disp,
                            "subject": subject,
                            "page": page,
                            "filename": filename,
                            "chunk_index": chunk_idx,
                            "chunk_text": chunk_text
                        })

        if not candidate_units:
            return {
                "success": True,
                "answer": "I couldn't find that information in the document.",
                "filename": None,
                "confidence": 0.0,
                "sources": []
            }

        # Deduplicate units by eval_text
        seen_eval = set()
        dedup_units = []
        for u in candidate_units:
            if u["eval_text"] not in seen_eval:
                seen_eval.add(u["eval_text"])
                dedup_units.append(u)
        candidate_units = dedup_units

        # 3. Vectorized Bi-Encoder + Semantic Pre-Scoring (< 10ms)
        eval_texts = [resolve_pronouns(u["eval_text"], u["subject"]) for u in candidate_units]
        q_texts = [resolve_pronouns(u.get("question_text", u["eval_text"]), u["subject"]) for u in candidate_units]

        embs_full = self.bi_embedder.embed_matrix(eval_texts)
        embs_q = self.bi_embedder.embed_matrix(q_texts)
        q_vec = np.array(self.bi_embedder.embed_query(query), dtype=np.float32)

        sims_full = np.dot(embs_full, q_vec)
        sims_q = np.dot(embs_q, q_vec)

        for idx, u in enumerate(candidate_units):
            bi_full = float(sims_full[idx]) if len(sims_full) > idx else 0.0
            bi_q = float(sims_q[idx]) if len(sims_q) > idx else 0.0
            bi_sim = max(bi_full, bi_q, (0.6 * bi_q + 0.4 * bi_full))

            u_lower = eval_texts[idx].lower()
            u_words = set(re.findall(r'\w+', u_lower))
            overlap = len(target_set & u_words)

            u["bi_sim"] = bi_sim
            u["overlap"] = overlap
            u["pre_score"] = (0.7 * bi_q) + (0.3 * bi_full) + (overlap * 0.15)

        candidate_units.sort(key=lambda x: x["pre_score"], reverse=True)

        # 4. Cross-Encoder on Top Candidates (evaluated on up to 35 candidates for high coverage)
        top_candidates = candidate_units[:35]
        pairs = [[query, u["eval_text"]] for u in top_candidates]
        ce_scores = self.cross_encoder.predict(pairs)

        for idx, u in enumerate(top_candidates):
            raw_ce = float(ce_scores[idx]) if len(ce_scores) > idx else -10.0
            norm_ce = 1.0 / (1.0 + np.exp(-raw_ce / 2.5))
            u["raw_ce_score"] = raw_ce
            u["combined_score"] = (0.50 * norm_ce) + (0.30 * max(0.0, u["bi_sim"])) + (0.20 * min(1.0, u["overlap"] / max(1, len(query_targets))))
            
            boost = 0.0
            if is_procedural_query:
                if u["type"] == "procedure":
                    # Reward complete hierarchical navigation paths (e.g. 2+ arrows: Module → Section → Button) over partial 1-arrow fragments
                    nav_hops = u.get("sentence", "").count("→") + u.get("sentence", "").count("->")
                    hop_bonus = 1.0 if nav_hops >= 2 else 0.0
                    boost += 2.0 + hop_bonus
                elif u["type"] == "qa_pair":
                    boost += 1.5
            u["ranking_score"] = raw_ce + boost + (u["pre_score"] * 1.5)

        top_candidates.sort(key=lambda x: x["ranking_score"], reverse=True)
        best_cand = top_candidates[0]
        best_ce = best_cand["raw_ce_score"]
        best_bi = best_cand["bi_sim"]

        # 5. Non-hallucination confidence check
        if (best_ce < -4.0 and best_cand["overlap"] < 0.5 and best_bi < 0.30) or best_cand["combined_score"] < 0.18:
            return {
                "success": True,
                "answer": "I couldn't find that information in the document.",
                "filename": best_cand["filename"],
                "confidence": round(best_cand["combined_score"], 4),
                "sources": []
            }

        final_answer = best_cand["answer_text"]
        if not final_answer.endswith(('.', '!', '?')) and not any(a in final_answer for a in ["?", "!"]):
            final_answer += '.'

        sources = [{
            "page": best_cand["page"],
            "score": round(best_cand["combined_score"], 4),
            "text": best_cand["chunk_text"]
        }]

        return {
            "success": True,
            "answer": final_answer,
            "filename": best_cand["filename"],
            "confidence": round(best_cand["combined_score"], 4),
            "sources": sources
        }

_global_extractor = None

def get_qa_engine() -> ExtractiveQAEngine:
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = ExtractiveQAEngine()
    return _global_extractor
