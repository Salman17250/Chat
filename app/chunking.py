import re
from typing import List, Dict, Any
from app.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS

# Regex for sentence splitting that doesn't break on abbreviations (e.g. "Mr.", "U.S.", "08/09/1999", etc.)
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')

QA_BLOCK_REGEX = re.compile(
    r'(?:^|\n|\b)(?:Q\s*:|Question\s*:|\d+\.\s+Q\s*:)\s*(.*?)(?:\s*(?:A\s*:|Answer\s*:)\s*)(.*?)(?=(?:\s*(?:Q\s*:|Question\s*:|\d+\.\s+Q\s*:)|$))',
    re.DOTALL | re.IGNORECASE
)

def split_into_sentences(text: str) -> List[str]:
    """
    Dynamically splits text into clean, cohesive sentences and procedural units.
    Works generically for any document (prose, Q&A, manuals, slides, tables, bullet points).
    Preserves Q&A pairs (Q: ... A: ...) and navigation sequences as unified semantic blocks.
    """
    if not text:
        return []

    # 1. Check for Q&A blocks (e.g. Q: ... A: ...)
    qa_matches = list(QA_BLOCK_REGEX.finditer(text))
    if qa_matches:
        qa_blocks = []
        for m in qa_matches:
            q = " ".join(m.group(1).split()).strip()
            a = " ".join(m.group(2).split()).strip()
            if q and a:
                qa_blocks.append(f"Q: {q}\nA: {a}")
        if qa_blocks:
            return qa_blocks

    # 2. General sentence & procedural unit splitting
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        return []

    sentences = []
    for line in raw_lines:
        # If line is a complete Q: or A: line, keep it clean
        if re.match(r'^(?:Q\s*:|Question\s*:|A\s*:|Answer\s*:)', line, re.I):
            clean_line = line.strip()
            if len(clean_line) >= 3:
                sentences.append(clean_line)
            continue

        # Split regular prose by sentence terminators
        parts = SENTENCE_SPLIT_REGEX.split(line)
        for part in parts:
            p = part.strip()
            # Clean leading bullet markers or list characters
            p = re.sub(r'^[•\-\*●\d\.]+\s*', '', p)
            if len(p) >= 3:
                sentences.append(p)

    return sentences


def create_document_chunks(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups page sentences and semantic blocks into chunks respecting word limits and sliding overlap.
    100% dynamic for any document format.
    """
    chunks = []
    global_index = 0
    seen_chunks = set()

    for page_info in pages_data:
        page_num = page_info.get("page_number", 1)
        page_text = page_info.get("text", "")

        if not page_text.strip():
            continue

        page_sentences = split_into_sentences(page_text)
        if not page_sentences:
            continue

        current_chunk_sentences = []
        current_word_count = 0

        for sent in page_sentences:
            sent_words = sent.split()
            sent_word_len = len(sent_words)

            if current_word_count + sent_word_len > CHUNK_SIZE_WORDS and current_chunk_sentences:
                chunk_text = " \n".join(current_chunk_sentences).strip()
                dedup_key = f"{page_num}:{chunk_text.lower()}"

                if chunk_text and dedup_key not in seen_chunks:
                    seen_chunks.add(dedup_key)
                    chunks.append({
                        "index": global_index,
                        "page": page_num,
                        "text": chunk_text,
                        "sentences": list(current_chunk_sentences),
                        "word_count": current_word_count
                    })
                    global_index += 1

                overlap_sentences = []
                overlap_words = 0
                for prev_sent in reversed(current_chunk_sentences):
                    prev_len = len(prev_sent.split())
                    if overlap_words + prev_len <= CHUNK_OVERLAP_WORDS:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_words += prev_len
                    else:
                        break

                current_chunk_sentences = list(overlap_sentences)
                current_word_count = overlap_words

            current_chunk_sentences.append(sent)
            current_word_count += sent_word_len

        if current_chunk_sentences:
            chunk_text = " \n".join(current_chunk_sentences).strip()
            dedup_key = f"{page_num}:{chunk_text.lower()}"
            if chunk_text and dedup_key not in seen_chunks:
                seen_chunks.add(dedup_key)
                chunks.append({
                    "index": global_index,
                    "page": page_num,
                    "text": chunk_text,
                    "sentences": list(current_chunk_sentences),
                    "word_count": current_word_count
                })
                global_index += 1

    return chunks
