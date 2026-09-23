import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from app.config import OLLAMA_BASE_URL, OLLAMA_STRUCTURING_MODEL, ENABLE_LLM_STRUCTURING

def is_ollama_available() -> bool:
    """Checks if the local Ollama instance is active and responsive."""
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return res.status_code == 200
    except Exception:
        return False

def is_fragmented_page(text: str) -> bool:
    """
    Detects if a document page contains fragmented slide text, bullet lists,
    metadata headers, or navigation arrows that require AI restructuring.
    """
    if not text or len(text.strip()) < 15:
        return False
        
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False

    has_arrows = any(arrow in text for arrow in ["→", "->", ">", "»"])
    has_bullets = any(l.startswith(('●', '•', '-', '*', '1.', '2.', '3.')) for l in lines)
    has_headers = any(re.search(r'(?:navigation\s+path|when\s+clicked|after\s+submit|page\s+header\s+title|section|options\s*:)', l, re.I) for l in lines)
    avg_line_words = sum(len(l.split()) for l in lines) / max(1, len(lines))

    return has_arrows or has_bullets or has_headers or (avg_line_words < 8 and len(lines) >= 3)

def structure_page_text(text: str) -> str:
    """
    Uses local Ollama (qwen2.5:1.5b) to transform raw/fragmented text into clean,
    cohesive, informative factual and procedural sentences.
    """
    prompt = (
        "You are an expert technical documentation parser.\n"
        "Rewrite the following raw/slide document text into clean, cohesive, informative sentences.\n"
        "Strict Rules:\n"
        "1. If an arrow path is in the text (e.g. Sales -> Deals -> Create), format it as: 'Go to Sales -> Deals -> Create to [action]'.\n"
        "2. Do NOT invent navigation paths or fake numbers if no arrows exist in the raw text.\n"
        "3. Remove section numbers (e.g. 3.1, 2.1), metadata headers (e.g. 'Page Header Title:'), and parentheses counts (e.g. '(1000)').\n"
        "4. Keep all factual details, button names, and explanations intact.\n"
        "5. Output ONLY the clean factual text without introduction or commentary.\n\n"
        f"Raw Text:\n{text}\n\n"
        "Clean Text:"
    )
    
    try:
        payload = {
            "model": OLLAMA_STRUCTURING_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 250
            }
        }
        res = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=12)
        if res.status_code == 200:
            cleaned = res.json().get("response", "").strip()
            if cleaned and len(cleaned) >= 10:
                return cleaned
    except Exception:
        pass
        
    return text

def _process_single_page(page_info: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = page_info.get("text", "")
    page_num = page_info.get("page_number", 1)

    if is_fragmented_page(raw_text):
        clean_text = structure_page_text(raw_text)
        return {
            "page_number": page_num,
            "text": clean_text,
            "raw_text": raw_text
        }
    return page_info

def structure_document_pages(pages_data: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Parallel Smart Ingestion Layer:
    Processes document pages concurrently during upload for maximum speed.
    """
    if not ENABLE_LLM_STRUCTURING:
        return pages_data

    if not is_ollama_available():
        print("[!] Ollama is not running. Using standard text extraction.")
        return pages_data

    print(f"[*] Smart Ingestion: Concurrently processing {len(pages_data)} pages with {OLLAMA_STRUCTURING_MODEL} ({max_workers} workers)...")
    
    # Process only fragmented pages through thread pool
    results = [None] * len(pages_data)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_single_page, page_info): idx
            for idx, page_info in enumerate(pages_data)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = pages_data[idx]

    restructured_count = sum(1 for r, orig in zip(results, pages_data) if r.get("text") != orig.get("text"))
    print(f"[+] Smart Ingestion Complete: {restructured_count}/{len(pages_data)} pages restructured.")
    return results
