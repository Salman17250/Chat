import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    # Normalize unicode (accents, curly quotes, special spaces)
    text = unicodedata.normalize("NFKC", text)
    
    # Replace smart quotes and special hyphens
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    
    # Fix hyphenated words broken across lines (e.g. "docu-\nment" -> "document")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    raw_lines = [re.sub(r'[ \t]+', ' ', l).strip() for l in text.splitlines() if l.strip()]
    lines = []
    
    for line in raw_lines:
        if not lines:
            lines.append(line)
            continue
        prev = lines[-1]
        
        # Check if prev line looks like a title/header (short, has dash/colon, <= 4 words, or ends in colon)
        is_title = bool(re.search(r'[-–—:]\s*[A-Za-z\s]+$', prev)) or len(prev.split()) <= 4 or prev.endswith(':')
        
        # If prev line does not end in sentence-ending punctuation (.!?), and is not a header, join continuation
        if not re.search(r'[.!?]$', prev) and not is_title:
            lines[-1] = f"{prev} {line}"
        else:
            lines.append(line)
            
    return "\n".join(lines).strip()

def extract_pdf_pages(file_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(file_path))
    pages_data = []
    
    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
            
        cleaned = clean_text(raw_text)
        if cleaned:
            pages_data.append({
                "page_number": page_idx,
                "text": cleaned
            })
            
    return pages_data

def extract_pdf_full_text(file_path: Path) -> str:
    pages = extract_pdf_pages(file_path)
    return "\n\n".join([p["text"] for p in pages])
