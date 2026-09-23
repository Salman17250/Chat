import re
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader
try:
    import docx
except ImportError:
    docx = None

from app.rag.ingestion.models import DocumentBlock, ParsedDocument

def normalize_text(text: str) -> str:
    """Normalizes Unicode characters, curly quotes, and linebreaks."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    # Fix hyphenated line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text.strip()

def detect_question_structure(line: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
    """
    Generic, domain-agnostic question structure detector.
    Detects numbered questions, Q-prefixed items, or standalone questions.
    Returns: (is_question, question_number, question_text, inline_answer)
    """
    l = line.strip()
    if not l or len(l) > 250:
        return False, None, None, None

    # 1. Numbered or Q-prefixed question: e.g. "1. What is X?", "Q2: How does Y work?", "120. Why ...?"
    # Supports optional inline answer following the question mark
    m = re.match(
        r'^(?:(?:Q(?:uestion)?\s*)?([0-9]+)[\.\:\)]\s*|(?:Q(?:uestion)?\s*[:\.\-]\s*))([^\n\.\?!]+?\?)(?:\s+(.+))?$',
        l,
        re.IGNORECASE
    )
    if m:
        q_num = int(m.group(1)) if m.group(1) else None
        q_text = m.group(2).strip()
        ans_inline = m.group(3).strip() if m.group(3) else None
        return True, q_num, q_text, ans_inline

    # 2. Standalone question starting with capital letter and ending with '?'
    # Length between 8 and 140 chars, max 25 words
    if l.endswith('?') and 8 <= len(l) <= 140 and len(l.split()) <= 25:
        if re.match(r'^[A-Z][^\n\.\?!]+?\?$', l):
            return True, None, l, None

    return False, None, None, None

def is_likely_heading(line: str) -> bool:
    """Detects if a single line is likely a section heading."""
    line = line.strip()
    if not line or len(line) > 100:
        return False
    # If line is a question, question detector handles it
    is_q, _, _, _ = detect_question_structure(line)
    if is_q:
        return False
    # Numbered heading pattern (e.g. "1. Introduction", "2.1 Quotation Module", "Section 3:")
    if re.match(r'^(?:[0-9]+(?:\.[0-9]+)*[\.\:\)]?\s+[A-Z]|Section\s+[0-9]+|Chapter\s+[0-9]+|Part\s+[0-9]+|Article\s+[0-9]+)', line, re.I):
        return True
    # All caps short line (e.g. "LEAD MANAGEMENT", "TERMS AND CONDITIONS")
    if line.isupper() and 2 <= len(line.split()) <= 7:
        return True
    # Line ends with colon and <= 6 words
    if line.endswith(':') and len(line.split()) <= 6:
        return True
    return False

class DocumentParser:
    """
    Unified, structure-aware document parser.
    Supports PDF, DOCX, and DOC formats.
    Preserves document name, page numbers, headings, sections, paragraphs, tables, and Q&A pairs.
    """

    def parse(
        self,
        file_path: Path,
        tenant_id: str = "default",
        user_id: str = "default"
    ) -> ParsedDocument:
        suffix = file_path.suffix.lower()
        content_bytes = file_path.read_bytes()
        file_hash = hashlib.sha256(content_bytes).hexdigest()
        doc_name = file_path.name
        doc_id = f"doc_{file_hash[:12]}"

        if suffix == ".pdf":
            blocks, total_pages = self._parse_pdf(file_path)
        elif suffix == ".docx":
            blocks, total_pages = self._parse_docx(file_path)
        elif suffix == ".doc":
            blocks, total_pages = self._parse_doc(file_path)
        else:
            # Fallback plain text / markdown
            blocks, total_pages = self._parse_text(file_path)

        return ParsedDocument(
            document_id=doc_id,
            document_name=doc_name,
            total_pages=total_pages,
            blocks=blocks,
            file_hash=file_hash,
            tenant_id=tenant_id,
            user_id=user_id
        )

    def _parse_pdf(self, file_path: Path) -> (List[DocumentBlock], int):
        reader = PdfReader(str(file_path))
        blocks: List[DocumentBlock] = []
        total_pages = len(reader.pages)
        current_section = "General"

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""

            cleaned = normalize_text(raw_text)
            if not cleaned:
                continue

            lines = cleaned.splitlines()
            buffer_paragraph = []

            for line in lines:
                l_str = line.strip()
                if not l_str:
                    if buffer_paragraph:
                        p_text = " ".join(buffer_paragraph).strip()
                        if p_text:
                            blocks.append(DocumentBlock(
                                text=p_text,
                                page=page_idx,
                                block_type="paragraph",
                                section_title=current_section
                            ))
                        buffer_paragraph = []
                    continue

                is_q, q_num, q_text, ans_inline = detect_question_structure(l_str)
                if is_q:
                    if buffer_paragraph:
                        p_text = " ".join(buffer_paragraph).strip()
                        if p_text:
                            blocks.append(DocumentBlock(
                                text=p_text,
                                page=page_idx,
                                block_type="paragraph",
                                section_title=current_section
                            ))
                        buffer_paragraph = []
                    blocks.append(DocumentBlock(
                        text=l_str if not ans_inline else f"{q_text}",
                        page=page_idx,
                        block_type="question",
                        section_title=current_section,
                        is_qa_pair=True,
                        question_text=q_text,
                        question_number=q_num
                    ))
                    if ans_inline:
                        blocks.append(DocumentBlock(
                            text=ans_inline,
                            page=page_idx,
                            block_type="answer",
                            section_title=current_section,
                            is_qa_pair=True,
                            question_text=q_text,
                            question_number=q_num
                        ))
                elif is_likely_heading(l_str):
                    if buffer_paragraph:
                        p_text = " ".join(buffer_paragraph).strip()
                        if p_text:
                            blocks.append(DocumentBlock(
                                text=p_text,
                                page=page_idx,
                                block_type="paragraph",
                                section_title=current_section
                            ))
                        buffer_paragraph = []

                    current_section = l_str.rstrip(':')
                    blocks.append(DocumentBlock(
                        text=l_str,
                        page=page_idx,
                        block_type="heading",
                        section_title=current_section
                    ))
                elif re.match(r'^(?:[•\-\*●]|\d+[\.\)])\s+', l_str):
                    # List item
                    if buffer_paragraph:
                        p_text = " ".join(buffer_paragraph).strip()
                        if p_text:
                            blocks.append(DocumentBlock(
                                text=p_text,
                                page=page_idx,
                                block_type="paragraph",
                                section_title=current_section
                            ))
                        buffer_paragraph = []
                    blocks.append(DocumentBlock(
                        text=l_str,
                        page=page_idx,
                        block_type="list_item",
                        section_title=current_section
                    ))
                else:
                    buffer_paragraph.append(l_str)

            if buffer_paragraph:
                p_text = " ".join(buffer_paragraph).strip()
                if p_text:
                    blocks.append(DocumentBlock(
                        text=p_text,
                        page=page_idx,
                        block_type="paragraph",
                        section_title=current_section
                    ))

        return blocks, max(total_pages, 1)

    def _parse_docx(self, file_path: Path) -> (List[DocumentBlock], int):
        if docx is None:
            return self._parse_text(file_path)

        doc = docx.Document(str(file_path))
        blocks: List[DocumentBlock] = []
        current_section = "General"
        page_approx = 1
        word_counter = 0

        for element in doc.element.body:
            tag = element.tag.lower()
            if tag.endswith('p'):
                # Paragraph
                text = element.text.strip() if element.text else ""
                if not text:
                    continue
                cleaned = normalize_text(text)

                is_q, q_num, q_text, ans_inline = detect_question_structure(cleaned)
                if is_q:
                    blocks.append(DocumentBlock(
                        text=cleaned if not ans_inline else f"{q_text}",
                        page=page_approx,
                        block_type="question",
                        section_title=current_section,
                        is_qa_pair=True,
                        question_text=q_text,
                        question_number=q_num
                    ))
                    if ans_inline:
                        blocks.append(DocumentBlock(
                            text=ans_inline,
                            page=page_approx,
                            block_type="answer",
                            section_title=current_section,
                            is_qa_pair=True,
                            question_text=q_text,
                            question_number=q_num
                        ))
                elif is_likely_heading(cleaned):
                    current_section = cleaned.rstrip(':')
                    blocks.append(DocumentBlock(
                        text=cleaned,
                        page=page_approx,
                        block_type="heading",
                        section_title=current_section
                    ))
                else:
                    blocks.append(DocumentBlock(
                        text=cleaned,
                        page=page_approx,
                        block_type="paragraph",
                        section_title=current_section
                    ))

                word_counter += len(cleaned.split())
                if word_counter > 400:
                    page_approx += 1
                    word_counter = 0

            elif tag.endswith('tbl'):
                # Table element
                rows_text = []
                for row in element.findall('.//{*}tr'):
                    cell_texts = [c.text.strip() for c in row.findall('.//{*}tc') if c.text and c.text.strip()]
                    if cell_texts:
                        rows_text.append(" | ".join(cell_texts))
                if rows_text:
                    table_str = "Table:\n" + "\n".join(rows_text)
                    blocks.append(DocumentBlock(
                        text=table_str,
                        page=page_approx,
                        block_type="table",
                        section_title=current_section
                    ))

        return blocks, page_approx

    def _parse_doc(self, file_path: Path) -> (List[DocumentBlock], int):
        # Fallback stream decoding for legacy .doc
        try:
            content = file_path.read_bytes()
            # Extract printable strings of 4 or more chars
            strings = re.findall(rb'[A-Za-z0-9\s.,!?:;\'"()_\-\n]{4,}', content)
            decoded_lines = []
            for s in strings:
                try:
                    dec = s.decode('utf-8', errors='ignore').strip()
                    if len(dec) >= 4:
                        decoded_lines.append(dec)
                except Exception:
                    continue
            cleaned = normalize_text("\n".join(decoded_lines))
        except Exception:
            cleaned = ""

        blocks = []
        if cleaned:
            blocks.append(DocumentBlock(
                text=cleaned,
                page=1,
                block_type="paragraph",
                section_title="General"
            ))
        return blocks, 1

    def _parse_text(self, file_path: Path) -> (List[DocumentBlock], int):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = normalize_text(text)
        blocks = []
        for line in cleaned.splitlines():
            l_str = line.strip()
            if l_str:
                blocks.append(DocumentBlock(
                    text=l_str,
                    page=1,
                    block_type="paragraph",
                    section_title="General"
                ))
        return blocks, 1
