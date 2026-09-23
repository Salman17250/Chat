from typing import List, Dict, Any
from app.rag.ingestion.models import ParsedDocument, DocumentChunk, DocumentBlock
from app.rag.config import MIN_CHUNK_WORDS, MAX_CHUNK_WORDS, CHUNK_OVERLAP_WORDS
from app.rag.table.table_processor import TableProcessor
from app.rag.ingestion.entity_extractor import DynamicEntityExtractor

class StructureAwareChunker:
    """
    Structure-aware chunker that respects document hierarchy.
    Groups Heading -> Section -> Paragraphs -> Lists -> Tables without blind slicing.
    Decomposes tables into parent table chunks and row-level sub-chunks with typed bindings.
    Preserves adjacent chunk linkages strictly within the same section and document.
    """

    def __init__(
        self,
        min_words: int = MIN_CHUNK_WORDS,
        max_words: int = MAX_CHUNK_WORDS,
        overlap_words: int = CHUNK_OVERLAP_WORDS
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk_document(self, parsed_doc: ParsedDocument) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        doc_id = parsed_doc.document_id
        doc_name = parsed_doc.document_name
        tenant_id = parsed_doc.tenant_id
        user_id = parsed_doc.user_id

        # Group blocks by section
        sections: Dict[str, List[DocumentBlock]] = {}
        for block in parsed_doc.blocks:
            sec = block.section_title or "General"
            if sec not in sections:
                sections[sec] = []
            sections[sec].append(block)

        chunk_counter = 0

        for sec_name, blocks in sections.items():
            current_texts: List[str] = []
            current_word_count = 0
            current_page = blocks[0].page if blocks else 1

            idx = 0
            while idx < len(blocks):
                block = blocks[idx]
                text = block.text.strip()
                if not text:
                    idx += 1
                    continue

                # 1. Check if block is a discrete Question
                if block.block_type == "question" or block.is_qa_pair:
                    # Flush any accumulated preceding prose
                    if current_texts:
                        chunk_str = "\n\n".join(current_texts).strip()
                        if chunk_str:
                            chunk_counter += 1
                            c_id = f"{doc_id}_c{chunk_counter}"
                            chunks.append(DocumentChunk(
                                document_id=doc_id,
                                document_name=doc_name,
                                page=current_page,
                                section=sec_name,
                                chunk_id=c_id,
                                text=chunk_str,
                                word_count=current_word_count,
                                tenant_id=tenant_id,
                                user_id=user_id
                            ))
                        current_texts = []
                        current_word_count = 0

                    q_text = block.question_text or text
                    q_num = block.question_number
                    q_page = block.page

                    # Collect answer blocks immediately following this question
                    ans_parts: List[str] = []
                    idx += 1
                    while idx < len(blocks):
                        next_b = blocks[idx]
                        if next_b.block_type in ("question", "heading", "table") or next_b.is_qa_pair:
                            # Reached boundary of the answer
                            break
                        if next_b.text.strip():
                            ans_parts.append(next_b.text.strip())
                        idx += 1

                    full_ans = "\n\n".join(ans_parts).strip()
                    qa_chunk_text = f"{text}\n\n{full_ans}" if full_ans else text
                    chunk_counter += 1
                    qa_cid = f"{doc_id}_c{chunk_counter}"
                    chunks.append(DocumentChunk(
                        document_id=doc_id,
                        document_name=doc_name,
                        page=q_page,
                        section=sec_name,
                        chunk_id=qa_cid,
                        text=qa_chunk_text,
                        word_count=len(qa_chunk_text.split()),
                        tenant_id=tenant_id,
                        user_id=user_id,
                        is_qa_pair=True,
                        question_text=q_text,
                        answer_text=full_ans if full_ans else None,
                        question_number=q_num
                    ))
                    continue

                # 2. Check if block is a table
                is_table = getattr(block, "block_type", "") == "table" or TableProcessor.is_table_text(text)
                if is_table:
                    # Flush any accumulated prose before table
                    if current_texts:
                        chunk_str = "\n\n".join(current_texts).strip()
                        if chunk_str:
                            chunk_counter += 1
                            c_id = f"{doc_id}_c{chunk_counter}"
                            chunks.append(DocumentChunk(
                                document_id=doc_id,
                                document_name=doc_name,
                                page=current_page,
                                section=sec_name,
                                chunk_id=c_id,
                                text=chunk_str,
                                word_count=current_word_count,
                                tenant_id=tenant_id,
                                user_id=user_id
                            ))
                        current_texts = []
                        current_word_count = 0

                    parsed_table = TableProcessor.parse_markdown_table(text)
                    if parsed_table:
                        headers, rows_str, rows_typed = parsed_table
                        chunk_counter += 1
                        parent_cid = f"{doc_id}_c{chunk_counter}"
                        parent_chunk = DocumentChunk(
                            document_id=doc_id,
                            document_name=doc_name,
                            page=block.page,
                            section=sec_name,
                            chunk_id=parent_cid,
                            text=text,
                            word_count=len(text.split()),
                            tenant_id=tenant_id,
                            user_id=user_id,
                            is_table_row=False,
                            table_headers=headers
                        )
                        chunks.append(parent_chunk)

                        # Emit row sub-chunks
                        for r_idx, (r_data, r_typed) in enumerate(zip(rows_str, rows_typed)):
                            row_txt = TableProcessor.format_row_text(sec_name, headers, r_data)
                            row_cid = f"{parent_cid}_r{r_idx + 1}"
                            chunks.append(DocumentChunk(
                                document_id=doc_id,
                                document_name=doc_name,
                                page=block.page,
                                section=sec_name,
                                chunk_id=row_cid,
                                text=row_txt,
                                word_count=len(row_txt.split()),
                                tenant_id=tenant_id,
                                user_id=user_id,
                                is_table_row=True,
                                table_headers=headers,
                                table_row_data=r_data,
                                typed_attributes=r_typed,
                                parent_chunk_id=parent_cid
                            ))
                        idx += 1
                        continue

                words = text.split()
                w_len = len(words)

                # If adding this block exceeds max_words, flush current chunk
                if current_word_count + w_len > self.max_words and current_texts:
                    chunk_str = "\n\n".join(current_texts).strip()
                    if current_word_count >= self.min_words:
                        chunk_counter += 1
                        c_id = f"{doc_id}_c{chunk_counter}"
                        chunks.append(DocumentChunk(
                            document_id=doc_id,
                            document_name=doc_name,
                            page=current_page,
                            section=sec_name,
                            chunk_id=c_id,
                            text=chunk_str,
                            word_count=current_word_count,
                            tenant_id=tenant_id,
                            user_id=user_id
                        ))

                        # Build overlap text from the end of current_texts
                        overlap_tokens = []
                        for prev_txt in reversed(current_texts):
                            prev_words = prev_txt.split()
                            overlap_tokens = prev_words + overlap_tokens
                            if len(overlap_tokens) >= self.overlap_words:
                                break
                        current_texts = [" ".join(overlap_tokens[:self.overlap_words])] if overlap_tokens else []
                        current_word_count = len(current_texts[0].split()) if current_texts else 0

                current_texts.append(text)
                current_word_count += w_len
                current_page = block.page
                idx += 1

            # Flush any remaining text in section
            if current_texts:
                chunk_str = "\n\n".join(current_texts).strip()
                if chunk_str:
                    chunk_counter += 1
                    c_id = f"{doc_id}_c{chunk_counter}"
                    chunks.append(DocumentChunk(
                        document_id=doc_id,
                        document_name=doc_name,
                        page=current_page,
                        section=sec_name,
                        chunk_id=c_id,
                        text=chunk_str,
                        word_count=current_word_count,
                        tenant_id=tenant_id,
                        user_id=user_id
                    ))

        # Establish bi-directional adjacent links ONLY within same section and non-row chunks
        section_chunks: Dict[str, List[DocumentChunk]] = {}
        for c in chunks:
            if not c.is_table_row:
                section_chunks.setdefault(c.section, []).append(c)

        for sec_list in section_chunks.values():
            for i in range(len(sec_list)):
                if i > 0:
                    sec_list[i].prev_chunk_id = sec_list[i - 1].chunk_id
                if i < len(sec_list) - 1:
                    sec_list[i].next_chunk_id = sec_list[i + 1].chunk_id

        # Dynamically extract entities, attributes, and relationships across all chunks
        for c in chunks:
            DynamicEntityExtractor.enrich_chunk(c)

        return chunks

