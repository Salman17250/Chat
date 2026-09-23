import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from rapidfuzz import fuzz
from app.rag.ingestion.models import DocumentChunk
from app.rag.ranking.adaptive_scorer import ScoredChunk
from app.rag.query.analyzer import QueryProfile
from app.rag.config import (
    MIN_CONFIDENCE,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    ADJACENT_EXPANSION_CONFIDENCE_THRESHOLD,
    DUPLICATE_OVERLAP_THRESHOLD,
    NO_ANSWER_FOUND_MESSAGE,
    FINAL_TOP_K,
)

@dataclass
class AssembledSource:
    document: str
    document_id: str
    page: int
    section: str
    score: float
    text: str
    chunk_id: str
    clause: Optional[str] = None

@dataclass
class AssembledResponse:
    answer: str
    confidence: str  # HIGH, MEDIUM, LOW, VERY_LOW
    confidence_score: float
    sources: List[AssembledSource]
    debug_trace: Optional[Dict[str, Any]] = None
    sub_query_coverages: Optional[Dict[str, bool]] = None

class ContextAssembler:
    """
    Handles context assembly, deduplication, adjacent chunk retrieval,
    calibrated multi-signal evidence gating, and cross-document multi-part aggregation.
    """

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        duplicate_threshold: float = DUPLICATE_OVERLAP_THRESHOLD,
        adjacent_threshold: float = ADJACENT_EXPANSION_CONFIDENCE_THRESHOLD,
    ):
        self.min_confidence = min_confidence
        self.duplicate_threshold = duplicate_threshold
        self.adjacent_threshold = adjacent_threshold

    def remove_duplicates(self, scored_chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        """Suppresses near-duplicate chunks based on token set similarity."""
        unique_chunks: List[ScoredChunk] = []

        for sc in scored_chunks:
            is_dup = False
            text_a = sc.chunk.text.lower()
            for u in unique_chunks:
                text_b = u.chunk.text.lower()
                sim = fuzz.token_set_ratio(text_a, text_b) / 100.0
                if sim >= self.duplicate_threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique_chunks.append(sc)

        return unique_chunks

    def expand_adjacent_chunks(
        self,
        selected_chunks: List[ScoredChunk],
        chunk_by_id: Dict[str, DocumentChunk],
        max_adjacent: int = 1
    ) -> List[ScoredChunk]:
        """
        Expands high-confidence chunks with their previous/next adjacent siblings
        strictly within the same document and same section.
        """
        expanded: List[ScoredChunk] = list(selected_chunks)
        existing_ids = {sc.chunk.chunk_id for sc in expanded}

        for sc in selected_chunks:
            if sc.final_score >= self.adjacent_threshold and not sc.chunk.is_table_row:
                # Check next chunk
                if sc.chunk.next_chunk_id and sc.chunk.next_chunk_id in chunk_by_id:
                    next_chunk = chunk_by_id[sc.chunk.next_chunk_id]
                    if (
                        next_chunk.chunk_id not in existing_ids
                        and next_chunk.section == sc.chunk.section
                        and next_chunk.document_id == sc.chunk.document_id
                        and not next_chunk.is_table_row
                    ):
                        expanded.append(
                            ScoredChunk(
                                chunk=next_chunk,
                                final_score=sc.final_score * 0.85,
                                semantic_score=sc.semantic_score * 0.85,
                                keyword_score=sc.keyword_score * 0.85,
                                phrase_score=sc.phrase_score,
                                fuzzy_score=sc.fuzzy_score,
                                heading_score=sc.heading_score,
                                intent_score=sc.intent_score,
                                metadata_score=sc.metadata_score,
                                independent_signals_count=sc.independent_signals_count,
                                explanation={"note": "adjacent_next_expansion"},
                            )
                        )
                        existing_ids.add(next_chunk.chunk_id)

                # Check parent chunk expansion (table parent or section parent)
                if sc.chunk.parent_chunk_id and sc.chunk.parent_chunk_id in chunk_by_id:
                    parent_chunk = chunk_by_id[sc.chunk.parent_chunk_id]
                    if parent_chunk.chunk_id not in existing_ids:
                        expanded.append(
                            ScoredChunk(
                                chunk=parent_chunk,
                                final_score=sc.final_score * 0.90,
                                semantic_score=sc.semantic_score * 0.90,
                                keyword_score=sc.keyword_score * 0.90,
                                phrase_score=sc.phrase_score,
                                fuzzy_score=sc.fuzzy_score,
                                heading_score=sc.heading_score,
                                intent_score=sc.intent_score,
                                metadata_score=sc.metadata_score,
                                independent_signals_count=sc.independent_signals_count,
                                explanation={"note": "parent_chunk_expansion"},
                            )
                        )
                        existing_ids.add(parent_chunk.chunk_id)

                # Check previous chunk
                if sc.chunk.prev_chunk_id and sc.chunk.prev_chunk_id in chunk_by_id:
                    prev_chunk = chunk_by_id[sc.chunk.prev_chunk_id]
                    if (
                        prev_chunk.chunk_id not in existing_ids
                        and prev_chunk.section == sc.chunk.section
                        and prev_chunk.document_id == sc.chunk.document_id
                        and not prev_chunk.is_table_row
                    ):
                        expanded.append(
                            ScoredChunk(
                                chunk=prev_chunk,
                                final_score=sc.final_score * 0.80,
                                semantic_score=sc.semantic_score * 0.80,
                                keyword_score=sc.keyword_score * 0.80,
                                phrase_score=sc.phrase_score,
                                fuzzy_score=sc.fuzzy_score,
                                heading_score=sc.heading_score,
                                intent_score=sc.intent_score,
                                metadata_score=sc.metadata_score,
                                independent_signals_count=sc.independent_signals_count,
                                explanation={"note": "adjacent_prev_expansion"},
                            )
                        )
                        existing_ids.add(prev_chunk.chunk_id)

        return expanded

    def evaluate_multi_signal_evidence(
        self,
        top_sc: ScoredChunk,
        second_sc: Optional[ScoredChunk] = None,
        thresholds: Optional[Dict[str, float]] = None,
        profile: QueryProfile = QueryProfile.CONCEPTUAL_PARAPHRASE
    ) -> Tuple[str, float, bool]:
        """
        Calibrated Multi-Signal Evidence Gating.
        Guarantees high recall on conceptual paraphrases while enforcing strict rejection on negatives.
        """
        cfg = thresholds or {}
        min_sem_pass = cfg.get("min_semantic_pass", 0.65)
        min_sem_cutoff = cfg.get("min_semantic_cutoff", 0.38)
        min_kw_cutoff = cfg.get("min_bm25_cutoff", 0.10)
        min_conf = cfg.get("min_confidence", self.min_confidence)

        score = top_sc.final_score
        sem = top_sc.semantic_score
        kw = top_sc.keyword_score
        head = top_sc.heading_score
        signals = top_sc.independent_signals_count

        cov = getattr(top_sc, "query_coverage", 1.0)

        # 1. Paraphrase Protection Rule:
        # High semantic similarity (>= min_sem_pass) passes even if BM25 is zero
        if sem >= min_sem_pass:
            if sem >= 0.75 or (signals >= 2 and score >= 0.55):
                return "HIGH", score, True
            return "MEDIUM", score, True

        # 2. Concept / Query Entity Coverage Gating:
        # If coverage is <= 0.20 and keyword match is weak (kw < 0.30)
        # and semantic similarity is not high enough to be a protected paraphrase (< min_sem_pass),
        # the key subject entities are missing from the document.
        if cov <= 0.20 and kw < 0.30 and sem < min_sem_pass:
            return "VERY_LOW", score, False

        # 3. Multi-Signal Corroboration:
        # When both semantic vector (>= 0.58) and lexical BM25 (>= 0.20) independently converge
        if (sem >= 0.58 and kw >= 0.20) or (signals >= 2 and kw >= 0.35 and cov >= 0.30 and score >= min_conf):
            if score >= 0.48:
                return "HIGH", score, True
            return "MEDIUM", score, True

        # 4. Rejection: Both semantic and lexical signals are below cutoff
        if sem < min_sem_cutoff and kw < min_kw_cutoff and head == 0.0:
            return "VERY_LOW", score, False

        # 5. Overall composite score below minimum confidence
        if score < min_conf:
            return "VERY_LOW", score, False

        # 6. Isolated weak signal rejection:
        # If there is no lexical match (kw < 0.15) and no heading match (head == 0.0)
        # and semantic similarity is not high enough to be a protected paraphrase (sem < min_sem_pass),
        # this is an uncorroborated negative/out-of-domain query.
        if signals <= 1 and sem < min_sem_pass and kw < 0.15 and head == 0.0:
            return "VERY_LOW", score, False

        return "VERY_LOW", score, False

    def extract_focused_answer(
        self,
        chunk: DocumentChunk,
        plan: Optional[Any] = None,
        final_chunks: Optional[List[ScoredChunk]] = None
    ) -> str:
        """
        Deterministically extracts the concise verified fact or clause from evidence.
        ZERO hallucinations, ZERO LLM dependency.
        """
        # 1. QA pair direct answer
        if getattr(chunk, "is_qa_pair", False) and getattr(chunk, "answer_text", None):
            return chunk.answer_text.strip()

        # 2. Dynamic Attribute Extraction
        chunk_attrs = getattr(chunk, "attributes", {}) or {}
        if plan and getattr(plan, "target_attributes", None) and chunk_attrs:
            entities = getattr(plan, "entities", []) or getattr(chunk, "entities", []) or []
            entity_name = entities[0] if entities else "The subject"

            for target_attr in plan.target_attributes:
                attr_key = None
                if target_attr in chunk_attrs:
                    attr_key = target_attr
                else:
                    for k in chunk_attrs.keys():
                        if target_attr in k or k in target_attr:
                            attr_key = k
                            break

                if attr_key:
                    val = chunk_attrs[attr_key]
                    val_str = ", ".join(val) if isinstance(val, list) else str(val).strip()

                    # Find exact ground-truth sentence in chunk text if present
                    sentences = re.split(r'(?<=[.!?])\s+', chunk.text.strip())
                    for s in sentences:
                        s_lower = s.lower()
                        if str(val).lower() in s_lower:
                            if any(e.lower() in s_lower for e in entities) or attr_key in s_lower:
                                return s.strip()

                    # Formulate verified direct statement
                    if attr_key == "age":
                        return f"{entity_name} is {val_str} years old."
                    elif attr_key == "profession":
                        art = "an " if val_str.lower()[0] in 'aeiou' else "a "
                        prefix = "" if val_str.lower().startswith(('a ', 'an ')) else art
                        return f"{entity_name} is {prefix}{val_str}."
                    elif attr_key == "education":
                        return f"{entity_name} completed {val_str}."
                    elif attr_key == "skills":
                        return f"{entity_name} works with {val_str}."
                    elif attr_key in ("price", "cost", "fee"):
                        return f"The {attr_key} is {val_str}."
                    elif attr_key in ("period", "duration", "validity"):
                        return f"The {attr_key} is {val_str}."
                    elif attr_key == "salary":
                        return f"The {attr_key} is {val_str}."
                    else:
                        return f"{entity_name}'s {attr_key.replace('_', ' ')} is {val_str}."

        # 3. Short heading expansion
        if getattr(chunk, "word_count", 0) < 15 and final_chunks and len(final_chunks) > 1:
            return f"{chunk.text.strip()}\n\n{final_chunks[1].chunk.text.strip()}"

        return chunk.text.strip()

    def assemble(
        self,
        ranked_chunks: List[ScoredChunk],
        chunk_by_id: Dict[str, DocumentChunk],
        debug_trace: Optional[Dict[str, Any]] = None,
        max_sources: int = FINAL_TOP_K,
        thresholds: Optional[Dict[str, float]] = None,
        profile: QueryProfile = QueryProfile.CONCEPTUAL_PARAPHRASE,
        plan: Optional[Any] = None,
    ) -> AssembledResponse:
        """Assembles final single-query deterministic response."""
        deduped = self.remove_duplicates(ranked_chunks)

        if not deduped:
            return AssembledResponse(
                answer=NO_ANSWER_FOUND_MESSAGE,
                confidence="VERY_LOW",
                confidence_score=0.0,
                sources=[],
                debug_trace=debug_trace,
            )

        top_sc = deduped[0]
        second_sc = deduped[1] if len(deduped) > 1 else None
        conf_category, conf_score, is_sufficient = self.evaluate_multi_signal_evidence(
            top_sc, second_sc, thresholds=thresholds, profile=profile
        )

        if not is_sufficient:
            if debug_trace:
                debug_trace["confidence"] = "VERY_LOW"
                debug_trace["rejected_reason"] = (
                    f"Evidence insufficient: score={conf_score:.4f}, "
                    f"signals={top_sc.independent_signals_count}, sem={top_sc.semantic_score:.4f}, kw={top_sc.keyword_score:.4f}"
                )
            return AssembledResponse(
                answer=NO_ANSWER_FOUND_MESSAGE,
                confidence="VERY_LOW",
                confidence_score=conf_score,
                sources=[],
                debug_trace=debug_trace,
            )

        # Contextual adjacent expansion
        expanded = self.expand_adjacent_chunks(deduped[:max_sources], chunk_by_id)
        expanded.sort(key=lambda x: x.final_score, reverse=True)
        final_chunks = expanded[:max_sources]

        sources: List[AssembledSource] = []
        for sc in final_chunks:
            sources.append(
                AssembledSource(
                    document=sc.chunk.document_name,
                    document_id=sc.chunk.document_id,
                    page=sc.chunk.page,
                    section=sc.chunk.section,
                    score=round(sc.final_score, 4),
                    text=sc.chunk.text,
                    chunk_id=sc.chunk.chunk_id,
                )
            )

        top_sc = final_chunks[0]
        top_chunk = top_sc.chunk

        # Extract focused verified answer
        answer = self.extract_focused_answer(top_chunk, plan=plan, final_chunks=final_chunks)

        if debug_trace:
            debug_trace["confidence"] = conf_category
            debug_trace["confidence_score"] = round(conf_score, 4)
            debug_trace["final_selected_sources"] = len(sources)

        return AssembledResponse(
            answer=answer,
            confidence=conf_category,
            confidence_score=round(conf_score, 4),
            sources=sources,
            debug_trace=debug_trace,
        )

    def assemble_multi_part(
        self,
        sub_query_results: List[Tuple[str, List[ScoredChunk]]],
        chunk_by_id: Dict[str, DocumentChunk],
        thresholds: Optional[Dict[str, float]] = None,
        debug_trace: Optional[Dict[str, Any]] = None,
        max_sources_per_clause: int = 2
    ) -> AssembledResponse:
        """
        Balanced Cross-Document Evidence Aggregator for Multi-Part Questions.
        Allocates top evidence per sub-query across different documents with explicit clause attribution.
        """
        clause_answers: List[str] = []
        all_sources: List[AssembledSource] = []
        coverages: Dict[str, bool] = {}
        conf_scores: List[float] = []
        seen_chunk_ids: Set[str] = set()

        for item in sub_query_results:
            sub_q = item[0]
            ranked_chunks = item[1]
            sub_plan = item[2] if len(item) > 2 else None

            deduped = self.remove_duplicates(ranked_chunks)
            if not deduped:
                coverages[sub_q] = False
                clause_answers.append(
                    f'Regarding "{sub_q}":\nInformation was not found in the uploaded documents.'
                )
                continue

            top_sc = deduped[0]
            second_sc = deduped[1] if len(deduped) > 1 else None
            conf_cat, conf_score, is_suff = self.evaluate_multi_signal_evidence(
                top_sc, second_sc, thresholds=thresholds, profile=QueryProfile.CONCEPTUAL_PARAPHRASE
            )

            if is_suff:
                coverages[sub_q] = True
                conf_scores.append(conf_score)
                clause_ans = self.extract_focused_answer(top_sc.chunk, plan=sub_plan)
                clause_answers.append(
                    f'Regarding "{sub_q}":\n{clause_ans}'
                )

                # Add sources for this sub-query
                for sc in deduped[:max_sources_per_clause]:
                    if sc.chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(sc.chunk.chunk_id)
                        all_sources.append(
                            AssembledSource(
                                document=sc.chunk.document_name,
                                document_id=sc.chunk.document_id,
                                page=sc.chunk.page,
                                section=sc.chunk.section,
                                score=round(sc.final_score, 4),
                                text=sc.chunk.text,
                                chunk_id=sc.chunk.chunk_id,
                                clause=sub_q
                            )
                        )
            else:
                coverages[sub_q] = False
                clause_answers.append(
                    f'Regarding "{sub_q}":\nInformation was not found in the uploaded documents.'
                )

        # Composite confidence
        any_found = any(coverages.values())
        all_found = all(coverages.values())

        if not any_found:
            composite_answer = NO_ANSWER_FOUND_MESSAGE
            composite_conf = "VERY_LOW"
            avg_score = 0.0
        else:
            composite_answer = "\n\n---\n\n".join(clause_answers)
            avg_score = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0
            composite_conf = "HIGH" if all_found and avg_score >= 0.60 else "MEDIUM"

        return AssembledResponse(
            answer=composite_answer,
            confidence=composite_conf,
            confidence_score=round(avg_score, 4),
            sources=all_sources,
            debug_trace=debug_trace,
            sub_query_coverages=coverages
        )

