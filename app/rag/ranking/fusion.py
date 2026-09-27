import math
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from app.rag.ingestion.models import DocumentChunk
from app.rag.query.planner import QueryPlan

@dataclass
class FusedCandidate:
    chunk: DocumentChunk
    normalized_score: float
    retrieval_sources: List[str]
    individual_scores: Dict[str, float]
    provenance: Dict[str, Any] = field(default_factory=dict)
    evidence_density: float = 0.0
    signal_convergence: int = 0

class CandidateFusionEngine:
    """
    Multi-Branch Candidate Score Normalization & Fusion Engine.
    Fuses retrieval candidates across:
      - Semantic Vector Search
      - Okapi BM25 Lexical Search
      - Exact Token / Code Search
      - Entity & Relationship Search
      - Structured Table Search
      - RapidFuzz Typo Search
    Normalizes divergent score distributions to [0.0, 1.0] and deduplicates while preserving provenance.
    """

    @staticmethod
    def _normalize_branch_scores(scored_list: List[Tuple[DocumentChunk, float]]) -> Dict[str, float]:
        """Normalizes raw scores from a branch to [0.0, 1.0]."""
        if not scored_list:
            return {}

        raw_scores = [s for _, s in scored_list]
        max_s = max(raw_scores)
        min_s = min(raw_scores)

        normalized: Dict[str, float] = {}
        for chunk, score in scored_list:
            cid = chunk.chunk_id
            if max_s == min_s:
                norm = 1.0 if max_s > 0 else 0.0
            else:
                # MinMax normalization with safe ceiling
                norm = (score - min_s) / (max_s - min_s) if (max_s > min_s) else score
            normalized[cid] = max(normalized.get(cid, 0.0), round(norm, 4))

        return normalized

    @classmethod
    def fuse_candidates(
        cls,
        branch_results: Dict[str, List[Tuple[DocumentChunk, float]]],
        plan: QueryPlan,
        top_k: int = 25
    ) -> List[FusedCandidate]:
        """
        Merges multi-branch candidates with plan-directed dynamic branch weights.
        """
        # Map of chunk_id -> DocumentChunk
        all_chunks: Dict[str, DocumentChunk] = {}
        # Map of chunk_id -> branch_name -> normalized_score
        candidate_branch_scores: Dict[str, Dict[str, float]] = {}

        weights = plan.retrieval_strategy.get("weights", {
            "vector": 0.45,
            "bm25": 0.25,
            "exact": 0.15,
            "entity": 0.15,
            "table": 0.15,
            "fuzzy": 0.05
        })

        for branch_name, results in branch_results.items():
            if not results:
                continue

            if branch_name == "vector":
                # Cosine similarity is already in [-1.0, 1.0], clip to [0.0, 1.0]
                norm_dict = {
                    c.chunk_id: max(0.0, min(1.0, s))
                    for c, s in results
                }
            elif branch_name in ("exact", "entity", "table", "fuzzy"):
                # Usually in [0.0, 1.0] already, clamp safely
                norm_dict = {
                    c.chunk_id: max(0.0, min(1.0, s))
                    for c, s in results
                }
            else:
                # BM25 scores can be > 10, normalize via MinMax
                norm_dict = cls._normalize_branch_scores(results)

            for chunk, raw_score in results:
                cid = chunk.chunk_id
                all_chunks[cid] = chunk
                if cid not in candidate_branch_scores:
                    candidate_branch_scores[cid] = {}
                candidate_branch_scores[cid][branch_name] = norm_dict.get(cid, 0.0)

        fused: List[FusedCandidate] = []
        for cid, chunk in all_chunks.items():
            branch_scores = candidate_branch_scores[cid]
            sources = list(branch_scores.keys())

            # Weighted fusion sum
            weighted_sum = 0.0
            total_weight = 0.0
            for b_name, s_val in branch_scores.items():
                w = weights.get(b_name, 0.10)
                weighted_sum += (s_val * w)
                total_weight += w

            normalized_composite = weighted_sum / max(0.01, total_weight)

            # Signal convergence boost: when multiple independent branches corroborate
            active_signals = len([s for s in branch_scores.values() if s >= 0.35])
            if active_signals >= 3:
                normalized_composite = min(1.0, normalized_composite * 1.08)
            elif active_signals == 2:
                normalized_composite = min(1.0, normalized_composite * 1.03)

            # Calculate evidence density (ratio of substantive words to text length)
            word_count = chunk.word_count or len(chunk.text.split())
            evidence_density = min(1.0, round(word_count / 150.0, 3))

            provenance = {
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "section": chunk.section,
                "page": chunk.page,
                "is_table_row": getattr(chunk, "is_table_row", False),
                "is_qa_pair": getattr(chunk, "is_qa_pair", False),
                "raw_branch_scores": branch_scores
            }

            fused.append(
                FusedCandidate(
                    chunk=chunk,
                    normalized_score=round(normalized_composite, 4),
                    retrieval_sources=sources,
                    individual_scores=branch_scores,
                    provenance=provenance,
                    evidence_density=evidence_density,
                    signal_convergence=active_signals
                )
            )

        fused.sort(key=lambda x: x.normalized_score, reverse=True)
        return fused[:top_k]
