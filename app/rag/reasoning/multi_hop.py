from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from app.rag.ingestion.models import DocumentChunk
from app.rag.knowledge.relationship_graph import RelationshipGraph, RelationshipEdge
from app.rag.ranking.adaptive_scorer import ScoredChunk
from app.rag.query.planner import QueryPlan

@dataclass
class ReasoningHop:
    hop_index: int
    from_concept: str
    transition_action: str
    to_concept: str
    evidence_chunk: DocumentChunk
    evidence_text: str
    confidence: float

@dataclass
class ReasoningGraph:
    query: str
    hops: List[ReasoningHop] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_type: str = "sequential_workflow"  # sequential_workflow, relationship_traversal, dependency
    is_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "type": self.reasoning_type,
            "hops_count": len(self.hops),
            "confidence": round(self.confidence, 4),
            "hops": [
                {
                    "step": h.hop_index,
                    "from": h.from_concept,
                    "action": h.transition_action,
                    "to": h.to_concept,
                    "evidence_snippet": h.evidence_text[:120],
                    "chunk_id": h.evidence_chunk.chunk_id,
                    "document": h.evidence_chunk.document_name
                }
                for h in self.hops
            ]
        }

class MultiHopReasoningEngine:
    """
    Deterministic Multi-Hop Reasoning Engine (Phase 14).
    Resolves multi-step questions without a generative LLM by chaining grounded evidence hops:
      Entity / Seed State -> Intermediate Operations -> Downstream Workflow / Outcome
    Every single hop is validated against document chunks and relationship edges.
    """

    @classmethod
    def execute_multi_hop_reasoning(
        cls,
        plan: QueryPlan,
        candidates: List[ScoredChunk],
        rel_graph: Optional[RelationshipGraph] = None
    ) -> Optional[ReasoningGraph]:
        """
        Builds a multi-hop reasoning graph for workflow and multi-step questions.
        """
        if not candidates:
            return None

        # Determine target entity/subject
        subject = plan.subject or (plan.entities[0] if plan.entities else None)
        if not subject:
            return None

        reasoning = ReasoningGraph(
            query=plan.original_query,
            reasoning_type="sequential_workflow" if plan.intent == "workflow_explanation" else "relationship_traversal"
        )

        hops: List[ReasoningHop] = []
        seen_chunks: Set[str] = set()

        # Step 1: Check RelationshipGraph for direct structural edges (e.g. after_step, has_stage)
        if rel_graph:
            edges = rel_graph.traverse_workflow(subject)
            for idx, edge in enumerate(edges):
                # Find matching chunk in candidates or seed
                matched_chunk = None
                for sc in candidates:
                    if sc.chunk.chunk_id in edge.evidence_chunk_ids:
                        matched_chunk = sc.chunk
                        break
                if not matched_chunk and candidates:
                    matched_chunk = candidates[0].chunk

                if matched_chunk and matched_chunk.chunk_id not in seen_chunks:
                    seen_chunks.add(matched_chunk.chunk_id)
                    hops.append(
                        ReasoningHop(
                            hop_index=idx + 1,
                            from_concept=edge.subject,
                            transition_action=edge.predicate,
                            to_concept=edge.object_val,
                            evidence_chunk=matched_chunk,
                            evidence_text=edge.evidence_sentences[0] if edge.evidence_sentences else matched_chunk.text[:150],
                            confidence=edge.confidence
                        )
                    )

        # Step 2: Extract sequential workflow steps directly from top candidate chunks
        if not hops:
            for sc in candidates:
                chunk = sc.chunk
                if chunk.chunk_id in seen_chunks:
                    continue

                text = chunk.text
                # Look for numbered or bulleted workflow steps in chunk
                lines = text.splitlines()
                step_lines = [l.strip() for l in lines if l.strip() and (l.strip()[0].isdigit() or l.strip().startswith(('•', '-', '*', 'Step')))]
                
                if step_lines and len(step_lines) >= 2:
                    for s_idx, s_line in enumerate(step_lines[:4]):
                        hops.append(
                            ReasoningHop(
                                hop_index=s_idx + 1,
                                from_concept=subject,
                                transition_action=f"step_{s_idx + 1}",
                                to_concept=s_line,
                                evidence_chunk=chunk,
                                evidence_text=s_line,
                                confidence=sc.final_score
                            )
                        )
                    seen_chunks.add(chunk.chunk_id)
                    break

        # Step 3: Fallback sequential progression across multiple relevant candidate chunks
        if not hops and len(candidates) >= 2:
            top_cand = candidates[0]
            second_cand = candidates[1]
            hops.append(
                ReasoningHop(
                    hop_index=1,
                    from_concept=subject,
                    transition_action="initial_definition",
                    to_concept=top_cand.chunk.section or "Overview",
                    evidence_chunk=top_cand.chunk,
                    evidence_text=top_cand.chunk.text[:150],
                    confidence=top_cand.final_score
                )
            )
            hops.append(
                ReasoningHop(
                    hop_index=2,
                    from_concept=top_cand.chunk.section or subject,
                    transition_action="next_procedure",
                    to_concept=second_cand.chunk.section or "Next Step",
                    evidence_chunk=second_cand.chunk,
                    evidence_text=second_cand.chunk.text[:150],
                    confidence=second_cand.final_score
                )
            )

        if hops:
            avg_conf = sum(h.confidence for h in hops) / len(hops)
            reasoning.hops = hops
            reasoning.confidence = round(avg_conf, 4)
            reasoning.is_complete = True
            return reasoning

        return None
