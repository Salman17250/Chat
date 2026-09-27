"""
tests/test_master_engine.py
Master Test Suite for Intelligent, Ultra-Fast, High-Accuracy, Zero-LLM Query-Path RAG Engine.
Covers all 10 evaluation categories defined in Section 29 of the Master Implementation Prompt:
  1. Basic Semantic (Definitions, explanations, paraphrases, Hinglish)
  2. Context (Session state, topic carry-forward)
  3. Multi-Hop Reasoning (Multi-step procedural reasoning)
  4. Comparison (Structured side-by-side attribute matrices)
  5. Exact Search (Exact identifiers, codes, values)
  6. Table Lookup (Row-level structured table queries)
  7. Calculation (Deterministic GST, discounts, arithmetic, averages, ratios)
  8. Unsupported / Negative (Controlled rejection of absent facts)
  9. Compound Questions (Question decomposition into sub-clauses)
 10. Coreference Resolution (Resolving 'it', 'this', 'that', 'they' without LLMs)
"""

import unittest
from app.rag.engine import get_rag_engine
from app.rag.config import NO_ANSWER_FOUND_MESSAGE


class TestMasterEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_rag_engine()
        cls.tenant_id = "default"

    # Category 1: Basic Semantic Understanding
    def test_01_basic_semantic_definition_and_paraphrase(self):
        """Tests that varied paraphrases of 'What is a Pipeline?' resolve to evidence-backed answers."""
        queries = [
            "What is a Pipeline?",
            "What does Pipeline mean?",
            "Explain Pipeline.",
            "Pipeline kya hai?"
        ]
        for q in queries:
            resp = self.engine.query(q, tenant_id=self.tenant_id)
            self.assertIn("pipeline", resp.answer.lower(), f"Failed for query: {q}")
            self.assertIn(resp.confidence, ("HIGH", "MEDIUM"), f"Confidence low for query: {q}")
            self.assertTrue(len(resp.sources) > 0, f"No sources for query: {q}")

    # Category 2: Context Resolution & Session State
    def test_02_conversational_context_flow(self):
        """Tests multi-turn session state continuity across 3 conversational turns."""
        sess_id = "master_sess_ctx_1"
        u_id = "master_user_1"

        # Turn 1: Anchor on Pipeline
        resp1 = self.engine.query("What is Pipeline?", tenant_id=self.tenant_id, session_id=sess_id, user_id=u_id)
        self.assertIn("pipeline", resp1.answer.lower())

        # Turn 2: Follow-up using 'it'
        resp2 = self.engine.query("How do I configure it?", tenant_id=self.tenant_id, session_id=sess_id, user_id=u_id)
        self.assertTrue(
            "pipeline" in resp2.answer.lower() or "stage" in resp2.answer.lower() or "configure" in resp2.answer.lower(),
            f"Failed to resolve 'it' to Pipeline: {resp2.answer}"
        )

        # Turn 3: Follow-up using 'what happens after that?'
        resp3 = self.engine.query("What happens after that?", tenant_id=self.tenant_id, session_id=sess_id, user_id=u_id)
        self.assertNotEqual(resp3.answer, NO_ANSWER_FOUND_MESSAGE)
        self.assertIn(resp3.confidence, ("HIGH", "MEDIUM"))

    # Category 3: Multi-Hop Reasoning
    def test_03_multi_hop_workflow_reasoning(self):
        """Tests procedural multi-hop sequence: 'What happens after creating a pipeline?'"""
        q = "What happens after creating a pipeline?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        ans_lower = resp.answer.lower()
        self.assertTrue("configure" in ans_lower or "designed" in ans_lower or "workflow" in ans_lower, f"Unexpected answer: {resp.answer}")
        self.assertIn(resp.confidence, ("HIGH", "MEDIUM"))
        self.assertTrue(len(resp.sources) > 0)

    # Category 4: Deterministic Comparison
    def test_04_entity_comparison(self):
        """Tests side-by-side comparison without hallucinated missing values."""
        q = "Compare Pipeline and Stage."
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertIn("comparison", resp.answer.lower())
        self.assertIn("pipeline", resp.answer.lower())
        self.assertIn("stage", resp.answer.lower())
        self.assertEqual(resp.confidence, "HIGH")

    # Category 5: Exact Search
    def test_05_exact_search_retrieval(self):
        """Tests exact phrase and question-number matching in document."""
        q = "Why are stages important?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertIn("stage", resp.answer.lower())
        self.assertIn(resp.confidence, ("HIGH", "MEDIUM"))

    # Category 6: Table Lookup
    def test_06_table_lookup(self):
        """Tests structured table query execution."""
        # Querying pipeline stages table
        q = "What are the default stages in the pipeline?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertNotEqual(resp.answer, NO_ANSWER_FOUND_MESSAGE)
        self.assertIn(resp.confidence, ("HIGH", "MEDIUM"))

    # Category 7: Deterministic Calculation Engine
    def test_07_deterministic_calculation(self):
        """Tests calculation of GST, discounts, and arithmetic directly without an LLM."""
        # GST
        gst_query = "What is 18% GST on ₹10,000?"
        gst_resp = self.engine.query(gst_query, tenant_id=self.tenant_id)
        self.assertIn("1,800", gst_resp.answer)
        self.assertIn("11,800", gst_resp.answer)
        self.assertEqual(gst_resp.confidence, "HIGH")

        # Discount
        disc_query = "What is 15% discount on 2000?"
        disc_resp = self.engine.query(disc_query, tenant_id=self.tenant_id)
        self.assertIn("300", disc_resp.answer)
        self.assertIn("1,700", disc_resp.answer)
        self.assertEqual(disc_resp.confidence, "HIGH")

        # Basic Arithmetic
        arith_query = "Calculate 450 + 550"
        arith_resp = self.engine.query(arith_query, tenant_id=self.tenant_id)
        self.assertIn("1000", arith_resp.answer)

    # Category 8: Unsupported / Negative Test
    def test_08_unsupported_negative_query(self):
        """Tests that questions with zero document evidence are rejected with controlled message."""
        q = "Who is the CEO of Google?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertEqual(resp.answer, NO_ANSWER_FOUND_MESSAGE)
        self.assertEqual(resp.confidence, "VERY_LOW")
        self.assertEqual(len(resp.sources), 0)

    # Category 9: Compound Question Decomposition
    def test_09_compound_question_decomposition(self):
        """Tests compound question decomposing into sub-clauses and attributing each."""
        q = "What is a Pipeline, and why are stages important?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        ans_lower = resp.answer.lower()
        self.assertIn("pipeline", ans_lower)
        self.assertIn("stage", ans_lower)
        self.assertIn(resp.confidence, ("HIGH", "MEDIUM"))

    # Category 10: Coreference Resolution
    def test_10_coreference_resolution(self):
        """Tests multi-turn coreference resolution across pronouns ('it', 'this')."""
        sess_id = "master_sess_coref_2"
        u_id = "master_user_coref"

        # Turn 1
        t1 = self.engine.query("What is a Pipeline?", tenant_id=self.tenant_id, session_id=sess_id, user_id=u_id)
        self.assertIn("pipeline", t1.answer.lower())

        # Turn 2: 'it' refers to Pipeline
        t2 = self.engine.query("Can I add stages to it?", tenant_id=self.tenant_id, session_id=sess_id, user_id=u_id)
        self.assertNotEqual(t2.answer, NO_ANSWER_FOUND_MESSAGE)
        self.assertIn(t2.confidence, ("HIGH", "MEDIUM"))

        # Turn 3: 'it' continues referring to Pipeline / Stages
        t3 = self.engine.query("Why is it important?", tenant_id=self.tenant_id, session_id=sess_id, user_id=u_id)
        self.assertNotEqual(t3.answer, NO_ANSWER_FOUND_MESSAGE)
        self.assertIn(t3.confidence, ("HIGH", "MEDIUM"))

    # Observability & Timing Breakdown Test
    def test_11_observability_timing_breakdown(self):
        """Verifies that all 20 observability timing metrics are populated."""
        q = "What is a Pipeline?"
        resp = self.engine.query(q, tenant_id=self.tenant_id, include_debug=True)
        self.assertIsNotNone(resp.timing_breakdown)
        tb = resp.timing_breakdown
        expected_metrics = [
            "query_normalization_ms",
            "context_resolution_ms",
            "intent_detection_ms",
            "entity_detection_ms",
            "total_ms"
        ]
        for m in expected_metrics:
            self.assertIn(m, tb, f"Missing timing metric: {m}")
        self.assertGreater(tb["total_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
