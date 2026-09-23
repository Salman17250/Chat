import unittest
import numpy as np
from app.rag.ingestion.models import ParsedDocument, DocumentBlock, DocumentChunk
from app.rag.chunking.structure_chunker import StructureAwareChunker
from app.rag.embeddings.fast_embedder import get_fast_embedder
from app.rag.tenant.tenant_manager import TenantManager
from app.rag.engine import RAGEngine
from app.rag.config import NO_ANSWER_FOUND_MESSAGE

class TestDynamicAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RAGEngine()
        cls.embedder = get_fast_embedder()
        cls.tenant_id = "test_dynamic_accuracy"
        tenant = cls.engine.tenant_manager.get_tenant(cls.tenant_id)
        tenant.chunks = []
        tenant.embeddings = []
        tenant.documents = {}

        # 1. Synthesize Doc A: Bio / Resume
        doc_a_text = (
            "My name is Ovesh Sheikh, and I am 25 years old. "
            "I am a software developer. He completed his BCA and works with React, Node.js, JavaScript and SQL."
        )
        chunker = StructureAwareChunker()
        parsed_doc_a = ParsedDocument(
            document_id="doc_bio_1",
            document_name="About_Me.docx",
            total_pages=1,
            tenant_id=cls.tenant_id,
            user_id="test",
            blocks=[
                DocumentBlock(text="About Me - Professional Profile", page=1, block_type="heading", section_title="About Me"),
                DocumentBlock(text=doc_a_text, page=1, block_type="paragraph", section_title="About Me")
            ]
        )
        chunks_a = chunker.chunk_document(parsed_doc_a)

        # 2. Synthesize Doc B: Policy Document
        doc_b_text = (
            "Customers may terminate their subscription within thirty days of initial purchase. "
            "A standard turnaround time of five business days applies to all processing."
        )
        parsed_doc_b = ParsedDocument(
            document_id="doc_policy_2",
            document_name="Subscription_Terms.pdf",
            total_pages=1,
            tenant_id=cls.tenant_id,
            user_id="test",
            blocks=[
                DocumentBlock(text="Cancellation and Refund Policy", page=1, block_type="heading", section_title="Cancellation Policy"),
                DocumentBlock(text=doc_b_text, page=1, block_type="paragraph", section_title="Cancellation Policy")
            ]
        )
        chunks_b = chunker.chunk_document(parsed_doc_b)

        # 3. Synthesize Doc C: Pricing Table
        doc_c_table = (
            "| Plan | Price | Users | Discount |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| Basic | 999 | 5 | 5% |\n"
            "| Pro | 1999 | 20 | 10% |\n"
            "| Enterprise | 4999 | 100 | 20% |"
        )
        parsed_doc_c = ParsedDocument(
            document_id="doc_pricing_3",
            document_name="Product_Plans.txt",
            total_pages=1,
            tenant_id=cls.tenant_id,
            user_id="test",
            blocks=[
                DocumentBlock(text="Subscription Plans and Pricing", page=1, block_type="heading", section_title="Plans"),
                DocumentBlock(text=doc_c_table, page=1, block_type="table", section_title="Plans")
            ]
        )
        chunks_c = chunker.chunk_document(parsed_doc_c)

        # 4. Synthesize Doc D: Completely Unseen Domain & Person
        doc_d_text = (
            "Dr. Sarah Jenkins is 42 years old and is a Neurosurgeon at St. Jude Hospital. "
            "Location: Chicago. Department: Neurosurgery."
        )
        parsed_doc_d = ParsedDocument(
            document_id="doc_med_4",
            document_name="Staff_Directory.pdf",
            total_pages=1,
            tenant_id=cls.tenant_id,
            user_id="test",
            blocks=[
                DocumentBlock(text="Hospital Staff Directory - Dr. Sarah Jenkins", page=1, block_type="heading", section_title="Medical Staff"),
                DocumentBlock(text=doc_d_text, page=1, block_type="paragraph", section_title="Medical Staff")
            ]
        )
        chunks_d = chunker.chunk_document(parsed_doc_d)

        all_chunks = chunks_a + chunks_b + chunks_c + chunks_d
        chunk_texts = [
            f"search_document: {c.section}: {c.text}" if c.section else f"search_document: {c.text}"
            for c in all_chunks
        ]
        embeddings = cls.embedder.embed_batch(chunk_texts)

        tenant.chunks = all_chunks
        tenant.embeddings = embeddings
        tenant.documents = {
            "doc_bio_1": {"document_id": "doc_bio_1", "filename": "About_Me.docx"},
            "doc_policy_2": {"document_id": "doc_policy_2", "filename": "Subscription_Terms.pdf"},
            "doc_pricing_3": {"document_id": "doc_pricing_3", "filename": "Product_Plans.txt"},
            "doc_med_4": {"document_id": "doc_med_4", "filename": "Staff_Directory.pdf"},
        }
        tenant.reindex()

    def test_01_semantic_paraphrase_policy(self):
        """Tests that a completely paraphrased question matches via semantic vector search."""
        q = "How long do I have to cancel?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertIn("terminate", resp.answer.lower())
        self.assertIn("thirty days", resp.answer.lower())
        self.assertIn(resp.confidence, ("HIGH", "MEDIUM"))

    def test_02_entity_attribute_age_paraphrases(self):
        """Tests multiple diverse paraphrases of age inquiry."""
        queries = [
            "How old is Ovesh?",
            "What is Ovesh's age?",
            "Tell me Ovesh's current age.",
            "How many years old is Ovesh?"
        ]
        for q in queries:
            resp = self.engine.query(q, tenant_id=self.tenant_id)
            self.assertIn("25", resp.answer, f"Failed for query: {q}")
            self.assertIn(resp.confidence, ("HIGH", "MEDIUM"), f"Low confidence for query: {q}")

    def test_03_education_extraction(self):
        """Tests education attribute extraction."""
        q = "What did Ovesh study?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertIn("bca", resp.answer.lower())

    def test_04_skills_technologies_extraction(self):
        """Tests technologies & skills extraction."""
        q = "What technologies does he work with?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        ans_lower = resp.answer.lower()
        self.assertTrue("react" in ans_lower or "node" in ans_lower or "javascript" in ans_lower)

    def test_05_structured_table_lookup(self):
        """Tests structured table row lookups with attributes."""
        q = "What is the price of the Pro plan?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertIn("1999", resp.answer)

    def test_06_deterministic_calculation(self):
        """Tests pure deterministic math evaluation."""
        q = "Calculate total for 50000 with 18% GST"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertIn("59,000", resp.answer)
        self.assertEqual(resp.confidence, "HIGH")

    def test_07_negative_unsupported_query(self):
        """Tests safe fallback when evidence is completely absent."""
        q = "What is the maternity leave allowance?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertEqual(resp.confidence, "VERY_LOW")
        self.assertEqual(resp.answer, NO_ANSWER_FOUND_MESSAGE)

    def test_08_completely_unseen_domain_and_person(self):
        """Tests that a totally novel person and domain works dynamically without code changes."""
        q_age = "How old is Sarah Jenkins?"
        resp_age = self.engine.query(q_age, tenant_id=self.tenant_id)
        self.assertIn("42", resp_age.answer)

        q_role = "What is Sarah's profession?"
        resp_role = self.engine.query(q_role, tenant_id=self.tenant_id)
        self.assertIn("neurosurgeon", resp_role.answer.lower())

    def test_09_multi_hop_multi_part_query(self):
        """Tests multi-part question decomposing and synthesizing both clauses."""
        q = "What did Ovesh study and what technologies does he work with?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        ans_lower = resp.answer.lower()
        self.assertIn("bca", ans_lower)
        self.assertTrue("react" in ans_lower or "node" in ans_lower or "sql" in ans_lower)

    def test_10_table_attribute_condition(self):
        """Tests table attribute lookups across rows."""
        q = "What is the discount on the Enterprise plan?"
        resp = self.engine.query(q, tenant_id=self.tenant_id)
        self.assertIn("20%", resp.answer)

if __name__ == "__main__":
    unittest.main()
