"""
Unit tests for verification_1 Claim-to-Evidence Semantic Similarity Matcher.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from verification_1.similarity_matcher import (
    compute_cosine_similarity,
    compute_similarity_for_snapshot,
    get_chunk_vector,
    process_snapshot_file,
)


class MockEmbeddingModel:
    """Mock embedding model producing deterministic vectors for fast isolated unit tests."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for t in texts:
            # Deterministic pseudo-vector based on string hash/length
            val = float(len(t) % 10 + 1)
            vec = np.zeros(384, dtype=np.float32)
            vec[0] = val
            vec[1] = 1.0
            norm = np.linalg.norm(vec)
            if norm > 0 and normalize_embeddings:
                vec = vec / norm
            vectors.append(vec)
        if len(texts) == 1 and not isinstance(texts, list):
            return vectors[0]
        return np.vstack(vectors) if len(vectors) > 1 else np.array(vectors)


class TestSimilarityMatcher(unittest.TestCase):

    def setUp(self):
        self.mock_model = MockEmbeddingModel()
        self.dummy_chunk_1 = {
            "chunk_id": "chunk_001",
            "text": "Classical conditioning is an associative learning process.",
            "section": "6.2 Classical Conditioning",
            "section_path": "learning/classical_conditioning",
            "chapter": "Chapter 6 Learning",
            "page_start": 195,
            "page_end": 196,
            "pages": [195, 196],
            "rrf_score": 0.033333,
            "source": "both",
            "is_supplementary": False,
        }
        self.dummy_chunk_2 = {
            "chunk_id": "chunk_002",
            "text": "Operant conditioning involves rewards and punishments.",
            "section": "6.3 Operant Conditioning",
            "section_path": "learning/operant_conditioning",
            "chapter": "Chapter 6 Learning",
            "page_start": 199,
            "page_end": 201,
            "pages": [199, 200, 201],
            "rrf_score": 0.031250,
            "source": "vector",
            "is_supplementary": False,
        }
        self.sample_snapshot = {
            "query_id": "TEST_SIM_001",
            "question": "What is classical conditioning?",
            "answer": "Classical conditioning is a learning process. It associates stimuli.",
            "retrieved_chunks": [self.dummy_chunk_1, self.dummy_chunk_2],
            "context_chunks": [self.dummy_chunk_1],  # chunk_2 omitted from context
            "final_context": "[Source 1] ...",
            "metadata": {
                "max_context_chars": 20000,
                "retrieved_chunk_count": 2,
                "context_chunk_count": 1,
            },
        }

    def test_1_cosine_similarity_calculation(self):
        """TEST 1: Verify cosine similarity basic calculation."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        sim = compute_cosine_similarity(v1, v2)
        expected = 0.5 / np.sqrt(0.5)
        self.assertAlmostEqual(sim, expected, places=4)

    def test_2_identical_normalized_vectors_produce_one(self):
        """TEST 2: Verify identical normalized vectors yield similarity ≈ 1.0."""
        vec = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        sim = compute_cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_3_orthogonal_vectors_produce_zero(self):
        """TEST 3: Verify orthogonal vectors yield similarity ≈ 0.0."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_4_opposite_normalized_vectors_produce_minus_one(self):
        """TEST 4: Verify opposite normalized vectors yield similarity ≈ -1.0."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        sim = compute_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, -1.0, places=5)

    def test_5_claim_and_chunk_metadata_preserved(self):
        """TEST 5: Verify all chunk metadata and claim fields are preserved."""
        result = compute_similarity_for_snapshot(
            self.sample_snapshot, model=self.mock_model, cached_map={}
        )
        self.assertEqual(result["query_id"], "TEST_SIM_001")
        self.assertEqual(len(result["claims"]), 2)

        claim1 = result["claims"][0]
        self.assertEqual(claim1["claim_id"], "TEST_SIM_001_C1")
        self.assertEqual(claim1["sentence_index"], 1)
        self.assertEqual(claim1["text"], "Classical conditioning is a learning process.")

        match1 = claim1["evidence_matches"][0]
        self.assertEqual(match1["chunk_id"], "chunk_001")
        self.assertEqual(match1["section"], "6.2 Classical Conditioning")
        self.assertEqual(match1["section_path"], "learning/classical_conditioning")
        self.assertEqual(match1["chapter"], "Chapter 6 Learning")
        self.assertEqual(match1["page_start"], 195)
        self.assertEqual(match1["page_end"], 196)
        self.assertEqual(match1["pages"], [195, 196])
        self.assertEqual(match1["rrf_score"], 0.033333)

    def test_6_context_chunks_correctly_get_in_llm_context_true(self):
        """TEST 6: Verify chunk in context_chunks gets in_llm_context = True."""
        result = compute_similarity_for_snapshot(
            self.sample_snapshot, model=self.mock_model, cached_map={}
        )
        match1 = result["claims"][0]["evidence_matches"][0]
        self.assertEqual(match1["chunk_id"], "chunk_001")
        self.assertTrue(match1["in_llm_context"])

    def test_7_omitted_retrieved_chunks_get_in_llm_context_false(self):
        """TEST 7: Verify retrieved chunk NOT in context_chunks gets in_llm_context = False."""
        result = compute_similarity_for_snapshot(
            self.sample_snapshot, model=self.mock_model, cached_map={}
        )
        match2 = result["claims"][0]["evidence_matches"][1]
        self.assertEqual(match2["chunk_id"], "chunk_002")
        self.assertFalse(match2["in_llm_context"])

    def test_8_missing_cached_embedding_falls_back_to_encode(self):
        """TEST 8: Verify missing cached vector seamlessly falls back to model encoding."""
        cached_map = {
            "chunk_001": np.array([1.0] + [0.0] * 383, dtype=np.float32)
        }
        # chunk_002 is not in cached_map -> must fallback to encode
        vec = get_chunk_vector(self.dummy_chunk_2, cached_map=cached_map, model=self.mock_model)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_9_output_schema_is_valid(self):
        """TEST 9: Verify complete output dictionary conforms to required schema."""
        result = compute_similarity_for_snapshot(
            self.sample_snapshot, model=self.mock_model, cached_map={}
        )
        self.assertIn("query_id", result)
        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("claims", result)
        self.assertIn("metadata", result)

        meta = result["metadata"]
        self.assertEqual(meta["embedding_model"], "all-MiniLM-L6-v2")
        self.assertEqual(meta["similarity_metric"], "cosine_similarity")
        self.assertEqual(meta["claim_count"], 2)
        self.assertEqual(meta["evaluated_chunks_count"], 2)

        for c in result["claims"]:
            self.assertIn("claim_id", c)
            self.assertIn("sentence_index", c)
            self.assertIn("text", c)
            self.assertIn("evidence_matches", c)
            for m in c["evidence_matches"]:
                self.assertIn("chunk_id", m)
                self.assertIn("similarity_score", m)
                self.assertIsInstance(m["similarity_score"], float)
                self.assertIn("in_llm_context", m)
                self.assertIsInstance(m["in_llm_context"], bool)

    def test_10_no_threshold_or_verdict_field_produced(self):
        """TEST 10: Verify no threshold, verdict, support, or hallucination fields exist."""
        result = compute_similarity_for_snapshot(
            self.sample_snapshot, model=self.mock_model, cached_map={}
        )
        prohibited_terms = {
            "verdict",
            "supported",
            "unsupported",
            "hallucinated",
            "hallucination",
            "contradiction",
            "threshold",
            "confidence",
            "faithfulness",
        }

        def check_no_prohibited(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    self.assertNotIn(
                        k.lower(),
                        prohibited_terms,
                        f"Prohibited field '{k}' found at {path}",
                    )
                    if isinstance(v, str):
                        self.assertNotIn(
                            v.lower(),
                            prohibited_terms,
                            f"Prohibited value '{v}' found at {path}.{k}",
                        )
                    check_no_prohibited(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    check_no_prohibited(item, f"{path}[{idx}]")

        check_no_prohibited(result)

    def test_11_process_snapshot_file_end_to_end(self):
        """TEST 11: Verify end-to-end file loading, computing, and saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "test_snap.json"
            out_dir = Path(tmpdir) / "similarity_scores"

            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(self.sample_snapshot, f)

            out_file = process_snapshot_file(
                snapshot_path=snap_path,
                output_dir=out_dir,
                model=self.mock_model,
                cached_map={},
            )
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["query_id"], "TEST_SIM_001")
            self.assertEqual(len(loaded["claims"]), 2)

    def test_12_empty_answer_handled_gracefully(self):
        """TEST 12: Verify empty answer produces 0 claims without errors."""
        empty_snap = {
            "query_id": "TEST_EMPTY",
            "question": "What is nothing?",
            "answer": "",
            "retrieved_chunks": [self.dummy_chunk_1],
            "context_chunks": [self.dummy_chunk_1],
            "final_context": "",
            "metadata": {
                "max_context_chars": 20000,
                "retrieved_chunk_count": 1,
                "context_chunk_count": 1,
            },
        }
        result = compute_similarity_for_snapshot(
            empty_snap, model=self.mock_model, cached_map={}
        )
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["metadata"]["claim_count"], 0)


if __name__ == "__main__":
    unittest.main()
