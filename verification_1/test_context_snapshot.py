"""
Unit tests for verification_1 RAG Evidence & Context Snapshotting.
Tests do not require external APIs, models, ChromaDB, or internet access.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from verification_1.context_snapshot import (
    create_snapshot,
    extract_context_chunks,
    load_snapshot,
    save_snapshot,
    validate_snapshot,
)


def make_dummy_chunk(
    idx: int,
    text: str = "Sample chunk text for psychology concept.",
    section: str = "1.1 What Is Psychology?",
    section_path: str = "intro/what_is_psychology",
    chapter: str = "Chapter 1 Introduction to Psychology",
    page_start: int = 20,
    page_end: int = 22,
    pages: Optional[List[int]] = None,
    rrf_score: float = 0.033333,
    source: str = "both",
    is_supplementary: Any = False,
) -> Dict[str, Any]:
    """Helper to create a realistic dummy chunk with all required fields."""
    if pages is None:
        pages = list(range(page_start, page_end + 1))
    return {
        "chunk_id": f"dummy_chunk_{idx}",
        "text": text,
        "section": section,
        "section_path": section_path,
        "chapter": chapter,
        "page_start": page_start,
        "page_end": page_end,
        "pages": pages,
        "rrf_score": rrf_score,
        "source": source,
        "is_supplementary": is_supplementary,
    }


class TestContextSnapshot(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            make_dummy_chunk(
                idx=1,
                text="Classical conditioning involves associating an involuntary response and a stimulus.",
                section="6.2 Classical Conditioning",
                section_path="learning/classical_conditioning",
                chapter="Chapter 6 Learning",
                page_start=196,
                page_end=198,
                pages=[196, 197, 198],
                rrf_score=0.032258,
                source="both",
            ),
            make_dummy_chunk(
                idx=2,
                text="Operant conditioning focuses on using either reinforcement or punishment.",
                section="6.3 Operant Conditioning",
                section_path="learning/operant_conditioning",
                chapter="Chapter 6 Learning",
                page_start=199,
                page_end=201,
                pages=[199, 200, 201],
                rrf_score=0.031250,
                source="vector",
            ),
            make_dummy_chunk(
                idx=3,
                text="Observational learning occurs by watching others.",
                section="6.4 Observational Learning",
                section_path="learning/observational_learning",
                chapter="Chapter 6 Learning",
                page_start=202,
                page_end=203,
                pages=[202, 203],
                rrf_score=0.030303,
                source="bm25",
            ),
        ]
        self.query_id = "Q_TEST_001"
        self.question = "What is the difference between classical and operant conditioning?"
        self.answer = "Classical conditioning associates involuntary stimuli, while operant conditioning uses reinforcement."

    def test_1_snapshot_creation_works(self):
        """TEST 1: Verify snapshot creation produces a structured dictionary."""
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=self.sample_chunks,
        )
        self.assertIsInstance(snapshot, dict)
        self.assertEqual(snapshot["query_id"], self.query_id)
        self.assertTrue(validate_snapshot(snapshot))

    def test_2_snapshot_contains_question_and_answer(self):
        """TEST 2: Verify question and answer strings are accurately retained."""
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=self.sample_chunks,
        )
        self.assertEqual(snapshot["question"], self.question)
        self.assertEqual(snapshot["answer"], self.answer)

    def test_3_all_retrieved_chunk_metadata_is_preserved(self):
        """TEST 3: Verify all 11 fields of retrieved chunks are preserved untouched."""
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=self.sample_chunks,
        )
        self.assertEqual(len(snapshot["retrieved_chunks"]), 3)
        for original, preserved in zip(self.sample_chunks, snapshot["retrieved_chunks"]):
            self.assertEqual(original, preserved)

    def test_4_retrieved_chunks_and_context_chunks_are_distinguishable(self):
        """TEST 4: Verify retrieved_chunks and context_chunks can differ when limit is hit."""
        # Create chunks with large text
        large_chunk_1 = make_dummy_chunk(1, text="A" * 150)
        large_chunk_2 = make_dummy_chunk(2, text="B" * 150)
        chunks = [large_chunk_1, large_chunk_2]

        # Use a tight limit that only fits the first chunk
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=chunks,
            max_context_chars=250,
        )
        self.assertEqual(len(snapshot["retrieved_chunks"]), 2)
        self.assertEqual(len(snapshot["context_chunks"]), 1)
        self.assertEqual(snapshot["metadata"]["retrieved_chunk_count"], 2)
        self.assertEqual(snapshot["metadata"]["context_chunk_count"], 1)

    def test_5_json_serialization_works(self):
        """TEST 5: Verify snapshot can be serialized and saved as JSON to disk."""
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=self.sample_chunks,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, output_dir=tmpdir)
            self.assertTrue(path.exists())
            self.assertTrue(path.name.endswith(".json"))

    def test_6_json_loading_works(self):
        """TEST 6: Verify saved JSON can be reloaded with complete fidelity."""
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=self.sample_chunks,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_snapshot(snapshot, output_dir=tmpdir)
            loaded = load_snapshot(path)
            self.assertEqual(snapshot, loaded)

    def test_7_validation_catches_missing_required_fields(self):
        """TEST 7: Verify validator catches missing keys or malformed structures."""
        bad_snapshot = {
            "query_id": "BAD_Q",
            "question": "Q?",
            # missing "answer"
            "retrieved_chunks": [],
            "context_chunks": [],
            "final_context": "",
            "metadata": {
                "max_context_chars": 20000,
                "retrieved_chunk_count": 0,
                "context_chunk_count": 0,
            },
        }
        with self.assertRaises(ValueError):
            validate_snapshot(bad_snapshot)

    def test_8_all_chunks_fit_under_max_context_chars(self):
        """TEST 8: Verify all chunks are included when total length is well within limit."""
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=self.sample_chunks,
            max_context_chars=20000,
        )
        self.assertEqual(len(snapshot["context_chunks"]), len(self.sample_chunks))
        self.assertEqual(snapshot["metadata"]["context_chunk_count"], 3)
        self.assertEqual(snapshot["metadata"]["retrieved_chunk_count"], 3)

    def test_9_later_chunks_excluded_when_context_limit_reached(self):
        """TEST 9: Verify strict cutoff behavior when character limit is reached."""
        chunks = [
            make_dummy_chunk(1, text="Chunk one text."),
            make_dummy_chunk(2, text="Chunk two text."),
            make_dummy_chunk(3, text="Chunk three text."),
        ]
        # Calculate length of first chunk formatted
        _, final_one = extract_context_chunks(chunks[:1], max_context_chars=20000)
        
        # Set limit just enough for chunk 1 but not chunk 2
        tight_limit = len(final_one) + 10
        ctx_chunks, final_ctx = extract_context_chunks(chunks, max_context_chars=tight_limit)

        self.assertEqual(len(ctx_chunks), 1)
        self.assertEqual(ctx_chunks[0]["chunk_id"], "dummy_chunk_1")
        self.assertNotIn("dummy_chunk_2", final_ctx)
        self.assertNotIn("dummy_chunk_3", final_ctx)

    def test_10_final_context_corresponds_to_context_chunks(self):
        """TEST 10: Verify final_context text corresponds exactly to context_chunks."""
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=self.sample_chunks,
        )
        final_context = snapshot["final_context"]
        for i, chunk in enumerate(snapshot["context_chunks"]):
            expected_header = f"[Source {i+1}] Section: {chunk['section']}"
            self.assertIn(expected_header, final_context)
            self.assertIn(chunk["text"], final_context)

    def test_11_page_metadata_preserved_exactly(self):
        """TEST 11: Verify page_start, page_end, and pages list match exactly."""
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=self.sample_chunks,
        )
        c1 = snapshot["retrieved_chunks"][0]
        self.assertEqual(c1["page_start"], 196)
        self.assertEqual(c1["page_end"], 198)
        self.assertEqual(c1["pages"], [196, 197, 198])

    def test_12_chunk_ids_preserved_exactly(self):
        """TEST 12: Verify unique chunk_ids are preserved in order."""
        snapshot = create_snapshot(
            query_id=self.query_id,
            question=self.question,
            answer=self.answer,
            retrieved_chunks=self.sample_chunks,
        )
        ids = [c["chunk_id"] for c in snapshot["retrieved_chunks"]]
        self.assertEqual(ids, ["dummy_chunk_1", "dummy_chunk_2", "dummy_chunk_3"])


if __name__ == "__main__":
    unittest.main()
