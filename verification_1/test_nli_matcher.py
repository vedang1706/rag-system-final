"""
Unit tests for verification_1 Claim-to-Evidence Natural Language Inference (NLI) Matcher.
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from verification_1.nli_matcher import (
    compute_nli_for_snapshot,
    compute_nli_scores_batch,
    get_nli_model,
    process_snapshot_file,
)


class MockTokenizer:
    """Mock tokenizer returning dummy dict structure for fast unit tests."""

    def __call__(self, text_pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        class DummyTensor:
            def __init__(self, count):
                self.count = count

        return {"input_ids": DummyTensor(len(text_pairs))}


class MockNLIModel:
    """Mock NLI model producing deterministic probabilities based on input keywords."""

    def __init__(self):
        class DummyConfig:
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
            label2id = {"contradiction": 0, "entailment": 1, "neutral": 2}

        self.config = DummyConfig()

    def eval(self):
        pass

    def __call__(self, **kwargs):
        class DummyOutput:
            def __init__(self, count):
                import torch
                # Return uniform or deterministic logits: [contradiction=-1.0, entailment=3.0, neutral=0.0]
                logits = torch.tensor([[-1.0, 3.0, 0.0]] * count, dtype=torch.float32)
                self.logits = logits

        count = kwargs["input_ids"].count
        return DummyOutput(count)


class TestNLIMatcher(unittest.TestCase):

    def setUp(self):
        self.mock_tokenizer = MockTokenizer()
        self.mock_model = MockNLIModel()
        self.mock_label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.mock_bundle = (self.mock_tokenizer, self.mock_model, self.mock_label_map)

        self.dummy_chunk_1 = {
            "chunk_id": "chunk_001",
            "text": "Classical conditioning is an associative learning process where stimuli are paired.",
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
            "text": "Operant conditioning involves rewards and punishments after a behavior.",
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
            "query_id": "TEST_NLI_001",
            "question": "What is classical conditioning?",
            "answer": "Classical conditioning associates stimuli. Pavlov demonstrated this with dogs.",
            "retrieved_chunks": [self.dummy_chunk_1, self.dummy_chunk_2],
            "context_chunks": [self.dummy_chunk_1, self.dummy_chunk_2],
            "final_context": "[Source 1] ...\n\n---\n\n[Source 2] ...",
            "metadata": {
                "max_context_chars": 20000,
                "retrieved_chunk_count": 2,
                "context_chunk_count": 2,
            },
        }

    def test_1_probabilities_sum_to_approximately_one(self):
        """TEST 1: Verify NLI probabilities approximately sum to 1.0."""
        pairs = [
            ("Premise text about psychology.", "Hypothesis text claiming a fact."),
            ("Another premise passage.", "Another claim statement."),
        ]
        scores = compute_nli_scores_batch(pairs, model_bundle=self.mock_bundle)
        self.assertEqual(len(scores), 2)
        for s in scores:
            prob_sum = s["entailment_score"] + s["neutral_score"] + s["contradiction_score"]
            self.assertAlmostEqual(prob_sum, 1.0, places=2)
            self.assertGreaterEqual(s["entailment_score"], 0.0)
            self.assertLessEqual(s["entailment_score"], 1.0)
            self.assertGreaterEqual(s["neutral_score"], 0.0)
            self.assertLessEqual(s["neutral_score"], 1.0)
            self.assertGreaterEqual(s["contradiction_score"], 0.0)
            self.assertLessEqual(s["contradiction_score"], 1.0)

    def test_2_all_expected_claim_chunk_pairs_produced(self):
        """TEST 2: Verify total comparisons equal N_claims * M_retrieved_chunks."""
        result = compute_nli_for_snapshot(
            self.sample_snapshot, model_bundle=self.mock_bundle, similarity_dir=None
        )
        self.assertEqual(len(result["claims"]), 2)
        for claim in result["claims"]:
            self.assertEqual(len(claim["evidence_matches"]), 2)

    def test_3_metadata_and_text_fields_preserved(self):
        """TEST 3: Verify chunk metadata, claim text, and chunk text are preserved."""
        result = compute_nli_for_snapshot(
            self.sample_snapshot, model_bundle=self.mock_bundle, similarity_dir=None
        )
        claim1 = result["claims"][0]
        self.assertEqual(claim1["claim_id"], "TEST_NLI_001_C1")
        self.assertEqual(claim1["text"], "Classical conditioning associates stimuli.")

        match1 = claim1["evidence_matches"][0]
        self.assertEqual(match1["claim_id"], "TEST_NLI_001_C1")
        self.assertEqual(match1["chunk_id"], "chunk_001")
        self.assertEqual(match1["claim_text"], "Classical conditioning associates stimuli.")
        self.assertEqual(
            match1["chunk_text"],
            "Classical conditioning is an associative learning process where stimuli are paired.",
        )
        self.assertTrue(match1["in_llm_context"])
        self.assertEqual(match1["section"], "6.2 Classical Conditioning")
        self.assertEqual(match1["section_path"], "learning/classical_conditioning")
        self.assertEqual(match1["chapter"], "Chapter 6 Learning")
        self.assertEqual(match1["page_start"], 195)
        self.assertEqual(match1["page_end"], 196)
        self.assertEqual(match1["pages"], [195, 196])
        self.assertEqual(match1["rrf_score"], 0.033333)

    def test_4_in_llm_context_flag_is_correct(self):
        """TEST 4: Verify in_llm_context reflects presence in context_chunks."""
        snapshot_partial = {
            "query_id": "TEST_PARTIAL",
            "question": "Question?",
            "answer": "Answer fact.",
            "retrieved_chunks": [self.dummy_chunk_1, self.dummy_chunk_2],
            "context_chunks": [self.dummy_chunk_1],  # chunk_2 excluded from context
            "final_context": "[Source 1] ...",
            "metadata": {
                "max_context_chars": 20000,
                "retrieved_chunk_count": 2,
                "context_chunk_count": 1,
            },
        }
        result = compute_nli_for_snapshot(
            snapshot_partial, model_bundle=self.mock_bundle, similarity_dir=None
        )
        matches = result["claims"][0]["evidence_matches"]
        self.assertTrue(matches[0]["in_llm_context"])
        self.assertFalse(matches[1]["in_llm_context"])

    def test_5_no_threshold_or_verdict_fields_produced(self):
        """TEST 5: Verify no threshold, verdict, supported, or hallucination fields exist."""
        result = compute_nli_for_snapshot(
            self.sample_snapshot, model_bundle=self.mock_bundle, similarity_dir=None
        )
        prohibited_terms = {
            "verdict",
            "supported",
            "unsupported",
            "hallucinated",
            "hallucination",
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
                    if isinstance(obj, list):
                        for idx, item in enumerate(obj):
                            check_no_prohibited(item, f"{path}[{idx}]")

        check_no_prohibited(result)

    def test_6_empty_answer_handled_gracefully(self):
        """TEST 6: Verify empty answer produces 0 claims without errors."""
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
        result = compute_nli_for_snapshot(
            empty_snap, model_bundle=self.mock_bundle, similarity_dir=None
        )
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["metadata"]["claim_count"], 0)

    def test_7_process_snapshot_file_end_to_end(self):
        """TEST 7: Verify process_snapshot_file writes valid JSON output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "test_snap.json"
            out_dir = Path(tmpdir) / "nli_scores"

            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(self.sample_snapshot, f)

            out_file = process_snapshot_file(
                snapshot_path=snap_path,
                output_dir=out_dir,
                similarity_dir=None,
                model_bundle=self.mock_bundle,
            )
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["query_id"], "TEST_NLI_001")
            self.assertEqual(len(loaded["claims"]), 2)


if __name__ == "__main__":
    unittest.main()
