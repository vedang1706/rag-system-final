"""
Unit tests for verification_1 ModernBERT Long-Context NLI Matcher (dleemiller/ModernCE-base-nli).
"""

import json
import tempfile
import unittest
from pathlib import Path
import torch

from verification_1.modern_nli_matcher import (
    MODERN_NLI_MODEL_NAME,
    MODERN_NLI_LABEL_MAP,
    MODERN_NLI_ID2LABEL,
    get_modern_nli_model,
    predict_pair,
    compute_modern_nli_scores_batch,
    compute_modern_nli_for_snapshot,
    process_snapshot_file,
)


class TestModernNLIMatcher(unittest.TestCase):
    """Test suite for ModernBERT cross-encoder NLI matching and long-context handling."""

    @classmethod
    def setUpClass(cls):
        """Load the model once for the test suite."""
        cls.bundle = get_modern_nli_model()
        cls.tokenizer, cls.model, cls.label_map = cls.bundle

    def test_1_model_initialization_and_label_mapping(self):
        """TEST 1: Verify model initialization, architecture, and verified 3-class label map."""
        self.assertIsNotNone(self.tokenizer)
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.label_map), 3)
        self.assertIn("contradiction", self.label_map)
        self.assertIn("entailment", self.label_map)
        self.assertIn("neutral", self.label_map)
        # Verify specific indices: contradiction=0, entailment=1, neutral=2
        self.assertEqual(self.label_map["contradiction"], 0)
        self.assertEqual(self.label_map["entailment"], 1)
        self.assertEqual(self.label_map["neutral"], 2)

    def test_2_single_pair_output_structure_and_valid_probabilities(self):
        """TEST 2: Verify predict_pair returns structured dictionary with valid probabilities."""
        premise = "Classical conditioning is a learning process where two stimuli are paired."
        hypothesis = "Classical conditioning involves pairing stimuli."

        result = predict_pair(premise, hypothesis, model_bundle=self.bundle)

        self.assertIn("premise", result)
        self.assertIn("hypothesis", result)
        self.assertIn("token_count", result)
        self.assertIn("raw_logits", result)
        self.assertIn("entailment_score", result)
        self.assertIn("neutral_score", result)
        self.assertIn("contradiction_score", result)
        self.assertIn("predicted_label", result)

        self.assertEqual(len(result["raw_logits"]), 3)
        self.assertIn(result["predicted_label"], ["contradiction", "neutral", "entailment"])

        # Probability bounds
        for key in ["entailment_score", "neutral_score", "contradiction_score"]:
            val = result[key]
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

        prob_sum = result["entailment_score"] + result["neutral_score"] + result["contradiction_score"]
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_3_known_3_way_nli_classification(self):
        """TEST 3: Verify classification accuracy on canonical Entailment, Contradiction, and Neutral cases."""
        entail_pair = (
            "Pavlov discovered classical conditioning while studying the digestive system of dogs.",
            "Pavlov conducted experiments on dogs.",
        )
        contra_pair = (
            "Pavlov discovered classical conditioning while studying the digestive system of dogs.",
            "Pavlov never performed any research on animals.",
        )
        neutral_pair = (
            "B.F. Skinner studied operant conditioning through behavioral experiments.",
            "Skinner had two children and lived in Boston.",
        )

        res_ent = predict_pair(entail_pair[0], entail_pair[1], model_bundle=self.bundle)
        res_con = predict_pair(contra_pair[0], contra_pair[1], model_bundle=self.bundle)
        res_neu = predict_pair(neutral_pair[0], neutral_pair[1], model_bundle=self.bundle)

        self.assertEqual(res_ent["predicted_label"], "entailment")
        self.assertGreater(res_ent["entailment_score"], 0.80)

        self.assertEqual(res_con["predicted_label"], "contradiction")
        self.assertGreater(res_con["contradiction_score"], 0.80)

        self.assertEqual(res_neu["predicted_label"], "neutral")
        self.assertGreater(res_neu["neutral_score"], 0.80)

    def test_4_batched_inference(self):
        """TEST 4: Verify compute_modern_nli_scores_batch correctly processes lists of pairs."""
        pairs = [
            ("A dog runs in the park.", "An animal is outdoors."),
            ("A boy is eating an apple.", "The boy is asleep in bed."),
            ("Sarah bought a car yesterday.", "The car cost fifteen thousand dollars."),
        ]
        scores = compute_modern_nli_scores_batch(pairs, model_bundle=self.bundle, batch_size=2)
        self.assertEqual(len(scores), 3)

        self.assertGreater(scores[0]["entailment_score"], 0.70)
        self.assertGreater(scores[1]["contradiction_score"], 0.70)
        self.assertGreater(scores[2]["neutral_score"], 0.70)

    def test_5_long_context_processing_exceeds_512_tokens(self):
        """
        TEST 5: Verify the implementation processes an evidence input > 512 tokens
        without crashing or applying the old 512-token truncation limit.
        """
        base_paragraph = (
            "Classical conditioning was discovered by Ivan Pavlov during his physiological research on digestion in dogs. "
            "Pavlov observed that dogs began salivating not only when food was placed in their mouths, but also at the sight "
            "of the laboratory assistant who brought the food. To investigate this systematically, Pavlov presented a neutral "
            "stimulus, such as a metronome ticking or a tone sounding, immediately before presenting meat powder. After multiple "
            "paired presentations, the previously neutral stimulus alone elicited salivation in the experimental subjects. "
        )
        # Repeat paragraph 20 times to create a ~1000-token passage
        long_premise = base_paragraph * 20
        hypothesis = "Pavlov paired a neutral sound stimulus with meat powder in conditioning trials."

        result = predict_pair(long_premise, hypothesis, model_bundle=self.bundle, max_length=2048)

        token_count = result["token_count"]
        print(f"\n[Test 5 Log] Long-context premise token count: {token_count} tokens (Threshold: > 512)")
        print(f"[Test 5 Log] Predicted label: {result['predicted_label']}, Entailment Score: {result['entailment_score']}")

        self.assertGreater(
            token_count,
            512,
            f"Expected input sequence length > 512 tokens, but got {token_count}",
        )
        self.assertEqual(result["predicted_label"], "entailment")
        self.assertGreater(result["entailment_score"], 0.85)

    def test_6_snapshot_processing_and_file_io(self):
        """TEST 6: Verify compute_modern_nli_for_snapshot and process_snapshot_file work end-to-end."""
        dummy_chunk = {
            "chunk_id": "chunk_001",
            "text": "Operant conditioning relies on reinforcement and punishment to modify voluntary behaviors.",
            "section": "6.3 Operant Conditioning",
            "section_path": "learning/operant_conditioning",
            "chapter": "Chapter 6 Learning",
            "page_start": 201,
            "page_end": 202,
            "pages": [201, 202],
            "rrf_score": 0.033333,
            "source": "both",
            "is_supplementary": False,
        }
        sample_snapshot = {
            "query_id": "TEST_MODERN_001",
            "question": "What is operant conditioning?",
            "answer": "Operant conditioning modifies behaviors through rewards and punishments.",
            "retrieved_chunks": [dummy_chunk],
            "context_chunks": [dummy_chunk],
            "final_context": "[Source 1] ...",
            "metadata": {
                "max_context_chars": 20000,
                "retrieved_chunk_count": 1,
                "context_chunk_count": 1,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = Path(tmpdir) / "test_snap.json"
            out_dir = Path(tmpdir) / "modern_nli_scores"

            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(sample_snapshot, f)

            out_file = process_snapshot_file(
                snapshot_path=snap_path,
                output_dir=out_dir,
                model_bundle=self.bundle,
                max_length=2048,
            )

            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["query_id"], "TEST_MODERN_001")
            self.assertEqual(loaded["metadata"]["nli_model"], MODERN_NLI_MODEL_NAME)
            self.assertEqual(loaded["metadata"]["max_sequence_length"], 2048)
            self.assertEqual(len(loaded["claims"]), 1)
            self.assertEqual(len(loaded["claims"][0]["evidence_matches"]), 1)


if __name__ == "__main__":
    unittest.main()
