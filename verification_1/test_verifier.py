"""
Unit & Integration Tests for ModernCE Complete-Answer Multi-Chunk Verifier (verification_1/verifier.py).

Tests:
1. Label mapping and singleton verifier instance.
2. Explicit abstention detection and bypass.
3. Supported answer verified with Top-2 evidence (empirical Q1 integration).
4. Supported answer requiring Top-3 recovery (empirical Q41 integration).
5. Unsupported answer resulting in INSUFFICIENT_EVIDENCE / NOT_ENTAILED.
6. Ranking logic and tie-breaking behavior.
7. Various chunk input formats and dictionary serialization.
"""

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List

from verification_1.verifier import (
    ModernCEVerifier,
    VerificationResult,
    verify_answer,
    is_abstention_answer,
    STATUS_SUPPORTED,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_CONTRADICTED,
    STATUS_NOT_FOUND,
    VERDICT_ENTAILED_BY_TOP2,
    VERDICT_ENTAILED_BY_TOP3,
    VERDICT_NOT_ENTAILED,
    VERDICT_ABSTENTION,
    get_verifier,
)

project_root = Path(__file__).resolve().parent.parent


def load_retrieved_context_chunks(qid: str) -> List[Dict[str, Any]]:
    """Helper to parse raw chunks from outputs/retrieved_contexts/{qid}.txt."""
    ctx_path = project_root / "outputs" / "retrieved_contexts" / f"{qid}.txt"
    if not ctx_path.exists():
        return []
    import re
    content = ctx_path.read_text(encoding="utf-8")
    blocks = re.split(r"--- Chunk (\d+) ---", content)
    chunks = []
    for i in range(1, len(blocks), 2):
        cnum = int(blocks[i])
        cbody = blocks[i+1].strip()
        raw_match = re.search(r"Raw Text:\s*\n(.*)", cbody, re.DOTALL)
        raw_text = raw_match.group(1).strip() if raw_match else ""
        chunks.append({"chunk_number": cnum, "text": raw_text})
    return chunks


class TestModernCEVerifier(unittest.TestCase):
    """Test suite for the integrated ModernCE complete-answer multi-chunk verifier."""

    @classmethod
    def setUpClass(cls):
        """Load the verifier and cached experimental results once for all tests."""
        cls.verifier = get_verifier()
        raw_results_path = project_root / "verification_1" / "output" / "modernce_50q_complete_answer" / "raw_results.json"
        if raw_results_path.exists():
            cls.cached_data = json.loads(raw_results_path.read_text(encoding="utf-8"))
        else:
            cls.cached_data = None

    def test_1_label_mapping_and_singleton(self):
        """TEST 1: Verify label mapping and singleton verifier instance."""
        self.assertIsNotNone(self.verifier.tokenizer)
        self.assertIsNotNone(self.verifier.model)
        self.assertEqual(self.verifier.label_map["contradiction"], 0)
        self.assertEqual(self.verifier.label_map["entailment"], 1)
        self.assertEqual(self.verifier.label_map["neutral"], 2)

    def test_2_explicit_abstention_handling(self):
        """TEST 2: Explicit abstention answers bypass NLI and return NOT_FOUND."""
        abstention_texts = [
            "Not found in the provided textbook.",
            "Information is not found in the provided textbook.",
            "Not found in provided textbook",
            "Not found",
        ]

        chunks = [
            {"text": "Psychology is the scientific study of mind and behavior.", "chunk_id": "c1"},
            {"text": "Sigmund Freud founded psychoanalysis.", "chunk_id": "c2"},
        ]

        for ans in abstention_texts:
            self.assertTrue(is_abstention_answer(ans), f"Failed to identify abstention: {ans}")
            res = self.verifier.verify(answer=ans, retrieved_chunks=chunks)
            self.assertEqual(res.status, STATUS_NOT_FOUND)
            self.assertEqual(res.verdict, VERDICT_ABSTENTION)
            self.assertTrue(res.is_abstention)
            self.assertIsNone(res.combination_used)
            self.assertEqual(len(res.individual_chunk_evaluations), 0)

    def test_3_supported_answer_top2_evidence(self):
        """TEST 3: Verify complete answer supported by Top-2 combined evidence (Q1 validation)."""
        if self.cached_data:
            q1_data = self.cached_data["per_question_results"][0]
            answer = q1_data["generated_answer"]
            chunks = load_retrieved_context_chunks(q1_data["question_id"])
        else:
            answer = "The scientific method in psychology is an empirical, cyclical process that begins with a theory."
            chunks = [
                {"chunk_number": 1, "text": "The scientific method is a circular process involving theories and hypotheses."},
                {"chunk_number": 2, "text": "Psychological research tests hypotheses through empirical observation."},
            ]

        res = self.verifier.verify(answer=answer, retrieved_chunks=chunks)

        self.assertEqual(res.status, STATUS_SUPPORTED)
        self.assertEqual(res.verdict, VERDICT_ENTAILED_BY_TOP2)
        self.assertEqual(res.combination_used, "TOP_2")
        self.assertFalse(res.is_abstention)
        self.assertGreater(res.entailment_prob, 0.70)
        self.assertIsNone(res.top_3_evaluation)

    def test_4_supported_answer_top3_recovery(self):
        """TEST 4: Answer requiring 3 chunks recovers from Top-2 Neutral to Top-3 Entailment (Q41 validation)."""
        if self.cached_data:
            q41_data = self.cached_data["per_question_results"][40]
            answer = q41_data["generated_answer"]
            chunks = load_retrieved_context_chunks(q41_data["question_id"])
        else:
            self.skipTest("Cached experimental data not found for Q41")

        res = self.verifier.verify(answer=answer, retrieved_chunks=chunks)

        self.assertEqual(res.status, STATUS_SUPPORTED)
        self.assertEqual(res.verdict, VERDICT_ENTAILED_BY_TOP3)
        self.assertEqual(res.combination_used, "TOP_3")
        self.assertGreater(res.entailment_prob, 0.80)
        self.assertIsNotNone(res.top_2_evaluation)
        self.assertIsNotNone(res.top_3_evaluation)
        self.assertEqual(len(res.selected_chunk_numbers), 3)

    def test_5_insufficient_evidence_persistent_neutral(self):
        """TEST 5: Generated answer with unsupported claims remains INSUFFICIENT_EVIDENCE / NOT_ENTAILED."""
        chunk1 = {
            "chunk_number": 1,
            "chunk_id": "c1",
            "text": "Wilhelm Wundt founded structuralism and used introspection to study conscious experience in Leipzig.",
        }
        chunk2 = {
            "chunk_number": 2,
            "chunk_id": "c2",
            "text": "William James was the first American psychologist and advocated functionalism.",
        }
        chunk3 = {
            "chunk_number": 3,
            "chunk_id": "c3",
            "text": "Gestalt psychology explores how people perceive sensory elements as unified wholes.",
        }

        # Unsupported answer claiming fMRI brain scans in 1879
        answer = "Wilhelm Wundt utilized functional magnetic resonance imaging (fMRI) scanners in 1879 to map digital neural activation patterns."

        res = self.verifier.verify(answer=answer, retrieved_chunks=[chunk1, chunk2, chunk3])

        self.assertIn(res.status, [STATUS_INSUFFICIENT_EVIDENCE, STATUS_CONTRADICTED])
        self.assertEqual(res.verdict, VERDICT_NOT_ENTAILED)
        self.assertNotEqual(res.status, STATUS_SUPPORTED)

    def test_6_chunk_ranking_order(self):
        """TEST 6: Individual chunk evaluation correctly ranks relevant chunk first."""
        chunk_unrelated = {
            "chunk_number": 1,
            "chunk_id": "c_unrel",
            "text": "Geology is the scientific study of the Earth, its composition, rock formations, and tectonic plates.",
        }
        chunk_relevant = {
            "chunk_number": 2,
            "chunk_id": "c_rel",
            "text": "The amygdala is a structure in the limbic system that plays a critical role in processing emotional responses, especially fear.",
        }

        answer = "The amygdala is central to emotional processing and fear responses."

        res = self.verifier.verify(answer=answer, retrieved_chunks=[chunk_unrelated, chunk_relevant])

        self.assertEqual(res.ranked_chunk_numbers[0], 2)
        self.assertEqual(res.status, STATUS_SUPPORTED)

    def test_7_various_chunk_formats_and_serialization(self):
        """TEST 7: Verifier seamlessly handles string chunks and dict chunks with serialization."""
        raw_chunks = [
            "B.F. Skinner developed the operant conditioning chamber to study animal behavior.",
            "Reinforcement increases the likelihood of a behavior being repeated.",
        ]
        answer = "B.F. Skinner investigated operant conditioning and reinforcement."

        res = verify_answer(answer=answer, retrieved_chunks=raw_chunks)
        self.assertEqual(res.status, STATUS_SUPPORTED)

        # Check serialization
        res_dict = res.to_dict()
        self.assertIsInstance(res_dict, dict)
        self.assertIn("status", res_dict)
        self.assertIn("verdict", res_dict)
        self.assertIn("entailment_prob", res_dict)
        self.assertIn("selected_chunk_numbers", res_dict)
        self.assertIn("explanation", res_dict)


if __name__ == "__main__":
    unittest.main()
