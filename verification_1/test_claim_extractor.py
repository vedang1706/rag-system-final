"""
Unit tests for verification_1 Atomic Claim Decomposition with Gemini.

Tests all 8 required evaluation scenarios:
1. A simple sentence containing one factual claim.
2. A sentence containing multiple factual claims.
3. A multi-sentence answer containing several claims.
4. Facts that should remain separate because they may require different evidence.
5. An abstention such as "Not found in the provided textbook."
6. A claim containing important qualifiers/context that must not be lost.
7. An answer containing both factual claims and non-claim text.
8. Gemini returning invalid JSON / unexpected output.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from verification_1.claim_extractor import (
    extract_claims,
    decompose_with_gemini,
    parse_decomposition_response,
    fallback_rule_based_decomposition,
    is_non_claim_text,
    process_answers_file
)


class MockGeminiResponse:
    def __init__(self, text: str):
        self.text = text


class TestAtomicClaimExtractor(unittest.TestCase):

    # --------------------------------------------------------------------------
    # Test 1: Simple sentence containing one factual claim
    # --------------------------------------------------------------------------
    def test_1_simple_sentence_one_claim(self):
        """Test that a simple sentence produces one clear atomic claim."""
        query_id = "TEST_SIMPLE"
        answer = "Wilhelm Wundt founded the first psychology laboratory in Leipzig in 1879."

        # Mock Gemini response
        mock_json = json.dumps({
            "claims": [
                {
                    "claim_id": "claim_1",
                    "claim_text": "Wilhelm Wundt founded the first psychology laboratory in Leipzig in 1879.",
                    "source_sentence": answer
                }
            ],
            "non_claim_text": []
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_json)

        result = extract_claims(answer, query_id=query_id, client=mock_client)

        self.assertEqual(result["query_id"], query_id)
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["claim_id"], "TEST_SIMPLE_C1")
        self.assertEqual(result["claims"][0]["claim_text"], answer)
        self.assertEqual(result["claims"][0]["text"], answer)
        self.assertEqual(result["non_claim_text"], [])

    # --------------------------------------------------------------------------
    # Test 2: Sentence containing multiple factual claims
    # --------------------------------------------------------------------------
    def test_2_sentence_with_multiple_atomic_claims(self):
        """Test breaking a compound sentence into distinct atomic claims."""
        query_id = "TEST_COMPOUND"
        answer = (
            "Classical conditioning is a form of learning in which a neutral stimulus "
            "becomes associated with an unconditioned stimulus and eventually produces a conditioned response."
        )

        mock_json = json.dumps({
            "claims": [
                {
                    "claim_id": "claim_1",
                    "claim_text": "Classical conditioning is a form of learning.",
                    "source_sentence": answer
                },
                {
                    "claim_id": "claim_2",
                    "claim_text": "A neutral stimulus becomes associated with an unconditioned stimulus in classical conditioning.",
                    "source_sentence": answer
                },
                {
                    "claim_id": "claim_3",
                    "claim_text": "In classical conditioning, the neutral stimulus eventually produces a conditioned response.",
                    "source_sentence": answer
                }
            ],
            "non_claim_text": []
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_json)

        result = extract_claims(answer, query_id=query_id, client=mock_client)

        self.assertEqual(len(result["claims"]), 3)
        self.assertEqual(result["claims"][0]["claim_id"], "TEST_COMPOUND_C1")
        self.assertEqual(result["claims"][0]["claim_text"], "Classical conditioning is a form of learning.")
        self.assertEqual(result["claims"][0]["sentence_index"], 1)
        self.assertEqual(result["claims"][1]["claim_id"], "TEST_COMPOUND_C2")
        self.assertEqual(result["claims"][1]["sentence_index"], 1)
        self.assertEqual(result["claims"][2]["claim_id"], "TEST_COMPOUND_C3")
        self.assertEqual(result["claims"][2]["sentence_index"], 1)
        self.assertEqual(result["non_claim_text"], [])

    # --------------------------------------------------------------------------
    # Test 3: Multi-sentence answer containing several claims
    # --------------------------------------------------------------------------
    def test_3_multi_sentence_answer(self):
        """Test decomposing a full paragraph with multiple multi-fact sentences."""
        query_id = "TEST_MULTI_SENT"
        s1 = "The brain contains billions of neurons."
        s2 = "Neurons communicate across synapses using chemical neurotransmitters."
        answer = f"{s1} {s2}"

        mock_json = json.dumps({
            "claims": [
                {
                    "claim_id": "claim_1",
                    "claim_text": "The brain contains billions of neurons.",
                    "source_sentence": s1
                },
                {
                    "claim_id": "claim_2",
                    "claim_text": "Neurons communicate across synapses.",
                    "source_sentence": s2
                },
                {
                    "claim_id": "claim_3",
                    "claim_text": "Neuronal communication across synapses utilizes chemical neurotransmitters.",
                    "source_sentence": s2
                }
            ],
            "non_claim_text": []
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_json)

        result = extract_claims(answer, query_id=query_id, client=mock_client)

        self.assertEqual(len(result["claims"]), 3)
        self.assertEqual(result["claims"][0]["source_sentence"], s1)
        self.assertEqual(result["claims"][0]["sentence_index"], 1)
        self.assertEqual(result["claims"][1]["source_sentence"], s2)
        self.assertEqual(result["claims"][1]["sentence_index"], 2)
        self.assertEqual(result["claims"][2]["source_sentence"], s2)
        self.assertEqual(result["claims"][2]["sentence_index"], 2)

    # --------------------------------------------------------------------------
    # Test 3b: Explicit verification of sentence_index with 6 claims across 2 sentences
    # --------------------------------------------------------------------------
    def test_sentence_index_source_sentence_mapping(self):
        """Test that multiple atomic claims from the same sentence share the same sentence_index."""
        query_id = "TEST_OPERANT"
        s1 = "Operant conditioning is a type of associative learning in which an organism learns to link a behavior with its consequence."
        s2 = "A consequence that is pleasant, known as reinforcement, increases the likelihood of the behavior being repeated, while a consequence that is unpleasant, known as punishment, decreases the likelihood of that behavior being repeated."
        answer = f"{s1} {s2}"

        mock_json = json.dumps({
            "claims": [
                {"claim_id": "claim_1", "claim_text": "Operant conditioning is a type of associative learning.", "source_sentence": s1},
                {"claim_id": "claim_2", "claim_text": "In operant conditioning, an organism learns to link a behavior with its consequence.", "source_sentence": s1},
                {"claim_id": "claim_3", "claim_text": "A pleasant consequence of a behavior is known as reinforcement.", "source_sentence": s2},
                {"claim_id": "claim_4", "claim_text": "Reinforcement increases the likelihood of a behavior being repeated.", "source_sentence": s2},
                {"claim_id": "claim_5", "claim_text": "An unpleasant consequence of a behavior is known as punishment.", "source_sentence": s2},
                {"claim_id": "claim_6", "claim_text": "Punishment decreases the likelihood of a behavior being repeated.", "source_sentence": s2}
            ],
            "non_claim_text": []
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_json)

        result = extract_claims(answer, query_id=query_id, client=mock_client)

        self.assertEqual(len(result["claims"]), 6)
        # Check claim_id uniqueness and sequence
        expected_ids = [f"{query_id}_C{i}" for i in range(1, 7)]
        self.assertEqual([c["claim_id"] for c in result["claims"]], expected_ids)

        # Check sentence_index mapping: C1, C2 -> 1; C3..C6 -> 2
        expected_indices = [1, 1, 2, 2, 2, 2]
        self.assertEqual([c["sentence_index"] for c in result["claims"]], expected_indices)

    # --------------------------------------------------------------------------
    # Test 4: Facts requiring different evidence remain strictly separate
    # --------------------------------------------------------------------------
    def test_4_facts_requiring_different_evidence_remain_separate(self):
        """Ensure distinct propositions are not summarized or merged together."""
        query_id = "TEST_SEPARATE_EVIDENCE"
        answer = "Stage 2 sleep features sleep spindles, while Stage 3 is characterized by slow delta waves."

        mock_json = json.dumps({
            "claims": [
                {
                    "claim_id": "claim_1",
                    "claim_text": "Stage 2 sleep features sleep spindles.",
                    "source_sentence": answer
                },
                {
                    "claim_id": "claim_2",
                    "claim_text": "Stage 3 sleep is characterized by slow delta waves.",
                    "source_sentence": answer
                }
            ],
            "non_claim_text": []
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_json)

        result = extract_claims(answer, query_id=query_id, client=mock_client)

        self.assertEqual(len(result["claims"]), 2)
        # Verify that Stage 2 and Stage 3 facts are isolated
        self.assertIn("Stage 2", result["claims"][0]["claim_text"])
        self.assertIn("Stage 3", result["claims"][1]["claim_text"])

    # --------------------------------------------------------------------------
    # Test 5: Abstention such as "Not found in the provided textbook."
    # --------------------------------------------------------------------------
    def test_5_abstention_produces_no_claims(self):
        """Test that explicit abstention phrases are isolated to non_claim_text with 0 claims."""
        abstentions = [
            "Not found in the provided textbook.",
            "Not found in the textbook",
            "The provided context does not contain information about this topic."
        ]

        for ans in abstentions:
            result = extract_claims(ans, query_id="TEST_ABSTAIN", use_gemini=False)
            self.assertEqual(len(result["claims"]), 0, f"Expected 0 claims for '{ans}'")
            self.assertEqual(len(result["non_claim_text"]), 1)
            self.assertEqual(result["non_claim_text"][0], ans)

    # --------------------------------------------------------------------------
    # Test 6: Claim containing important qualifiers/context preserved
    # --------------------------------------------------------------------------
    def test_6_important_qualifiers_preserved(self):
        """Test that necessary contextual qualifiers and conditions are not lost."""
        query_id = "TEST_QUALIFIERS"
        answer = "Under chronic high stress, cortisol levels typically remain elevated, which impairs immune functioning."

        mock_json = json.dumps({
            "claims": [
                {
                    "claim_id": "claim_1",
                    "claim_text": "Under chronic high stress, cortisol levels typically remain elevated.",
                    "source_sentence": answer
                },
                {
                    "claim_id": "claim_2",
                    "claim_text": "Chronically elevated cortisol levels impair immune functioning.",
                    "source_sentence": answer
                }
            ],
            "non_claim_text": []
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_json)

        result = extract_claims(answer, query_id=query_id, client=mock_client)

        self.assertEqual(len(result["claims"]), 2)
        self.assertIn("Under chronic high stress", result["claims"][0]["claim_text"])
        self.assertIn("impair", result["claims"][1]["claim_text"])

    # --------------------------------------------------------------------------
    # Test 7: Answer containing both factual claims and non-claim text
    # --------------------------------------------------------------------------
    def test_7_mixed_factual_and_non_claim_text(self):
        """Test that answers with conversational preambles or disclaimers separate them properly."""
        query_id = "TEST_MIXED"
        answer = "Based on the textbook: Operant conditioning relies on reinforcement and punishment."

        mock_json = json.dumps({
            "claims": [
                {
                    "claim_id": "claim_1",
                    "claim_text": "Operant conditioning relies on reinforcement.",
                    "source_sentence": "Operant conditioning relies on reinforcement and punishment."
                },
                {
                    "claim_id": "claim_2",
                    "claim_text": "Operant conditioning relies on punishment.",
                    "source_sentence": "Operant conditioning relies on reinforcement and punishment."
                }
            ],
            "non_claim_text": ["Based on the textbook:"]
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_json)

        result = extract_claims(answer, query_id=query_id, client=mock_client)

        self.assertEqual(len(result["claims"]), 2)
        self.assertEqual(len(result["non_claim_text"]), 1)
        self.assertEqual(result["non_claim_text"][0], "Based on the textbook:")

    # --------------------------------------------------------------------------
    # Test 8: Gemini returning invalid JSON / unexpected output
    # --------------------------------------------------------------------------
    def test_8_gemini_invalid_json_fallback(self):
        """Test graceful fallback to rule-based decomposition when Gemini returns invalid JSON."""
        query_id = "TEST_FALLBACK"
        answer = "Ivan Pavlov studied digestive systems. He later discovered classical conditioning."

        # Invalid JSON string returned by API
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse("This is not JSON at all! { broken ...")

        result = extract_claims(answer, query_id=query_id, client=mock_client)

        # Pipeline should not crash, and should fall back gracefully
        self.assertEqual(result["query_id"], query_id)
        self.assertGreaterEqual(len(result["claims"]), 1)
        self.assertTrue(any("Ivan Pavlov" in c["claim_text"] for c in result["claims"]))


if __name__ == "__main__":
    unittest.main()
