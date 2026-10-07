"""
verification_1: Gemini-Based Atomic Claim Decomposition

Decomposes generated RAG answers into atomic, self-contained, independently verifiable
factual claims using the Gemini API (via google-genai SDK).

Preserves all factual propositions while isolating non-factual status text (e.g. abstentions).
Maintains seamless backwards compatibility with downstream similarity and NLI matchers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging
import os
import re
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default Gemini model according to modern API standards
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"

# Global client cache
_GEMINI_CLIENT = None

# Non-claim status phrases for heuristic detection & fallback
NON_CLAIM_PATTERNS = [
    r"^not found in the (?:provided )?textbook\.?$",
    r"^the provided (?:context|textbook|text) does not (?:contain|mention|provide)",
    r"^i (?:cannot|can not|could not) find",
    r"^no information (?:is )?provided",
    r"^information (?:is )?not available",
    r"^there is no mention",
    r"^unable to answer",
]

DECOMPOSER_SYSTEM_PROMPT = """You are an expert factual claim decomposition system for an academic psychology RAG assistant.

Your task is to decompose a given generated answer into a list of atomic, self-contained, independently verifiable factual claims.

### CORE REQUIREMENTS:
1. ATOMIC DECOMPOSITION:
   - Break compound or complex sentences into individual atomic factual propositions.
   - Each claim must represent exactly ONE standalone factual statement that can be verified independently against textbook evidence.
   - Do NOT split facts so aggressively that meaning is distorted or important contextual qualifiers are lost.

2. PRESERVE EVERY FACT:
   - Every single factual assertion in the original answer must be represented in the output claims.
   - Do not omit, generalize, or merge separate facts together.

3. CONTEXT & SELF-CONTAINMENT:
   - Each atomic claim must be fully understandable on its own without needing surrounding sentences.
   - Resolve pronouns (e.g., replace "He", "It", "They", "This" with the specific named entity or concept from context).
   - Preserve essential qualifiers, conditions, and technical terms (e.g., "typically", "often", "in Stage 2", "under high stress").

4. NON-FACTUAL / STATUS TEXT:
   - Do NOT convert non-factual statements into claims.
   - Identify and separate text such as:
     * Abstentions (e.g. "Not found in the provided textbook.", "I cannot find...")
     * Disclaimers
     * Conversational filler (e.g. "Here is the summary:")
     * Questions, greetings, or formatting headers
   - Place all non-factual statements into the "non_claim_text" list.

5. STRICT NEUTRALITY:
   - Do NOT verify whether the claims are true or false.
   - Do NOT correct errors or hallucinated facts in the answer.
   - Do NOT introduce outside knowledge not present in the answer.

### REQUIRED OUTPUT FORMAT:
Return ONLY a valid JSON object conforming strictly to this schema:
{
  "claims": [
    {
      "claim_id": "claim_1",
      "claim_text": "Atomic claim proposition here.",
      "source_sentence": "The exact original sentence from which this claim was extracted."
    }
  ],
  "non_claim_text": [
    "Any non-factual or status text here."
  ]
}

If the input is an abstention or contains no factual claims, return:
{
  "claims": [],
  "non_claim_text": ["Not found in the provided textbook."]
}"""


def is_non_claim_text(text: str) -> bool:
    """
    Checks if a text snippet represents an abstention, disclaimer, or non-factual status message.
    """
    cleaned = text.strip().lower()
    for pat in NON_CLAIM_PATTERNS:
        if re.search(pat, cleaned):
            return True
    return False


def get_gemini_client(api_key: Optional[str] = None):
    """
    Initializes and caches the Google GenAI client.
    Reads API key from argument, GEMINI_API_KEY, or GOOGLE_API_KEY.
    """
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None and api_key is None:
        return _GEMINI_CLIENT

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=key)
        if api_key is None:
            _GEMINI_CLIENT = client
        return client
    except Exception as e:
        logger.warning(f"Failed to initialize google-genai Client: {e}")
        return None


def decompose_with_gemini(
    answer: str,
    client: Optional[Any] = None,
    model_name: str = DEFAULT_GEMINI_MODEL
) -> Dict[str, Any]:
    """
    Calls Gemini API to perform atomic claim decomposition with structured JSON output.
    Attempts model_name, and falls back to candidate Gemini models if 503/404 occurs.
    """
    if client is None:
        client = get_gemini_client()

    if client is None:
        raise RuntimeError("Gemini client is not available. Please configure GEMINI_API_KEY.")

    from google.genai import types

    prompt = f"{DECOMPOSER_SYSTEM_PROMPT}\n\nGenerated RAG Answer to Decompose:\n\"\"\"\n{answer}\n\"\"\""

    candidate_models = [model_name, "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.1-pro-preview"]
    seen_models = set()
    last_error = None

    for m in candidate_models:
        if m in seen_models:
            continue
        seen_models.add(m)
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            response_text = response.text if hasattr(response, "text") else str(response)
            parsed = parse_decomposition_response(response_text)
            return parsed
        except Exception as e:
            last_error = e
            logger.info(f"Gemini model {m} attempt failed: {e}. Trying next candidate...")

    raise last_error or RuntimeError("All Gemini candidate models failed.")


def parse_decomposition_response(response_text: str) -> Dict[str, Any]:
    """
    Robustly parses and validates JSON returned by Gemini for atomic claim decomposition.
    """
    clean_text = response_text.strip()
    
    # Strip markdown code fences if present
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        data = json.loads(clean_text)
    except Exception as e:
        # Fallback regex extraction of JSON object
        json_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except Exception:
                raise ValueError(f"Malformed JSON in Gemini response: {e}\nResponse was:\n{response_text}")
        else:
            raise ValueError(f"Could not parse JSON from Gemini response: {e}\nResponse was:\n{response_text}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got: {type(data)}")

    claims = data.get("claims", [])
    non_claim_text = data.get("non_claim_text", [])

    if not isinstance(claims, list):
        claims = []
    if not isinstance(non_claim_text, list):
        non_claim_text = [str(non_claim_text)] if non_claim_text else []

    # Validate each claim object
    valid_claims = []
    for idx, c in enumerate(claims):
        if isinstance(c, dict):
            c_text = str(c.get("claim_text", "")).strip()
            if c_text:
                cid = str(c.get("claim_id", f"claim_{idx+1}")).strip()
                src_sent = str(c.get("source_sentence", "")).strip()
                valid_claims.append({
                    "claim_id": cid,
                    "claim_text": c_text,
                    "source_sentence": src_sent or c_text
                })
        elif isinstance(c, str) and c.strip():
            valid_claims.append({
                "claim_id": f"claim_{idx+1}",
                "claim_text": c.strip(),
                "source_sentence": c.strip()
            })

    return {
        "claims": valid_claims,
        "non_claim_text": [str(x).strip() for x in non_claim_text if str(x).strip()]
    }


def fallback_rule_based_decomposition(answer: str) -> Dict[str, Any]:
    """
    Deterministic rule-based fallback for claim decomposition when Gemini API
    is unavailable or fails. Uses sentence splitting with non-claim filtering.
    """
    if not isinstance(answer, str) or not answer.strip():
        return {"claims": [], "non_claim_text": []}

    raw_text = answer.strip()
    
    # Check if entire answer is an abstention
    if is_non_claim_text(raw_text):
        return {"claims": [], "non_claim_text": [raw_text]}

    # Split into candidate sentences
    sentences = re.split(r"(?<=[.!?])\s+|\n+", raw_text)
    claims: List[Dict[str, Any]] = []
    non_claims: List[str] = []

    c_idx = 1
    for s in sentences:
        s_clean = s.strip()
        # Remove bullet markers
        s_clean = re.sub(r"^[-*•]\s*", "", s_clean).strip()
        if not s_clean:
            continue

        if is_non_claim_text(s_clean):
            non_claims.append(s_clean)
        else:
            claims.append({
                "claim_id": f"claim_{c_idx}",
                "claim_text": s_clean,
                "source_sentence": s_clean
            })
            c_idx += 1

    return {"claims": claims, "non_claim_text": non_claims}


def find_sentence_index(source_sentence: str, claim_text: str, orig_sentences: List[str]) -> int:
    """
    Finds the 1-based index of the original source sentence in orig_sentences.
    """
    if not orig_sentences:
        return 1

    src_clean = source_sentence.strip().lower()

    # 1. Exact or normalized equality match
    for idx, s in enumerate(orig_sentences, 1):
        if src_clean == s.strip().lower():
            return idx

    # 2. Substring containment match
    for idx, s in enumerate(orig_sentences, 1):
        s_clean = s.strip().lower()
        if src_clean and (src_clean in s_clean or s_clean in src_clean):
            return idx

    # 3. Overlap / token matching with source_sentence or claim_text
    target_text = src_clean if src_clean else claim_text.strip().lower()
    target_words = set(re.findall(r"\w+", target_text))

    best_idx = 1
    best_overlap = -1
    for idx, s in enumerate(orig_sentences, 1):
        s_words = set(re.findall(r"\w+", s.strip().lower()))
        overlap = len(target_words & s_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    return best_idx


def extract_claims(
    answer: str,
    query_id: str = "Q",
    client: Optional[Any] = None,
    use_gemini: bool = True,
    model_name: str = DEFAULT_GEMINI_MODEL
) -> Dict[str, Any]:
    """
    Decomposes an answer string into atomic claim units.
    
    Uses Gemini API when available, and seamlessly falls back to rule-based decomposition
    if Gemini is offline or unconfigured.

    Args:
        answer: The text of the generated answer.
        query_id: A unique identifier for the query/answer.
        client: Optional pre-configured Google GenAI client or mock client.
        use_gemini: Whether to attempt Gemini decomposition.
        model_name: Model identifier to use.

    Returns:
        Standardized dictionary containing query_id, answer, claims list, and non_claim_text list.
        Each claim object includes 'claim_id', 'claim_text', 'text', 'source_sentence', and 'sentence_index'
        (representing the index of the original source sentence) for complete compatibility with all downstream matchers.
    """
    if not isinstance(answer, str) or not answer.strip():
        return {
            "query_id": query_id,
            "answer": answer if isinstance(answer, str) else "",
            "claims": [],
            "non_claim_text": []
        }

    raw_text = answer.strip()

    # Fast-path check for explicit abstentions
    if is_non_claim_text(raw_text):
        return {
            "query_id": query_id,
            "answer": answer,
            "claims": [],
            "non_claim_text": [raw_text]
        }

    decomp_result = None

    if use_gemini:
        try:
            decomp_result = decompose_with_gemini(raw_text, client=client, model_name=model_name)
        except Exception as e:
            logger.warning(f"Gemini claim decomposition failed ({e}). Falling back to rule-based decomposition.")
            decomp_result = None

    if decomp_result is None:
        decomp_result = fallback_rule_based_decomposition(raw_text)

    # Split original answer into sentences to map claims to their 1-based source sentence index
    orig_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", raw_text) if s.strip()]

    # Format claims with standardized keys for downstream compatibility
    standardized_claims: List[Dict[str, Any]] = []
    for idx, c in enumerate(decomp_result.get("claims", [])):
        claim_text = c.get("claim_text", "")
        src_sent = c.get("source_sentence", claim_text)
        claim_num = idx + 1
        claim_id = f"{query_id}_C{claim_num}" if query_id else f"claim_{claim_num}"
        sent_idx = find_sentence_index(src_sent, claim_text, orig_sentences)

        standardized_claims.append({
            "claim_id": claim_id,
            "claim_text": claim_text,
            "text": claim_text,  # Maintained for downstream matchers (similarity_matcher, nli_matcher)
            "source_sentence": src_sent,
            "sentence_index": sent_idx
        })

    return {
        "query_id": query_id,
        "answer": answer,
        "claims": standardized_claims,
        "non_claim_text": decomp_result.get("non_claim_text", [])
    }


def process_answers_file(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    use_gemini: bool = True
) -> List[Dict[str, Any]]:
    """
    Processes a JSON file containing a list of answer objects and writes
    the atomic claim units to the output JSON file.
    """
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of answer objects.")

    results: List[Dict[str, Any]] = []
    total_claims = 0

    for item in data:
        query_id = str(item.get("query_id", item.get("ID", "")))
        answer = str(item.get("answer", ""))
        extracted = extract_claims(answer=answer, query_id=query_id, use_gemini=use_gemini)
        results.append(extracted)
        total_claims += len(extracted["claims"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Processed {len(results)} answers -> Extracted {total_claims} claims -> Saved to {output_path}"
    )
    return results


def main():
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "input" / "test_answers.json"
    output_path = base_dir / "output" / "claims.json"

    logger.info(f"Running atomic claim extractor...")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    if input_path.exists():
        process_answers_file(input_path, output_path)
    else:
        sample = "Classical conditioning is a form of learning in which a neutral stimulus becomes associated with an unconditioned stimulus and eventually produces a conditioned response."
        sample_res = extract_claims(sample, "DEMO_01", use_gemini=False)
        print(json.dumps(sample_res, indent=2))


if __name__ == "__main__":
    main()
