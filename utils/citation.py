# utils/citation.py

import re
from typing import List, Dict


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Minimum overlap score to consider a chunk as a citation source
CITATION_THRESHOLD = 0.15

# Maximum citations to show per answer
MAX_CITATIONS = 3

# Words to ignore when calculating overlap
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "but", "or", "nor", "so",
    "yet", "both", "either", "neither", "not", "only", "own",
    "same", "than", "too", "very", "just", "that", "this", "these",
    "those", "it", "its", "we", "they", "them", "their", "what",
    "which", "who", "whom", "how", "all", "each", "more", "most",
    "other", "some", "such", "no", "up", "about", "also"
}


# ─────────────────────────────────────────────
# FUNCTION 1 — Tokenize Text
# ─────────────────────────────────────────────

def tokenize(text: str) -> set:
    """
    Converts text to a set of meaningful lowercase words.
    Removes stopwords and short words.
    """
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in STOPWORDS}


# ─────────────────────────────────────────────
# FUNCTION 2 — Score Chunk Against Answer
# ─────────────────────────────────────────────

def score_chunk(answer: str, chunk_text: str) -> float:
    """
    Scores how much a chunk contributed to the answer.

    Method: Jaccard-style overlap between answer words
    and chunk words. Higher = more overlap = more likely
    this chunk was used to generate the answer.

    Returns score between 0.0 and 1.0
    """
    answer_words = tokenize(answer)
    chunk_words  = tokenize(chunk_text)

    if not answer_words or not chunk_words:
        return 0.0

    # Count how many answer words appear in this chunk
    overlap = answer_words & chunk_words

    # Score = overlap / answer words
    # (what fraction of the answer's key terms came from this chunk)
    score = len(overlap) / len(answer_words)

    return round(score, 4)


# ─────────────────────────────────────────────
# FUNCTION 3 — Find Exact Citations
# ─────────────────────────────────────────────

def find_exact_citations(answer: str,
                          retrieved_chunks: List[Dict],
                          threshold: float = CITATION_THRESHOLD,
                          max_citations: int = MAX_CITATIONS
                          ) -> List[Dict]:
    """
    Finds which specific chunks the answer was derived from.

    Process:
    1. Score each chunk against the answer
    2. Keep only chunks above threshold
    3. Sort by score descending
    4. Return top max_citations

    Each returned citation includes:
    - chunk metadata (section, chapter, pages)
    - overlap score (confidence)
    - evidence snippet (matching text preview)
    """
    if not answer or "Not found" in answer:
        return []

    scored = []

    for chunk in retrieved_chunks:
        score = score_chunk(answer, chunk['text'])

        if score >= threshold:
            # Find evidence snippet — sentence in chunk
            # that best matches the answer
            evidence = find_evidence_snippet(answer, chunk['text'])

            scored.append({
                "chunk_id":     chunk['chunk_id'],
                "section":      chunk['section'],
                "section_path": chunk['section_path'],
                "chapter":      chunk['chapter'],
                "page_start":   chunk['page_start'],
                "page_end":     chunk['page_end'],
                "pages":        chunk['pages'],
                "score":        score,
                "evidence":     evidence,
                "source":       chunk.get('source', 'unknown')
            })

    # Sort by score descending
    scored.sort(key=lambda x: x['score'], reverse=True)

    return scored[:max_citations]


# ─────────────────────────────────────────────
# FUNCTION 4 — Find Evidence Snippet
# ─────────────────────────────────────────────

def find_evidence_snippet(answer: str,
                           chunk_text: str,
                           max_length: int = 200) -> str:
    """
    Finds the sentence in the chunk that best matches
    the answer. This is the "evidence" shown to users
    to prove the citation is accurate.
    """
    answer_words = tokenize(answer)

    # Split chunk into sentences
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text)

    best_sentence = ""
    best_score    = 0

    for sentence in sentences:
        if len(sentence) < 20:
            continue

        sentence_words = tokenize(sentence)
        overlap        = answer_words & sentence_words

        if len(answer_words) > 0:
            score = len(overlap) / len(answer_words)
        else:
            score = 0

        if score > best_score:
            best_score    = score
            best_sentence = sentence

    # Truncate if too long
    if len(best_sentence) > max_length:
        best_sentence = best_sentence[:max_length] + "..."

    return best_sentence if best_sentence else chunk_text[:max_length] + "..."


# ─────────────────────────────────────────────
# FUNCTION 5 — Format Citations For Display
# ─────────────────────────────────────────────

def format_citations(citations: List[Dict]) -> List[Dict]:
    """
    Formats citation data for clean display in the UI.
    Converts section paths to readable strings.
    """
    formatted = []

    for i, cite in enumerate(citations):
        section_readable = (cite['section_path']
                            .replace('_', ' ')
                            .replace('/', ' > '))

        confidence = "High" if cite['score'] > 0.35 else \
                     "Medium" if cite['score'] > 0.20 else "Low"

        formatted.append({
            "index":      i + 1,
            "section":    cite['section'],
            "chapter":    cite['chapter'],
            "pages":      cite['pages'],
            "page_start": cite['page_start'],
            "page_end":   cite['page_end'],
            "path":       section_readable,
            "evidence":   cite['evidence'],
            "confidence": confidence,
            "score":      cite['score']
        })

    return formatted


# ─────────────────────────────────────────────
# FUNCTION 6 — Build Reference JSON
# ─────────────────────────────────────────────

def build_references_from_citations(citations: List[Dict]) -> dict:
    """
    Builds the final reference JSON from exact citations.
    Used for submission.csv references column.

    Only includes sections and pages from chunks
    that actually contributed to the answer.
    """
    if not citations:
        return {"sections": [], "pages": []}

    sections = []
    pages    = set()

    for cite in citations:
        if cite['section_path'] not in sections:
            sections.append(cite['section_path'])
        for page in cite['pages']:
            pages.add(page)

    return {
        "sections": sections,
        "pages":    sorted(list(pages))
    }