# run_extraction_metrics.py

import os
import re
import sys
import difflib
import random

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
PDF_PATH    = "data/book.pdf"
REF_PATH    = "data/reference_book_text.txt"
EXT_PATH    = "cache/metrics_pymupdf_full.txt"
REPORT_PATH = "outputs/extraction_metrics_report.txt"

output_log = []

def log(msg=""):
    print(msg)
    output_log.append(str(msg))

def save_report():
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_log))
    print(f"\n[i] Full report saved to: {REPORT_PATH}")


# ═════════════════════════════════════════════
# STEP 1 — FULL PDF EXTRACTION
# ═════════════════════════════════════════════

def extract_full_pdf():
    """
    Extracts all text from PDF using PyMuPDF.
    Independent of ingest.py — raw extraction only.
    Saves to cache/metrics_pymupdf_full.txt
    """
    import fitz

    log("  Extracting full PDF with PyMuPDF (all pages)...")

    doc      = fitz.open(PDF_PATH)
    all_text = []

    for page in doc:
        text = page.get_text()
        if not text.strip():
            continue

        lines   = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if re.match(r'^\d+$', stripped):
                continue
            if 'access for free at openstax' in stripped.lower():
                continue
            if len(stripped) < 3:
                continue
            cleaned.append(stripped)

        page_text = ' '.join(cleaned)
        page_text = re.sub(r'\s+', ' ', page_text).strip()
        all_text.append(page_text)

    doc.close()
    full_text = "\n\n".join(all_text)

    os.makedirs(os.path.dirname(EXT_PATH), exist_ok=True)
    with open(EXT_PATH, 'w', encoding='utf-8') as f:
        f.write(full_text)

    log(f"  ✓ Saved PyMuPDF extraction to : {EXT_PATH}")
    log(f"    Total characters             : {len(full_text):,}")
    log(f"    Total words                  : {len(full_text.split()):,}")
    log(f"    Total pages processed        : {len(all_text)}")
    return full_text


# ═════════════════════════════════════════════
# NORMALIZATION HELPERS
# ═════════════════════════════════════════════

def normalize_text(text):
    """
    Normalizes text for fair comparison.
    Removes formatting differences between
    Adobe CTRL+A output and PyMuPDF extraction:
    - Collapses all whitespace and newlines
    - Lowercases
    - Removes punctuation
    Both should produce same words after this.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text


# ═════════════════════════════════════════════
# METRIC 1 — VOCABULARY OVERLAP (PRIMARY)
# ═════════════════════════════════════════════

def word_vocabulary_overlap(reference_text, extracted_text):
    """
    PRIMARY METRIC for comparing two extractors
    of the same digital PDF document.

    Measures: what fraction of unique vocabulary
    in the reference appears in extraction.

    NOT affected by word order, line breaks, or formatting.
    This is what matters for RAG quality.

    Industry equivalent: Token-level F1 used in
    SQuAD QA evaluation (Rajpurkar et al., 2016)
    """
    ref_words = set(normalize_text(reference_text).split())
    ext_words = set(normalize_text(extracted_text).split())

    ref_words = {w for w in ref_words if len(w) > 2}
    ext_words = {w for w in ext_words if len(w) > 2}

    intersection = ref_words & ext_words
    missing      = ref_words - ext_words
    extra        = ext_words - ref_words

    precision = len(intersection) / len(ext_words) if ext_words else 0
    recall    = len(intersection) / len(ref_words)  if ref_words else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0)

    log(f"\n{'─'*55}")
    log("METRIC 1: Vocabulary Overlap  [PRIMARY METRIC]")
    log(f"{'─'*55}")
    log(f"  Reference unique words : {len(ref_words):,}")
    log(f"  Extracted unique words : {len(ext_words):,}")
    log(f"  Common words           : {len(intersection):,}")
    log(f"  Missing from extract   : {len(missing):,}")
    log(f"  Extra in extract       : {len(extra):,}")
    log(f"  Precision              : {precision*100:.2f}%")
    log(f"  Recall                 : {recall*100:.2f}%")
    log(f"  F1 Score               : {f1*100:.2f}%")
    log(f"  Target F1              : > 95%")

    if missing:
        sample = sorted(list(missing))[:20]
        log(f"\n  Sample missing words   : {sample}")

    return precision, recall, f1


# ═════════════════════════════════════════════
# METRIC 2 — N-GRAM PHRASE OVERLAP
# ═════════════════════════════════════════════

def ngram_overlap(reference_text, extracted_text, n=3):
    """
    Checks if common PHRASES are preserved.
    More meaningful than single words for
    extraction quality — ensures context
    is preserved, not just vocabulary.

    n=2 : bigrams (word pairs)
    n=3 : trigrams (3-word phrases) — default

    Industry equivalent: ROUGE-N metric
    (Lin, 2004) used in NLP evaluation.
    """
    def get_ngrams(text, n):
        words = normalize_text(text).split()
        return set(
            tuple(words[i:i+n])
            for i in range(len(words) - n + 1)
        )

    ref_ngrams = get_ngrams(reference_text, n)
    ext_ngrams = get_ngrams(extracted_text, n)

    if not ref_ngrams:
        return 0

    overlap    = ref_ngrams & ext_ngrams
    recall     = len(overlap) / len(ref_ngrams)
    precision  = len(overlap) / len(ext_ngrams) if ext_ngrams else 0
    f1         = (2 * precision * recall / (precision + recall)
                  if precision + recall > 0 else 0)

    log(f"\n{'─'*55}")
    log(f"METRIC 2: {n}-gram Phrase Overlap")
    log(f"{'─'*55}")
    log(f"  Reference {n}-grams    : {len(ref_ngrams):,}")
    log(f"  Extracted {n}-grams    : {len(ext_ngrams):,}")
    log(f"  Common {n}-grams       : {len(overlap):,}")
    log(f"  Recall                 : {recall*100:.2f}%")
    log(f"  Precision              : {precision*100:.2f}%")
    log(f"  F1                     : {f1*100:.2f}%")
    log(f"  Target F1              : > 80%")

    return f1


# ═════════════════════════════════════════════
# METRIC 3 — KEY PSYCHOLOGY TERM VERIFICATION
# ═════════════════════════════════════════════

def key_phrase_verification(reference_text, extracted_text):
    """
    Tests whether critical psychology terms are
    preserved in extraction.

    Direct relevance to RAG quality:
    If these terms are present, the system can
    answer psychology questions correctly.

    Based on domain-specific term coverage
    evaluation used in biomedical NLP.
    """
    key_terms = [
        "classical conditioning",
        "operant conditioning",
        "pavlov",
        "b.f. skinner",
        "maslow",
        "hierarchy of needs",
        "sigmund freud",
        "psychoanalysis",
        "short-term memory",
        "long-term memory",
        "hippocampus",
        "amygdala",
        "neurotransmitter",
        "synapse",
        "cognitive dissonance",
        "schizophrenia",
        "big five",
        "behaviorism",
        "gestalt",
        "introspection",
        "functionalism",
        "piaget",
        "erikson",
        "developmental psychology",
        "sensation",
        "perception",
        "consciousness",
        "anxiety",
        "depression",
        "personality disorder",
    ]

    ref_lower = reference_text.lower()
    ext_lower = extracted_text.lower()

    found_in_ref  = 0
    found_in_ext  = 0
    missing_terms = []

    log(f"\n{'─'*55}")
    log("METRIC 3: Key Psychology Term Verification")
    log(f"{'─'*55}")

    for term in key_terms:
        in_ref = term in ref_lower
        in_ext = term in ext_lower

        if in_ref:
            found_in_ref += 1
        if in_ext:
            found_in_ext += 1
        if in_ref and not in_ext:
            missing_terms.append(term)

        status = "OK     " if (in_ref and in_ext)   else \
                 "MISSING" if (in_ref and not in_ext) else \
                 "EXTRA  "
        log(f"  {status} : {term}")

    preservation = (found_in_ext / found_in_ref * 100
                    if found_in_ref > 0 else 0)

    log(f"\n  Terms in reference  : {found_in_ref}/{len(key_terms)}")
    log(f"  Terms in extraction : {found_in_ext}/{len(key_terms)}")
    log(f"  Preservation rate   : {preservation:.1f}%")
    log(f"  Target              : 100%")

    if missing_terms:
        log(f"\n  MISSING TERMS: {missing_terms}")
    else:
        log(f"\n  All key terms preserved ✓")

    return preservation


# ═════════════════════════════════════════════
# METRIC 4 — SENTENCE PRESERVATION SAMPLE
# ═════════════════════════════════════════════

def page_sample_verification(reference_text, extracted_text):
    """
    Samples specific sentences from reference.
    Checks if they exist in extraction.

    Direct proof that extraction captured the
    content, not just vocabulary.

    This is the most intuitive verification:
    if you can find real sentences from the book
    in the extraction, the extraction is correct.
    """
    ref_sentences = [
        s.strip()
        for s in reference_text.replace('\n', ' ').split('.')
        if len(s.strip()) > 60
    ]

    if not ref_sentences:
        log("  No sentences found in reference.")
        return 0

    sample_size = min(50, len(ref_sentences))
    samples     = random.sample(ref_sentences, sample_size)
    ext_lower   = extracted_text.lower()
    found       = 0

    log(f"\n{'─'*55}")
    log("METRIC 4: Sentence Preservation Sample")
    log(f"{'─'*55}")
    log(f"  Sampling {sample_size} sentences from reference...")
    log(f"  Checking first 60 chars of each in extraction...")

    for sentence in samples:
        probe = sentence[:60].lower().strip()
        # Clean probe — remove extra spaces
        probe = re.sub(r'\s+', ' ', probe)
        if probe in ext_lower:
            found += 1

    preservation = found / sample_size * 100

    log(f"  Sentences found     : {found}/{sample_size}")
    log(f"  Preservation rate   : {preservation:.1f}%")
    log(f"  Target              : > 80%")

    return preservation


# ═════════════════════════════════════════════
# METRIC 5 — BAST & KORZEN 2017 CRITERIA
# ═════════════════════════════════════════════

def bast_korzen_metrics(reference_text, extracted_text):
    """
    Direct implementation of metrics from:
    Bast & Korzen, JCDL 2017
    "A Benchmark and Evaluation for Text Extraction from PDF"

    Word-level criteria (order-independent parts):
    W+  : spurious words (in extraction, not reference)
    W-  : missing words  (in reference, not extraction)
    W~  : garbled/misspelled words

    Note: NL+/NL- are not meaningful here because
    Adobe preserves line breaks and PyMuPDF collapses them.
    This is expected behavior, not an error.
    """
    ref_words_set = set(reference_text.lower().split())
    ext_words_set = set(extracted_text.lower().split())

    w_plus  = len(ext_words_set - ref_words_set)
    w_minus = len(ref_words_set - ext_words_set)

    # W~ computation on sample
    all_ref     = reference_text.lower().split()
    all_ext     = extracted_text.lower().split()
    sample_size = min(50000, len(all_ref), len(all_ext))
    ref_sample  = all_ref[:sample_size]
    ext_sample  = all_ext[:sample_size]

    matcher = difflib.SequenceMatcher(None, ref_sample, ext_sample)
    w_tilde = sum(1 for tag, *_ in matcher.get_opcodes()
                  if tag == 'replace')

    total_ref_words = len(all_ref)

    log(f"\n{'─'*55}")
    log("METRIC 5: Bast & Korzen 2017 Word-Level Criteria")
    log(f"{'─'*55}")
    log(f"  W+  (spurious words)  : {w_plus:,} "
        f"({w_plus/total_ref_words*100:.2f}%)")
    log(f"  W-  (missing words)   : {w_minus:,} "
        f"({w_minus/total_ref_words*100:.2f}%)")
    log(f"  W~  (garbled words)   : {w_tilde:,}")
    log(f"  Total reference words : {total_ref_words:,}")
    log(f"\n  Best benchmark (Icecite, JCDL 2017):")
    log(f"    W+: 0.3% | W-: 0.1% | W~: 0.6%")
    log(f"\n  Note: NL+/NL- intentionally skipped.")
    log(f"    Adobe preserves per-line breaks.")
    log(f"    PyMuPDF collapses to paragraphs.")
    log(f"    Both behaviors are CORRECT for their use case.")

    return {
        'W+': w_plus,
        'W-': w_minus,
        'W~': w_tilde
    }


# ═════════════════════════════════════════════
# METRIC 6 — COVERAGE & COMPLETENESS (F1)
# ═════════════════════════════════════════════

def coverage_metrics(reference_text, extracted_text):
    """
    Measures what fraction of reference content
    vocabulary is present in extraction.

    Removes stopwords for meaningful comparison.
    Content words (nouns, verbs, terms) matter.

    Equivalent to token-level F1 in QA evaluation.
    """
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were',
        'in', 'on', 'at', 'to', 'for', 'of', 'and',
        'or', 'but', 'it', 'its', 'be', 'been', 'has',
        'have', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'that', 'this', 'these',
        'those', 'with', 'from', 'by', 'as', 'not',
        'no', 'if', 'so', 'than', 'also', 'more',
        'their', 'they', 'them', 'we', 'our', 'which',
        'who', 'what', 'when', 'where', 'how', 'all'
    }

    ref_words = set(reference_text.lower().split())
    ext_words = set(extracted_text.lower().split())

    ref_words = ref_words - stopwords
    ext_words = ext_words - stopwords

    intersection = ref_words & ext_words
    precision    = len(intersection) / len(ext_words) if ext_words else 0
    recall       = len(intersection) / len(ref_words)  if ref_words else 0
    f1           = (2 * precision * recall / (precision + recall)
                    if precision + recall > 0 else 0)

    log(f"\n{'─'*55}")
    log("METRIC 6: Coverage & Completeness (Content Words)")
    log(f"{'─'*55}")
    log(f"  Unique ref words    : {len(ref_words):,}")
    log(f"  Unique ext words    : {len(ext_words):,}")
    log(f"  Common words        : {len(intersection):,}")
    log(f"  Precision           : {precision*100:.2f}%")
    log(f"  Recall              : {recall*100:.2f}%")
    log(f"  F1 Score            : {f1*100:.2f}%")
    log(f"  Target F1           : > 95%")
    return precision, recall, f1


# ═════════════════════════════════════════════
# METRIC 7 — PDFPLUMBER CROSS-VALIDATION
# ═════════════════════════════════════════════

def cross_validate_with_pdfplumber(pdf_path, pymupdf_text):
    """
    Cross-validates PyMuPDF extraction against
    pdfplumber — a completely independent extractor.

    High agreement between two independent tools
    is strong evidence both are correct.

    Industry practice: multi-tool consensus validation.
    """
    import pdfplumber

    log(f"\n{'─'*55}")
    log("METRIC 7: pdfplumber Cross-Validation")
    log(f"{'─'*55}")
    log("  Extracting with pdfplumber (independent tool)...")

    plumber_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                plumber_text += text + "\n"

    # Normalize both before comparing
    pymupdf_norm  = normalize_text(pymupdf_text)
    plumber_norm  = normalize_text(plumber_text)

    pymupdf_words = set(pymupdf_norm.split())
    plumber_words = set(plumber_norm.split())

    # Remove very short tokens
    pymupdf_words = {w for w in pymupdf_words if len(w) > 2}
    plumber_words = {w for w in plumber_words if len(w) > 2}

    union        = pymupdf_words | plumber_words
    intersection = pymupdf_words & plumber_words
    agreement    = len(intersection) / len(union) if union else 0

    log(f"  PyMuPDF unique words   : {len(pymupdf_words):,}")
    log(f"  pdfplumber unique words: {len(plumber_words):,}")
    log(f"  Common words           : {len(intersection):,}")
    log(f"  Tool agreement (Jaccard): {agreement*100:.2f}%")
    log(f"  Target agreement       : > 90%")
    return agreement


# ═════════════════════════════════════════════
# BONUS — GARBLED CHARACTER CHECK
# ═════════════════════════════════════════════

def garbled_word_check(extracted_text):
    """
    Counts words containing non-ASCII characters.
    These indicate garbled/corrupted extraction.

    For digital-native PDFs, this should be near zero.
    Small non-zero rate is from legitimate special chars:
    - Citation marks (â€œ)
    - Em-dashes (—) and en-dashes (–)
    - Greek letters
    - Smart quotes
    """
    words         = extracted_text.split()
    total_words   = len(words)
    garbled_words = sum(
        1 for word in words
        if re.search(r'[^\x00-\x7F]', word)
    )
    w_tilde_rate = (garbled_words / total_words * 100
                    if total_words > 0 else 0)

    log(f"\n{'─'*55}")
    log("BONUS: Garbled Character Check (W~ proxy)")
    log(f"{'─'*55}")
    log(f"  Total words         : {total_words:,}")
    log(f"  Garbled words       : {garbled_words:,}")
    log(f"  W~ rate             : {w_tilde_rate:.2f}%")
    log(f"  Paper benchmark     : 0.6% (Icecite, JCDL 2017)")
    log(f"  Your system         : {w_tilde_rate:.2f}%")
    log(f"  Note: Small rate expected from special chars")
    log(f"        (em-dashes, citations, smart quotes)")
    return w_tilde_rate


# ═════════════════════════════════════════════
# BONUS — PAGE COVERAGE STATISTICS
# ═════════════════════════════════════════════

def page_coverage_stats():
    """
    Analyzes how many pages were extracted
    and classifies each page's content.
    """
    import fitz

    doc        = fitz.open(PDF_PATH)
    total      = doc.page_count
    empty      = []
    short      = []
    normal     = []

    for i in range(total):
        text = doc[i].get_text().strip()
        if len(text) == 0:
            empty.append(i + 1)
        elif len(text) < 150:
            short.append(i + 1)
        else:
            normal.append(i + 1)

    doc.close()

    log(f"\n{'─'*55}")
    log("BONUS: Page Coverage Statistics")
    log(f"{'─'*55}")
    log(f"  Total pages         : {total}")
    log(f"  Pages with text     : {len(normal)}")
    log(f"  Pages with short txt: {len(short)}")
    log(f"  Pages with NO text  : {len(empty)}")
    log(f"  Coverage rate       : "
        f"{(len(normal)+len(short))/total*100:.1f}%")
    log(f"\n  Note: Empty pages are expected —")
    log(f"    figures, diagrams, full-page images")
    log(f"    have no selectable text in PDF")

    if empty:
        log(f"\n  Empty page numbers  : {empty[:20]}"
            + (" ..." if len(empty) > 20 else ""))

    return total, len(normal), len(short), len(empty)


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def main():
    log("=" * 60)
    log("  PDF EXTRACTION QUALITY EVALUATION")
    log("  Methodology: Bast & Korzen, JCDL 2017")
    log("=" * 60)

    # ── Step 1: Extract full text ───────────────────────────
    log("\n[STEP 1] Extracting text with PyMuPDF")
    log("─" * 55)
    extracted = extract_full_pdf()

    # ── Step 2: Page coverage stats ─────────────────────────
    log("\n[STEP 2] Page Coverage Analysis")
    log("─" * 55)
    total, normal, short_p, empty = page_coverage_stats()

    # ── Step 3: Garbled word check ──────────────────────────
    log("\n[STEP 3] Garbled Character Check")
    log("─" * 55)
    garbled_rate = garbled_word_check(extracted)

    # ── Step 4: pdfplumber cross-validation ─────────────────
    log("\n[STEP 4] pdfplumber Cross-Validation")
    log("─" * 55)
    agreement = None
    try:
        import pdfplumber
        if os.path.exists(PDF_PATH):
            agreement = cross_validate_with_pdfplumber(
                PDF_PATH, extracted
            )
    except ImportError:
        log("  ⚠ pdfplumber not installed. Skipping.")
        log("    Install: pip install pdfplumber")

    # ── Step 5: Reference-based metrics ─────────────────────
    log("\n[STEP 5] Reference-Based Metrics")
    log("─" * 55)

    if not os.path.exists(REF_PATH):
        log(f"\n  ⚠ Reference file not found: {REF_PATH}")
        log(f"  To enable reference metrics, do this:")
        log(f"  1. Open data/book.pdf in Adobe Acrobat Reader")
        log(f"  2. Press CTRL+A to select all text")
        log(f"  3. Press CTRL+C to copy")
        log(f"  4. Open a text editor")
        log(f"  5. Paste and save as:")
        log(f"     {REF_PATH}")
        log(f"  6. Re-run this script")
        log(f"\n  Running available metrics without reference...")

        # ── Summary without reference ─────────────────────
        log("\n" + "=" * 60)
        log("  PARTIAL REPORT (no reference file)")
        log("=" * 60)
        log(f"  Pages extracted     : {normal + short_p}/{total}")
        log(f"  Coverage rate       : "
            f"{(normal+short_p)/total*100:.1f}%")
        log(f"  Empty pages         : {empty} (expected — figures)")
        log(f"  Garbled word rate   : {garbled_rate:.2f}%")
        if agreement:
            log(f"  Tool agreement      : {agreement*100:.2f}%")
        log(f"\n  Add reference file to get full metrics.")
        save_report()
        return

    with open(REF_PATH, 'r', encoding='utf-8') as f:
        reference = f.read()

    log(f"\n  ✓ Reference loaded: {REF_PATH}")
    log(f"    Reference chars : {len(reference):,}")
    log(f"    Reference words : {len(reference.split()):,}")

    log("\n" + "=" * 60)
    log("  RUNNING ALL METRICS")
    log("=" * 60)

    # Run all metrics
    p1, r1, f1_vocab = word_vocabulary_overlap(reference, extracted)
    ngram_f1         = ngram_overlap(reference, extracted, n=3)
    term_pres        = key_phrase_verification(reference, extracted)
    sent_pres        = page_sample_verification(reference, extracted)
    bk               = bast_korzen_metrics(reference, extracted)
    p2, r2, f1_cov   = coverage_metrics(reference, extracted)

    # ── Final Summary ───────────────────────────────────────
    log("\n" + "=" * 60)
    log("  FINAL EXTRACTION QUALITY REPORT")
    log("=" * 60)
    log(f"\n  PRIMARY METRICS (order-independent):")
    log(f"  {'─'*45}")
    log(f"  Vocabulary F1        : {f1_vocab*100:.2f}%"
        f"   (target > 95%)")
    log(f"  Coverage F1          : {f1_cov*100:.2f}%"
        f"   (target > 95%)")
    log(f"  3-gram phrase F1     : {ngram_f1*100:.2f}%"
        f"   (target > 80%)")
    log(f"  Key term presence    : {term_pres:.1f}%"
        f"     (target 100%)")
    log(f"  Sentence sample      : {sent_pres:.1f}%"
        f"     (target > 80%)")

    log(f"\n  BAST & KORZEN 2017 WORD-LEVEL CRITERIA:")
    log(f"  {'─'*45}")
    log(f"  W+ (spurious words)  : {bk['W+']:,}"
        f"  (target < 0.3%)")
    log(f"  W- (missing words)   : {bk['W-']:,}"
        f"   (target < 0.2%)")
    log(f"  W~ (garbled words)   : {bk['W~']:,}"
        f"     (paper best: 0.6%)")

    log(f"\n  STRUCTURAL METRICS:")
    log(f"  {'─'*45}")
    log(f"  Pages extracted      : {normal+short_p}/{total}"
        f"  ({(normal+short_p)/total*100:.1f}%)")
    log(f"  Empty pages          : {empty}"
        f"          (figures, diagrams)")
    log(f"  Garbled word rate    : {garbled_rate:.2f}%"
        f"   (paper best: 0.6%)")
    if agreement is not None:
        log(f"  Tool agreement       : {agreement*100:.2f}%"
            f"   (target > 90%)")

    # ── Verdict ─────────────────────────────────────────────
    passed = sum([
        f1_vocab  > 0.95,
        ngram_f1  > 0.80,
        term_pres > 95.0,
        sent_pres > 80.0,
        f1_cov    > 0.95,
    ])

    log(f"\n  {'─'*45}")
    log(f"  Metrics passed       : {passed}/5")

    if passed >= 4:
        log("  VERDICT: ✓ Extraction quality is GOOD")
    elif passed >= 3:
        log("  VERDICT: ~ Extraction quality is ACCEPTABLE")
    else:
        log("  VERDICT: ✗ Extraction quality needs improvement")

    log(f"\n  KEY INSIGHT:")
    log(f"  CER/WER are NOT reported because they measure")
    log(f"  sequential character alignment — meaningless")
    log(f"  when comparing two digital PDF extractors that")
    log(f"  format line breaks differently. Vocabulary F1")
    log(f"  and phrase overlap are the correct metrics.")

    log("\n" + "=" * 60)
    log("  REFERENCES")
    log("=" * 60)
    log("  [1] Bast, H., & Korzen, C. (2017).")
    log("      A benchmark and evaluation for text extraction")
    log("      from PDF. Proc. 17th ACM/IEEE JCDL, Toronto.")
    log("")
    log("  [2] Rajpurkar, P. et al. (2016).")
    log("      SQuAD: 100,000+ questions for machine")
    log("      comprehension of text. EMNLP 2016.")
    log("      (Token F1 metric used in QA evaluation)")
    log("")
    log("  [3] Lin, C. (2004).")
    log("      ROUGE: A package for automatic evaluation")
    log("      of summaries. ACL Workshop 2004.")
    log("      (N-gram overlap methodology)")
    log("=" * 60)

    save_report()


if __name__ == "__main__":
    main()