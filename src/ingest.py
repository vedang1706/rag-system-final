# src/ingest.py

import fitz
import json
import re
import os
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# TOC entry titles to skip entirely
SKIP_TITLES = {
    "blank page",
    "contents",
    "table of contents",
}

# TOC entry titles that are valid but low-priority
# (keep them but mark as supplementary)
SUPPLEMENTARY_TITLES = {
    "key terms",
    "summary",
    "review questions",
    "critical thinking questions",
    "personal application questions",
    "references",
    "index",
    "preface",
    "introduction",
}

# Pages before this are front matter — skip entirely
CONTENT_START_PAGE = 19


# ─────────────────────────────────────────────
# FUNCTION 1 — Load PDF
# ─────────────────────────────────────────────

def load_pdf(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF not found at: {path}\n"
            f"Put book.pdf inside the data/ folder."
        )
    doc = fitz.open(path)
    print(f"✓ PDF loaded: {doc.page_count} pages found")
    return doc


# ─────────────────────────────────────────────
# FUNCTION 2 — Extract and Clean TOC
# ─────────────────────────────────────────────

def extract_toc(doc):
    """
    Extracts TOC and removes junk entries.
    Returns cleaned toc list and page offset (always 0 for this book).
    """
    raw_toc = doc.get_toc()

    if not raw_toc:
        raise ValueError("No TOC found in PDF.")

    print(f"✓ Raw TOC entries: {len(raw_toc)}")

    # Clean and filter TOC
    toc = []
    for entry in raw_toc:
        level, title, page = entry
        title = title.strip()

        # Skip blank page and junk entries
        if title.lower() in SKIP_TITLES:
            continue

        # Skip entries with no title
        if not title:
            continue

        toc.append({
            "level": level,
            "title": title,
            "page":  page,
            "is_supplementary": title.lower() in SUPPLEMENTARY_TITLES
        })

    print(f"✓ Cleaned TOC entries: {len(toc)}")

    # Show summary
    chapters  = [e for e in toc if e["level"] == 1]
    sections  = [e for e in toc if e["level"] == 2]
    print(f"  Chapters : {len(chapters)}")
    print(f"  Sections : {len(sections)}")

    return toc


# ─────────────────────────────────────────────
# FUNCTION 3 — Build Section Map
# ─────────────────────────────────────────────

def build_section_map(toc: list, total_pages: int) -> list:
    """
    Builds section map from cleaned TOC.

    Key fix: end_page of each section = start_page of next section - 1
    Last section ends at total_pages.

    Also skips front matter (pages before CONTENT_START_PAGE).
    """
    section_map = []
    current_chapter = ""
    current_chapter_level1_title = ""

    for i, entry in enumerate(toc):

        # Track current chapter
        if entry["level"] == 1:
            current_chapter = entry["title"]
            current_chapter_level1_title = entry["title"]

        # Calculate end page
        if i + 1 < len(toc):
            end_page = toc[i + 1]["page"] - 1
        else:
            end_page = total_pages

        # Skip front matter
        if entry["page"] < CONTENT_START_PAGE:
            continue

        # Ensure end_page >= start_page
        if end_page < entry["page"]:
            end_page = entry["page"]

        # Build section path
        section_path = build_section_path(
            current_chapter_level1_title,
            entry["title"]
        )

        section_map.append({
            "title":            entry["title"],
            "level":            entry["level"],
            "chapter":          current_chapter,
            "section_path":     section_path,
            "start_page":       entry["page"],
            "end_page":         end_page,
            "is_supplementary": entry["is_supplementary"]
        })

    print(f"✓ Section map built: {len(section_map)} sections")
    print(f"  Content pages: {CONTENT_START_PAGE} → {total_pages}")

    return section_map


# ─────────────────────────────────────────────
# HELPER — Build Section Path
# ─────────────────────────────────────────────

def build_section_path(chapter: str, section: str) -> str:
    """
    Converts chapter + section into url-style path.

    "Chapter 2 Psychological Research" + "2.1 Why Is Research Important?"
    → "psychological_research/why_is_research_important"
    """
    def slugify(text: str) -> str:
        text = text.lower()
        # Remove "chapter N" prefix
        text = re.sub(r'^chapter\s+\d+\s*', '', text)
        # Remove "N.N" section number prefix
        text = re.sub(r'^\d+\.\d+\s+', '', text)
        # Remove special characters
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text.strip())
        return text

    chapter_slug = slugify(chapter)
    section_slug = slugify(section)

    if not chapter_slug or chapter_slug == section_slug:
        return section_slug

    return f"{chapter_slug}/{section_slug}"


# ─────────────────────────────────────────────
# FUNCTION 4 — Extract Text For Pages
# ─────────────────────────────────────────────

def extract_text_for_pages(doc, start_page: int, end_page: int) -> list:
    """
    Extracts text from page range (1-indexed).
    Returns list of (page_number, text) tuples.
    """
    pages_text = []

    for page_num in range(start_page - 1, min(end_page, doc.page_count)):
        page   = doc[page_num]
        text   = page.get_text()

        if not text.strip():
            continue

        text = clean_text(text)

        if len(text.strip()) > 50:  # skip near-empty pages
            pages_text.append((page_num + 1, text))

    return pages_text


# ─────────────────────────────────────────────
# HELPER — Clean Text
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Cleans PDF extracted text.
    """
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Skip standalone page numbers
        if re.match(r'^\d+$', stripped):
            continue

        # Skip "Access for free at openstax.org" footer
        if 'access for free at openstax' in stripped.lower():
            continue

        # Skip very short artifact lines
        if len(stripped) < 3:
            continue

        cleaned.append(stripped)

    text = ' '.join(cleaned)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# FUNCTION 5 — Chunk Section
# ─────────────────────────────────────────────

def chunk_section(pages_text: list, section_meta: dict,
                  max_words: int = 400,
                  overlap_words: int = 50) -> list:
    """
    Splits a section into chunks.

    Short sections (<=400 words)  → 1 chunk
    Long sections  (>400 words)   → paragraph splits with overlap

    Every chunk carries full section metadata.
    """
    if not pages_text:
        return []

    # Build full text with page tracking
    full_text        = ""
    page_boundaries  = []
    word_count       = 0

    for page_num, text in pages_text:
        page_boundaries.append((word_count, page_num))
        full_text  += " " + text
        word_count += len(text.split())

    full_text = full_text.strip()
    words     = full_text.split()

    if len(words) == 0:
        return []

    # Short section — one chunk
    if len(words) <= max_words:
        return [make_chunk(
            text         = full_text,
            section_meta = section_meta,
            chunk_index  = 0,
            pages        = [p[1] for p in page_boundaries]
        )]

    # Long section — split with overlap
    chunks     = []
    paragraphs = split_into_paragraphs(full_text)

    current_words = []
    current_pages = set()
    chunk_index   = 0
    word_pos      = 0

    for para in paragraphs:
        para_words = para.split()
        if not para_words:
            continue

        para_page = get_page_for_position(word_pos, page_boundaries)
        current_pages.add(para_page)
        current_words.extend(para_words)
        word_pos += len(para_words)

        if len(current_words) >= max_words:
            chunks.append(make_chunk(
                text         = ' '.join(current_words),
                section_meta = section_meta,
                chunk_index  = chunk_index,
                pages        = sorted(current_pages)
            ))
            chunk_index  += 1
            # Keep overlap
            current_words = current_words[-overlap_words:]
            current_pages = {para_page}

    # Final remaining chunk
    if len(current_words) > 20:  # skip tiny leftover chunks
        chunks.append(make_chunk(
            text         = ' '.join(current_words),
            section_meta = section_meta,
            chunk_index  = chunk_index,
            pages        = sorted(current_pages)
        ))

    return chunks


# ─────────────────────────────────────────────
# HELPER — Split Into Paragraphs
# ─────────────────────────────────────────────

def split_into_paragraphs(text: str) -> list:
    """
    Splits text at paragraph boundaries.
    Falls back to sentence groups if no paragraphs found.
    """
    # Try double-space split (paragraph marker)
    paragraphs = [p.strip() for p in text.split('  ') if p.strip()]

    if len(paragraphs) >= 3:
        return paragraphs

    # Fallback: group every 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    groups    = []
    current   = []

    for sent in sentences:
        current.append(sent)
        if len(current) >= 3:
            groups.append(' '.join(current))
            current = []

    if current:
        groups.append(' '.join(current))

    return groups if groups else [text]


# ─────────────────────────────────────────────
# HELPER — Make Chunk Dict
# ─────────────────────────────────────────────

def make_chunk(text: str, section_meta: dict,
               chunk_index: int, pages: list) -> dict:

    section_slug = section_meta["section_path"].replace("/", "_")
    chunk_id     = f"{section_slug}_chunk_{chunk_index}"

    return {
        "chunk_id":         chunk_id,
        "text":             text,
        "section":          section_meta["title"],
        "section_path":     section_meta["section_path"],
        "chapter":          section_meta["chapter"],
        "page_start":       min(pages),
        "page_end":         max(pages),
        "pages":            pages,
        "is_supplementary": section_meta.get("is_supplementary", False)
    }


# ─────────────────────────────────────────────
# HELPER — Get Page For Word Position
# ─────────────────────────────────────────────

def get_page_for_position(word_pos: int,
                           page_boundaries: list) -> int:
    current_page = page_boundaries[0][1]
    for boundary_pos, page_num in page_boundaries:
        if word_pos >= boundary_pos:
            current_page = page_num
        else:
            break
    return current_page


# ─────────────────────────────────────────────
# FUNCTION 6 — Build All Chunks
# ─────────────────────────────────────────────

def build_chunks(doc, section_map: list) -> list:
    all_chunks = []

    print(f"\nChunking {len(section_map)} sections...")

    for section in tqdm(section_map):
        if section["start_page"] > section["end_page"]:
            continue
        if section["start_page"] > doc.page_count:
            continue

        pages_text = extract_text_for_pages(
            doc,
            section["start_page"],
            section["end_page"]
        )

        if not pages_text:
            continue

        chunks = chunk_section(pages_text, section)
        all_chunks.extend(chunks)

    # Stats breakdown
    main_chunks = [c for c in all_chunks if not c["is_supplementary"]]
    supp_chunks = [c for c in all_chunks if c["is_supplementary"]]

    print(f"✓ Total chunks        : {len(all_chunks)}")
    print(f"  Main content chunks : {len(main_chunks)}")
    print(f"  Supplementary chunks: {len(supp_chunks)}")

    return all_chunks


# ─────────────────────────────────────────────
# FUNCTION 7 — Save Chunks
# ─────────────────────────────────────────────

def save_chunks(chunks: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    total_chars = sum(len(c['text']) for c in chunks)

    print(f"✓ Chunks saved to: {path}")
    print(f"  Total chunks    : {len(chunks)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Avg chunk size  : {total_chars // len(chunks)} chars")



# ─────────────────────────────────────────────
# FUNCTION 8 — Verify Chunks
# ─────────────────────────────────────────────

def verify_chunks(chunks: list, num_samples: int = 5):
    """
    Prints sample chunks for manual verification against PDF.
    """
    import random

    # Sample only main content chunks
    main = [c for c in chunks if not c["is_supplementary"]]
    samples = random.sample(main, min(num_samples, len(main)))

    print(f"\n{'='*60}")
    print("CHUNK VERIFICATION — Open PDF and check these pages")
    print(f"{'='*60}")

    for i, chunk in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  chunk_id    : {chunk['chunk_id']}")
        print(f"  section     : {chunk['section']}")
        print(f"  chapter     : {chunk['chapter']}")
        print(f"  pages       : {chunk['page_start']} → {chunk['page_end']}")
        print(f"  word count  : {len(chunk['text'].split())}")
        print(f"  text preview: {chunk['text'][:200]}...")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    PDF_PATH    = "data/book.pdf"
    CHUNKS_PATH = "cache/chunks.json"

    print("=" * 60)
    print("STEP 1: PDF INGESTION AND CHUNKING")
    print("=" * 60)

    doc = load_pdf(PDF_PATH)

    toc = extract_toc(doc)

    print("\nFirst 15 cleaned TOC entries:")
    for entry in toc[:15]:
        indent = "  " * (entry["level"] - 1)
        marker = "[supp]" if entry["is_supplementary"] else ""
        print(f"  {indent}[p.{entry['page']}] {entry['title']} {marker}")

    section_map = build_section_map(toc, doc.page_count)

    chunks = build_chunks(doc, section_map)

    verify_chunks(chunks)

    save_chunks(chunks, CHUNKS_PATH)

    print("\n" + "=" * 60)
    print("✓ INGESTION COMPLETE — No blank page chunks")
    print(f"  Run next: python src/embed_index.py")
    print("=" * 60)


if __name__ == "__main__":
    main()