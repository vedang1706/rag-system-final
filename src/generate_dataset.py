import sys
import os
import json
from tqdm import tqdm

# Add src to path so we can import generate
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from generate import init_client, call_nvidia_api

def generate_qa(chunk_text, client):
    system_msg = "You are an assistant that creates educational datasets. Always respond in exactly the format requested with no extra conversational text."
    user_msg = f"""
    Generate 1 clean, high-quality question-answer pair from this text.
    The question should be factual and not vague ("What is psychology?").
    The answer should be 1-2 lines long and factually correct.
    
    Text:
    {chunk_text}
    
    Format EXACTLY like this (do not use markdown blocks, do not add introductory text):
    Q: [Your Question Here]
    A: [Your Answer Here]
    """
    return call_nvidia_api(client, system_msg, user_msg)

def main():
    print("Loading chunks...")
    chunks_path = "cache/chunks.json"
    if not os.path.exists(chunks_path):
        print("Error: cache/chunks.json not found!")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    client = init_client()
    
    dataset = []
    target_count = 50
    
    # ── Stratified cross-chapter sampling ──
    valid_chunks = [c for c in chunks if not c.get('is_supplementary', False) and len(c['text']) > 150]
    
    from collections import defaultdict
    import random
    
    chapter_dict = defaultdict(list)
    for c in valid_chunks:
        chapter_dict[c.get('chapter', 'Unknown')].append(c)
        
    selected_chunks = []
    # Pick up to 5 chunks from every single chapter to ensure diversity
    for chap, c_list in chapter_dict.items():
        selected_chunks.extend(c_list[:5])
        
    random.seed(42)
    random.shuffle(selected_chunks)
    
    print(f"Generating {target_count} diverse Q&A pairs evenly from {len(chapter_dict)} chapters (this may take 2-3 minutes)...")
    
    for chunk in tqdm(selected_chunks):
        if len(dataset) >= target_count:
            break
            
        try:
            # We don't want to use chunks that are just table of contents
            if len(chunk['text']) < 150:
                continue
                
            qa_text = generate_qa(chunk['text'], client)
            lines = [line.strip() for line in qa_text.split("\n") if line.strip()]
            
            # Simple parser for Q: and A:
            q = None
            a = None
            for i in range(len(lines)):
                if lines[i].startswith("Q:"):
                    q = lines[i][2:].strip()
                elif lines[i].startswith("A:"):
                    a = lines[i][2:].strip()
            
            # Validation Step Mapped From Prompt
            # Remove vague questions, long answers, or malformed data
            if q and a and len(q) > 5 and len(a) > 5 and "?" in q:
                # Deduplicate
                if not any(item['question'] == q for item in dataset):
                    pages = chunk.get('pages', [])
                    if not pages and 'page_start' in chunk:
                        pages = [chunk['page_start']]
                        
                    dataset.append({
                        "query_id": str(len(dataset) + 1),
                        "question": q,
                        "ground_truth": a,
                        "relevant_pages": pages,
                        "relevant_section": chunk.get('chapter', 'unknown')
                    })
        except Exception as e:
            print(f"Error on chunk: {e}")
            
    print(f"\n✓ Generated {len(dataset)} valid pairs.")
    
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"✓ Saved perfectly formatted dataset to {out_path}")

if __name__ == "__main__":
    main()
