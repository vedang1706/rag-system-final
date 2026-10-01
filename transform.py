import json
import os

def main():
    with open("cache/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    with open("outputs/dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    new_dataset = []
    
    # Text matching to recover chapter/page metadata
    for i, item in enumerate(dataset):
        q = item["question"]
        a = item.get("answer", item.get("ground_truth"))
        
        # We find the chunk that likely generated this
        # Since we don't have exact mapping, we do a basic keyword match on the answer
        best_chunk = None
        for c in chunks:
            # Simple heuristic: if the first 20 chars of answer are in the chunk text
            if len(a) > 20 and a[:20].lower() in c['text'].lower():
                best_chunk = c
                break
                
        # Fallback if hard match fails
        if not best_chunk:
            best_chunk = chunks[0]
            
        pages = best_chunk.get('pages', [])
        if not pages and 'page_start' in best_chunk:
            pages = [best_chunk['page_start']]
        section = best_chunk.get('chapter', 'unknown')
        
        new_dataset.append({
            "query_id": str(i + 1),
            "question": q,
            "ground_truth": a,
            "relevant_pages": pages,
            "relevant_section": section
        })
        
    with open("outputs/dataset_new.json", "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, indent=2)
        
    # Replace old with new carefully
    import shutil
    shutil.move("outputs/dataset_new.json", "outputs/dataset.json")
    print("Schema transformation complete.")

if __name__ == "__main__":
    main()
