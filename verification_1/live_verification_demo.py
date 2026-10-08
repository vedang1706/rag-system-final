import sys
import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verification_1.verifier import verify_answer, ModernCEVerifier, get_verifier


def parse_context_file(filepath: Path) -> List[Dict[str, Any]]:
    content = filepath.read_text(encoding="utf-8")
    blocks = re.split(r"--- Chunk (\d+) ---", content)
    chunks = []
    for i in range(1, len(blocks), 2):
        cnum = int(blocks[i])
        cbody = blocks[i+1].strip()
        sec_m = re.search(r"Section:\s*(.+)", cbody)
        pages_m = re.search(r"Pages:\s*(.+)", cbody)
        raw_m = re.search(r"Raw Text:\s*\n(.*)", cbody, re.DOTALL)
        chunks.append({
            "chunk_number": cnum,
            "section": sec_m.group(1).strip() if sec_m else "Unknown",
            "pages": pages_m.group(1).strip() if pages_m else "?",
            "text": raw_m.group(1).strip() if raw_m else ""
        })
    return chunks


def run_live_demo():
    print("=" * 80)
    print("LIVE MODERNCE COMPLETE-ANSWER VERIFICATION DEMO")
    print("=" * 80)

    verifier = get_verifier()

    # Load submission and queries
    import pandas as pd
    df_sub = pd.read_csv(project_root / "outputs" / "submission.csv")
    with open(project_root / "data" / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    demo_qids = ["1", "41", "48"]

    for qid in demo_qids:
        idx = int(qid) - 1
        q_item = queries[idx]
        sub_row = df_sub.iloc[idx]

        q_text = q_item.get("question") or q_item.get("query")
        gen_ans = str(sub_row["answer"])
        ctx_file = project_root / "outputs" / "retrieved_contexts" / f"{qid}.txt"
        chunks = parse_context_file(ctx_file)

        print(f"\n--- [DEMO QUERY {qid}] ---")
        print(f"Query           : {q_text}")
        print(f"Generated Answer: {gen_ans[:140]}..." if len(gen_ans) > 140 else f"Generated Answer: {gen_ans}")
        print(f"Retrieved Chunks: {len(chunks)} chunks loaded from {ctx_file.name}")

        t0 = time.perf_counter()
        result = verifier.verify(answer=gen_ans, retrieved_chunks=chunks, query=q_text)
        elapsed = time.perf_counter() - t0

        print(f"Status          : {result.status}")
        print(f"Verdict         : {result.verdict}")
        print(f"Abstention Flag : {result.is_abstention}")
        print(f"Combination Used: {result.combination_used}")
        print(f"Selected Chunks : Chunks {result.selected_chunk_numbers}")
        print(f"Probabilities   : Entailment={result.entailment_prob:.4f}, Neutral={result.neutral_prob:.4f}, Contradiction={result.contradiction_prob:.4f}")
        print(f"Explanation     : {result.explanation}")
        print(f"Verification Time: {elapsed:.3f} s (NLI time: {result.total_nli_time_sec:.3f} s)")

    print("\n" + "=" * 80)
    print("LIVE DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_live_demo()
