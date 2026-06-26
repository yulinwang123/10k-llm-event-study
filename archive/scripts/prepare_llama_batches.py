"""
prepare_llama_batches.py
────────────────────────
Reads master_panel.parquet + local MD&A texts and writes JSONL batch files
for Llama inference.

Usage:
    python prepare_llama_batches.py \
        --data-root /scratch/midway3/yulinwang/10k_data/10k-project \
        --shard-size 50

Output:
    $data-root/llm_batches/batch_000.jsonl
    $data-root/llm_batches/batch_001.jsonl
    ...
"""

import argparse
import json
import os
import pathlib

import pandas as pd
from tqdm import tqdm

SYSTEM_PROMPT = """\
You are a financial economist analyzing a 10-K Management Discussion and Analysis (MD&A) section.
Your task: score the text on FOUR dimensions that predict stock market reactions around earnings announcements.
Respond with ONLY a JSON object — no explanation, no markdown, no extra text.

Output format (integers 0–10):
{"management_optimism": <int>, "guidance_specificity": <int>, "uncertainty_hedging": <int>, "risk_framing": <int>}

Dimension definitions:
- management_optimism (0–10):
    Overall management sentiment about the company's future prospects.
    0 = strongly negative/pessimistic language throughout
    5 = balanced / neutral tone
    10 = strongly positive / confident about growth and performance
    Focus on FORWARD-LOOKING statements, not just backward results.

- guidance_specificity (0–10):
    How concrete and specific is management's forward guidance?
    0 = only vague qualitative statements ("we face challenges"), no numbers, no targets
    5 = some specific targets or ranges mentioned
    10 = detailed numerical guidance (revenue targets, margin expectations, specific timelines)
    High specificity = investors can price-in expectations accurately.

- uncertainty_hedging (0–10):
    Density of hedging/uncertain language.
    0 = highly confident, few qualifiers, management sounds certain
    5 = moderate hedging, typical disclaimer boilerplate
    10 = extreme uncertainty, constant "may", "could", "if", "subject to change", soft commitments
    Note: this is DIFFERENT from risk disclosure — it measures linguistic confidence.

- risk_framing (0–10):
    How does management frame risks relative to opportunities?
    0 = risks framed as manageable, downplayed, or offset by opportunities
    5 = balanced presentation
    10 = risks are prominent, severe, or dwarf the discussion of opportunities
    Key: same risk can score 2 (framed positively) or 8 (framed as threatening).
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/scratch/midway3/yulinwang/10k_data/10k-project")
    p.add_argument("--shard-size", type=int, default=50,
                   help="Number of filings per batch file")
    p.add_argument("--max-chars", type=int, default=6000,
                   help="MD&A characters to use per filing")
    return p.parse_args()


def read_mda(data_root, s3_key):
    if not s3_key:
        return ""
    rel = s3_key.replace("10k-project/", "", 1)
    local_path = os.path.join(data_root, rel)
    try:
        with open(local_path, "r", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def build_messages(mda_text, max_chars=6000):
    excerpt = mda_text[:max_chars].strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"MD&A excerpt (first ~6000 characters):\n\n{excerpt}"}
    ]


def main():
    args = parse_args()
    root = args.data_root

    panel_path = os.path.join(root, "processed", "master_panel.parquet")
    print(f"Loading master panel from {panel_path}...")
    panel = pd.read_parquet(panel_path)
    panel = panel.sort_values(["gvkey", "fyear"]).reset_index(drop=True)
    print(f"  {len(panel)} firm-years")

    out_dir = pathlib.Path(root) / "llm_batches"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    print("Building prompts...")
    for i, row in tqdm(panel.iterrows(), total=len(panel)):
        mda_text = read_mda(root, row.get("s3_key", ""))
        if not mda_text.strip():
            continue
        records.append({
            "id":       i,
            "gvkey":    row["gvkey"],
            "fyear":    int(row["fyear"]),
            "ticker":   row["ticker"],
            "messages": build_messages(mda_text, args.max_chars),
        })

    print(f"  {len(records)} records with non-empty MD&A")

    # Split into shards
    n_shards = (len(records) + args.shard_size - 1) // args.shard_size
    for s in range(n_shards):
        shard = records[s * args.shard_size : (s + 1) * args.shard_size]
        out_path = out_dir / f"batch_{s:03d}.jsonl"
        with open(out_path, "w") as f:
            for rec in shard:
                f.write(json.dumps(rec) + "\n")

    print(f"\n✓ Wrote {n_shards} batch files to {out_dir}")
    print(f"  Array range for sbatch: 0-{n_shards - 1}")
    print(f"  e.g.: sbatch --array=0-{n_shards - 1}%5 submit_llama.sh")


if __name__ == "__main__":
    main()
