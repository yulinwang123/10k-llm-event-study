"""
midway3_finbert_embed.py
────────────────────────
Runs on Midway3 GPU node.

Track 2: FinBERT (ProsusAI/finbert) — sentence-level sentiment
Track 4: Sentence-BERT (all-mpnet-base-v2) — year-over-year semantic similarity

Features:
- Reads master panel + MD&A texts from LOCAL disk (pre-downloaded from S3)
- Checkpoint every CKPT_EVERY rows → saved to local disk
- Resume: skips already-completed rows on restart

Usage:
    python midway3_finbert_embed.py --data-root /scratch/midway3/yulinwang/10k_data
"""

import argparse
import json
import os
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

CKPT_EVERY = 200   # save checkpoint every N rows

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/scratch/midway3/yulinwang/10k_data/10k-project")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


# ── Local file helpers ─────────────────────────────────────────────────────────
def read_mda(data_root, s3_key):
    """Read MD&A text from local disk. s3_key is relative to bucket root."""
    if not s3_key:
        return ""
    # s3_key looks like "10k-project/filings/AAPL/2015/0001234.txt"
    # data_root is "/scratch/.../10k_data/10k-project"
    # strip the "10k-project/" prefix since data_root already includes it
    rel = s3_key.replace("10k-project/", "", 1)
    local_path = os.path.join(data_root, rel)
    try:
        with open(local_path, "r", errors="replace") as f:
            return f.read()
    except Exception:
        return ""

def write_parquet(path, df):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  ✓ Saved {len(df)} rows → {path}")

def load_checkpoint(path):
    """Load checkpoint rows from local JSONL. Returns list of dicts or []."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_checkpoint(path, rows):
    """Write all rows as JSONL to local disk."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))


# ── Text chunking ──────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_words=340, stride_words=40) -> list:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_words])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_words - stride_words
    return chunks if chunks else [text[:2000]]


# ── Track 2: FinBERT sentiment ─────────────────────────────────────────────────
@torch.no_grad()
def finbert_score(text, tokenizer, model, device, batch_size):
    if not text.strip():
        return {"fb_positive": None, "fb_negative": None,
                "fb_neutral": None, "fb_net": None, "fb_n_chunks": 0}

    label_map = {v: k for k, v in model.config.id2label.items()}
    pos_idx = label_map["positive"]
    neg_idx = label_map["negative"]
    neu_idx = label_map["neutral"]

    chunks = chunk_text(text)
    probs_all = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs_all.append(F.softmax(logits, dim=-1).cpu().numpy())

    probs = np.vstack(probs_all)
    return {
        "fb_positive":  float(probs[:, pos_idx].mean()),
        "fb_negative":  float(probs[:, neg_idx].mean()),
        "fb_neutral":   float(probs[:, neu_idx].mean()),
        "fb_net":       float((probs[:, pos_idx] - probs[:, neg_idx]).mean()),
        "fb_n_chunks":  len(chunks),
    }


# ── Track 4: Sentence-BERT embedding ──────────────────────────────────────────
def sbert_embedding(text, sbert_model, max_chars=50000):
    """
    Sentence-BERT handles long text by chunking internally.
    We chunk manually to stay within memory limits and mean-pool.
    """
    if not text.strip():
        return None
    chunks = chunk_text(text[:max_chars])
    vecs = sbert_model.encode(chunks, batch_size=64,
                               show_progress_bar=False,
                               convert_to_numpy=True)
    return vecs.mean(axis=0)   # (768,)

def cosine_sim(a, b):
    if a is None or b is None:
        return None
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return None
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    root   = args.data_root   # e.g. /scratch/midway3/yulinwang/10k_data/10k-project
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data root: {root}")

    # Checkpoint paths (local)
    fb_ckpt_path    = os.path.join(root, "checkpoints", "finbert_ckpt.jsonl")
    embed_ckpt_path = os.path.join(root, "checkpoints", "embed_ckpt.jsonl")

    # ── Load master panel ──────────────────────────────────────────────────────
    panel_path = os.path.join(root, "processed", "master_panel.parquet")
    print(f"Loading master panel from {panel_path}...")
    panel = pd.read_parquet(panel_path)
    panel = panel.sort_values(["gvkey", "fyear"]).reset_index(drop=True)
    print(f"  {len(panel)} firm-years, {panel['ticker'].nunique()} firms")

    # ── Load MD&A texts ────────────────────────────────────────────────────────
    print("Loading MD&A texts from local disk...")
    panel["mda_text"] = [
        read_mda(root, key)
        for key in tqdm(panel["s3_key"].fillna(""), desc="Reading local")
    ]

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading FinBERT (Track 2)...")
    fb_tok   = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fb_model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert").eval().to(device)
    print("  ✓ FinBERT ready")

    print("Loading Sentence-BERT (Track 4)...")
    sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                 device=device)
    print("  ✓ Sentence-BERT ready")

    # ══════════════════════════════════════════════════════════════════════════
    # Track 2: FinBERT sentiment (with checkpoint resume)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Track 2: FinBERT sentiment ──")
    fb_done  = load_checkpoint(fb_ckpt_path)
    done_ids = {(r["gvkey"], r["fyear"]) for r in fb_done}
    print(f"  Resuming: {len(fb_done)} already done, "
          f"{len(panel) - len(fb_done)} remaining")

    fb_rows = list(fb_done)
    new_since_ckpt = 0

    for _, row in tqdm(panel.iterrows(), total=len(panel), desc="FinBERT"):
        key = (row["gvkey"], row["fyear"])
        if key in done_ids:
            continue
        result = finbert_score(row["mda_text"], fb_tok, fb_model,
                               device, args.batch_size)
        fb_rows.append({"gvkey": row["gvkey"], "fyear": row["fyear"], **result})
        new_since_ckpt += 1

        if new_since_ckpt >= CKPT_EVERY:
            save_checkpoint(fb_ckpt_path, fb_rows)
            print(f"  [checkpoint] {len(fb_rows)} rows saved")
            new_since_ckpt = 0

    save_checkpoint(fb_ckpt_path, fb_rows)
    fb_df = pd.DataFrame(fb_rows)
    write_parquet(os.path.join(root, "processed", "finbert_scores.parquet"), fb_df)
    print(f"  ✓ Track 2 complete: {len(fb_df)} rows")

    # ══════════════════════════════════════════════════════════════════════════
    # Track 4: Sentence-BERT embeddings + YoY cosine similarity
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Track 4: Sentence-BERT embeddings ──")
    embed_done  = load_checkpoint(embed_ckpt_path)
    done_ids_e  = {(r["gvkey"], r["fyear"]) for r in embed_done}
    print(f"  Resuming: {len(embed_done)} already done, "
          f"{len(panel) - len(embed_done)} remaining")

    embed_dict = {}
    for r in embed_done:
        if r.get("embedding") is not None:
            embed_dict[(r["gvkey"], r["fyear"])] = np.array(r["embedding"])

    new_since_ckpt  = 0
    embed_rows_ckpt = list(embed_done)

    for _, row in tqdm(panel.iterrows(), total=len(panel), desc="SBERT"):
        key = (row["gvkey"], row["fyear"])
        if key not in done_ids_e:
            vec = sbert_embedding(row["mda_text"], sbert)
            embed_dict[key] = vec
            embed_rows_ckpt.append({
                "gvkey": row["gvkey"],
                "fyear": row["fyear"],
                "embedding": vec.tolist() if vec is not None else None,
            })
            new_since_ckpt += 1

            if new_since_ckpt >= CKPT_EVERY:
                save_checkpoint(embed_ckpt_path, embed_rows_ckpt)
                print(f"  [checkpoint] {len(embed_rows_ckpt)} embeddings saved")
                new_since_ckpt = 0

    save_checkpoint(embed_ckpt_path, embed_rows_ckpt)

    # Compute year-over-year cosine similarity
    print("  Computing YoY cosine similarity...")
    sim_rows = []
    for gvkey, grp in panel.groupby("gvkey"):
        grp = grp.sort_values("fyear")
        for i, (_, row) in enumerate(grp.iterrows()):
            fyear = row["fyear"]
            prev_fyear = grp.iloc[i-1]["fyear"] if i > 0 else None
            vec_curr = embed_dict.get((gvkey, fyear))
            vec_prev = embed_dict.get((gvkey, prev_fyear)) if prev_fyear else None
            sim = cosine_sim(vec_curr, vec_prev)
            sim_rows.append({
                "gvkey":         gvkey,
                "fyear":         fyear,
                "embed_cos_sim": sim,
                "embed_novelty": (1 - sim) if sim is not None else None,
            })

    sim_df = pd.DataFrame(sim_rows)
    write_parquet(os.path.join(root, "processed", "embed_similarity.parquet"), sim_df)
    print(f"  ✓ Track 4 complete: {len(sim_df)} rows")

    print(f"\n{'='*55}")
    print("All done.")
    print(f"  Track 2 → {root}/processed/finbert_scores.parquet")
    print(f"  Track 4 → {root}/processed/embed_similarity.parquet")


if __name__ == "__main__":
    main()
