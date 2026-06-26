"""
compute_text_features.py
─────────────────────────
Compute text features from earnings call transcripts.

Methods:
  1. LM (Loughran-McDonald) dictionary — finance-domain tone/sentiment
  2. Gunning Fog Index — readability / complexity
  3. Basic counts — word count, sentence length, vocabulary richness

Features computed separately for prepared_text and qa_text:
  {prefix}_word_count        total word count
  {prefix}_lm_tone           (positive - negative) / total  ← main narrative variable
  {prefix}_lm_negative       negative word ratio
  {prefix}_lm_positive       positive word ratio
  {prefix}_lm_uncertainty    uncertainty word ratio
  {prefix}_lm_litigious      litigious word ratio
  {prefix}_fog_index         Gunning Fog readability (higher = more complex)
  {prefix}_avg_sent_len      words per sentence
  {prefix}_unique_word_ratio vocabulary richness

Input:  data/transcripts_raw.parquet
Output: data/text_features.parquet
"""

import re
import subprocess
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE    = Path(__file__).parent.parent / "data"
CHUNK   = 5000   # process N transcripts at a time to manage memory

# ── Step 1: Load LM Master Dictionary ────────────────────────────────────────
# Updated 2026: LM dict now hosted on Google Drive (old ND URL returns 410)
# File ID: 1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF  (LoughranMcDonald_MasterDictionary_1993-2025.csv)
LM_CSV    = BASE / "LM_MasterDictionary.csv"
LM_GDRIVE = "1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"

if not LM_CSV.exists():
    print("Downloading LM Master Dictionary from Google Drive …")
    try:
        import gdown
    except ImportError:
        print("  gdown not found — installing …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
    try:
        gdown.download(id=LM_GDRIVE, output=str(LM_CSV), quiet=False)
        print("  Downloaded.")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Please download manually from https://sraf.nd.edu/loughranmcdonald-master-dictionary/")
        print(f"  and save to {LM_CSV}")
        raise

lm = pd.read_csv(LM_CSV)
lm.columns = [c.strip() for c in lm.columns]
print(f"LM dictionary: {len(lm):,} words, columns: {list(lm.columns[:8])}")

neg_words = set(lm[lm["Negative"] != 0]["Word"].str.upper())
pos_words = set(lm[lm["Positive"] != 0]["Word"].str.upper())
unc_words = set(lm[lm["Uncertainty"] != 0]["Word"].str.upper())
lit_words = set(lm[lm["Litigious"] != 0]["Word"].str.upper())

print(f"  Negative: {len(neg_words):,}  Positive: {len(pos_words):,}  "
      f"Uncertainty: {len(unc_words):,}  Litigious: {len(lit_words):,}")

# ── Step 2: Feature functions ─────────────────────────────────────────────────
def syllable_count(word: str) -> int:
    """Approximate syllable count (for Fog Index)."""
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 0
    count = len(re.findall(r'[aeiouy]+', word))
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)

def compute_features(text: str) -> dict:
    """Return dict of text features for one transcript section."""
    if not text or not isinstance(text, str) or len(text.strip()) < 20:
        return {}

    # Sentence split
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sents   = max(1, len(sentences))

    # Word tokens (alpha only)
    words_raw   = re.findall(r'\b[a-zA-Z]+\b', text)
    words_upper = [w.upper() for w in words_raw]
    n = len(words_upper)
    if n == 0:
        return {}

    # LM counts
    n_neg = sum(1 for w in words_upper if w in neg_words)
    n_pos = sum(1 for w in words_upper if w in pos_words)
    n_unc = sum(1 for w in words_upper if w in unc_words)
    n_lit = sum(1 for w in words_upper if w in lit_words)

    # Fog Index: 0.4 * (avg_sent_len + pct_complex_words)
    complex_count = sum(1 for w in words_raw if syllable_count(w) >= 3)
    avg_sent_len  = n / n_sents
    fog           = 0.4 * (avg_sent_len + 100 * complex_count / n)

    return {
        "word_count":        n,
        "lm_tone":           (n_pos - n_neg) / n,
        "lm_negative":       n_neg / n,
        "lm_positive":       n_pos / n,
        "lm_uncertainty":    n_unc / n,
        "lm_litigious":      n_lit / n,
        "fog_index":         round(fog, 4),
        "avg_sent_len":      round(avg_sent_len, 2),
        "unique_word_ratio": round(len(set(words_upper)) / n, 4),
    }

# ── Step 3: Load transcripts from chunk files and compute features ─────────────
# Full text lives in data/transcripts_tmp/chunk_*.parquet (not in transcripts_raw.parquet)
# We only process transcripts that made it into the event panel (saves ~90% time)
print("\nLoading event panel to identify needed transcriptids …")
panel = pd.read_parquet(BASE / "event_panel.parquet", columns=["transcriptid"])
need_ids = set(panel["transcriptid"].dropna().astype(int).tolist())
print(f"  Need text features for {len(need_ids):,} transcripts")

tmp_dir    = BASE / "transcripts_tmp"
chunk_files = sorted(tmp_dir.glob("chunk_*.parquet"))
print(f"  Reading {len(chunk_files)} chunk files …\n")

results = []
total_done = 0

for cf in chunk_files:
    chunk = pd.read_parquet(cf)  # columns: transcriptid, prepared_text, qa_text
    chunk["transcriptid"] = chunk["transcriptid"].astype(int)
    chunk = chunk[chunk["transcriptid"].isin(need_ids)]
    if chunk.empty:
        continue

    rows = []
    for _, row in chunk.iterrows():
        prep = compute_features(row.get("prepared_text", ""))
        qa   = compute_features(row.get("qa_text", ""))
        entry = {"transcriptid": int(row["transcriptid"])}
        for k, v in prep.items():
            entry[f"prep_{k}"] = v
        for k, v in qa.items():
            entry[f"qa_{k}"] = v
        rows.append(entry)

    results.append(pd.DataFrame(rows))
    total_done += len(rows)
    print(f"  {cf.name}: {len(rows):,} processed  (running total: {total_done:,}/{len(need_ids):,})")

features = pd.concat(results, ignore_index=True)

# Merge back gvkey + call_date from transcript_events
meta = pd.read_parquet(BASE / "transcript_events.parquet",
                       columns=["transcriptid", "gvkey", "call_date", "ciq_ticker"])
meta["transcriptid"] = meta["transcriptid"].astype(int)
features = features.merge(meta, on="transcriptid", how="left")

# ── Step 4: Summary stats ─────────────────────────────────────────────────────
print(f"\nFeature matrix: {features.shape}")
print("\nKey variable summary:")
cols = ["prep_word_count", "prep_lm_tone", "prep_lm_uncertainty",
        "prep_fog_index", "qa_word_count", "qa_lm_tone"]
print(features[cols].describe().round(4).to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "text_features.parquet"
features.to_parquet(out, index=False)
print(f"\nSaved → {out}  ({len(features):,} rows, {features.shape[1]} columns)")
