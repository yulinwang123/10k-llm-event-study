"""
Build a small human-in-the-loop LLM demo batch.

This script does not call any API. It:
  1. Samples transcripts from the existing event panel.
  2. Loads prepared_text / qa_text from data/transcripts_tmp/chunk_*.parquet.
  3. Inserts each segment into the project prompt template.
  4. Writes an Excel file where the user can paste ChatGPT/Codex JSON outputs.

Outputs:
  data/llm_demo/demo_tasks.jsonl
  data/llm_demo/demo_prompts.md
  data/llm_demo/manual_scoring.xlsx
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
PROMPT_PATH = BASE / "skills" / "earnings-call-llm-inference" / "references" / "prompt_v1.md"


def extract_prompt_template(path: Path) -> str:
    text = path.read_text()
    match = re.search(r"```text\n(.*?)\n```", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not find ```text prompt block in {path}")
    return match.group(1)


def truncate_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return (
        text[:head]
        + "\n\n[... transcript segment truncated for demo ...]\n\n"
        + text[-tail:]
    )


def sample_transcripts(panel: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    panel = panel.dropna(subset=["transcriptid"]).copy()
    panel["transcriptid"] = panel["transcriptid"].astype(int)

    picks = []
    used = set()

    def add_rows(df, k):
        if df.empty or k <= 0:
            return
        sample = df.sample(n=min(k, len(df)), random_state=int(rng.integers(0, 1_000_000)))
        for _, row in sample.iterrows():
            tid = int(row["transcriptid"])
            if tid not in used:
                used.add(tid)
                picks.append(row)

    if "sue_win" in panel:
        add_rows(panel.nsmallest(max(n, 20), "sue_win"), max(2, n // 5))
        add_rows(panel.nlargest(max(n, 20), "sue_win"), max(2, n // 5))
    if "eps_vol" in panel:
        add_rows(panel.nlargest(max(n, 20), "eps_vol"), max(2, n // 5))
    if "numest" in panel:
        add_rows(panel.nsmallest(max(n, 20), "numest"), max(2, n // 5))

    remaining = panel[~panel["transcriptid"].isin(used)]
    add_rows(remaining, n - len(picks))

    out = pd.DataFrame(picks).head(n).copy()
    return out


def load_transcript_text(transcript_ids: set[int]) -> pd.DataFrame:
    rows = []
    for chunk_path in sorted((DATA / "transcripts_tmp").glob("chunk_*.parquet")):
        chunk = pd.read_parquet(chunk_path)
        chunk["transcriptid"] = chunk["transcriptid"].astype(int)
        chunk = chunk[chunk["transcriptid"].isin(transcript_ids)]
        if not chunk.empty:
            rows.append(chunk[["transcriptid", "prepared_text", "qa_text"]])
    if not rows:
        raise ValueError("No transcript text found for sampled transcript IDs.")
    return pd.concat(rows, ignore_index=True).drop_duplicates("transcriptid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12, help="Number of transcripts to sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=7000, help="Max chars per segment prompt.")
    parser.add_argument("--segments", default="prepared,qa", help="Comma-separated: prepared,qa")
    parser.add_argument("--out-dir", default=str(DATA / "llm_demo"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_cols = [
        "transcriptid", "gvkey", "permno", "ticker", "call_date", "fpedats",
        "sue_win", "car_m1_p1_win", "eps_vol", "numest", "io_pct"
    ]
    panel = pd.read_parquet(DATA / "event_panel.parquet")
    keep_cols = [c for c in panel_cols if c in panel.columns]
    panel = panel[keep_cols].copy()

    sampled = sample_transcripts(panel, args.n, args.seed)
    text_df = load_transcript_text(set(sampled["transcriptid"].astype(int)))
    sampled = sampled.merge(text_df, on="transcriptid", how="left")

    prompt_template = extract_prompt_template(PROMPT_PATH)
    wanted_segments = [s.strip() for s in args.segments.split(",") if s.strip()]

    tasks = []
    for _, row in sampled.iterrows():
        for segment in wanted_segments:
            text_col = "prepared_text" if segment == "prepared" else "qa_text"
            raw_text = row.get(text_col, "")
            if not isinstance(raw_text, str) or len(raw_text.strip()) < 200:
                continue
            segment_text = truncate_text(raw_text, args.max_chars)
            prompt = (
                prompt_template
                .replace("{segment_type}", segment)
                .replace("{text}", segment_text)
            )
            task = {
                "task_id": f"{int(row['transcriptid'])}_{segment}",
                "transcriptid": int(row["transcriptid"]),
                "segment": segment,
                "gvkey": row.get("gvkey", None),
                "ticker": row.get("ticker", None),
                "call_date": str(row.get("call_date", "")),
                "fpedats": str(row.get("fpedats", "")),
                "sue_win": row.get("sue_win", None),
                "eps_vol": row.get("eps_vol", None),
                "numest": row.get("numest", None),
                "io_pct": row.get("io_pct", None),
                "text_excerpt": segment_text[:1200],
                "prompt": prompt,
                "model_json": "",
                "human_check": "",
                "notes": "",
            }
            tasks.append(task)

    jsonl_path = out_dir / "demo_tasks.jsonl"
    with jsonl_path.open("w") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False, default=str) + "\n")

    md_path = out_dir / "demo_prompts.md"
    with md_path.open("w") as f:
        f.write("# Demo Prompts\n\n")
        f.write("Copy one prompt at a time into ChatGPT/Codex. Paste the returned JSON into `manual_scoring.xlsx` column `model_json`.\n\n")
        for i, task in enumerate(tasks, 1):
            f.write(f"## Task {i}: {task['task_id']}\n\n")
            f.write("```text\n")
            f.write(task["prompt"])
            f.write("\n```\n\n")

    xlsx_path = out_dir / "manual_scoring.xlsx"
    df = pd.DataFrame(tasks)
    ordered = [
        "task_id", "transcriptid", "segment", "ticker", "call_date", "fpedats",
        "sue_win", "eps_vol", "numest", "io_pct", "text_excerpt",
        "model_json", "human_check", "notes", "prompt"
    ]
    ordered = [c for c in ordered if c in df.columns]
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df[ordered].to_excel(writer, sheet_name="Manual Scoring", index=False)
        workbook = writer.book
        ws = writer.sheets["Manual Scoring"]
        ws.freeze_panes = "A2"
        widths = {
            "A": 24, "B": 14, "C": 12, "D": 12, "E": 14, "F": 14,
            "G": 12, "H": 12, "I": 10, "J": 10, "K": 80, "L": 60,
            "M": 16, "N": 28, "O": 100
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
        for row_cells in ws.iter_rows():
            for cell in row_cells:
                cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")

    print(f"Wrote {len(tasks)} segment tasks")
    print(f"JSONL: {jsonl_path}")
    print(f"Prompts: {md_path}")
    print(f"Manual scoring workbook: {xlsx_path}")


if __name__ == "__main__":
    main()

