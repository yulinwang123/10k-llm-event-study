"""
Parse human-in-the-loop LLM demo outputs pasted into manual_scoring.xlsx.

Input:
  data/llm_demo/manual_scoring.xlsx
  - Fill the model_json column with the JSON returned by ChatGPT/Codex.

Outputs:
  data/llm_demo/llm_demo_features.csv
  data/llm_demo/llm_demo_features.xlsx
  results/llm_demo_validation.md
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"

FIELDS = {
    "contextual_sentiment": (-1, 1),
    "uncertainty": (0, 1),
    "hedging": (0, 1),
    "forward_looking_specificity": (0, 1),
    "informativeness": (0, 1),
    "defensiveness": (0, 1),
    "qa_evasiveness": (0, 1),
    "self_serving_attribution": (0, 1),
}


def clean_json_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    return match.group(0) if match else text


def parse_one(text: str):
    if not isinstance(text, str) or not text.strip():
        return None, "missing"
    try:
        return json.loads(clean_json_text(text)), "ok"
    except Exception as e:
        return None, f"parse_error: {type(e).__name__}"


def validate_value(field, value, segment):
    if field == "qa_evasiveness" and segment == "prepared" and value is None:
        return None
    if value is None:
        return f"{field}: missing"
    if not isinstance(value, (int, float)):
        return f"{field}: nonnumeric"
    lo, hi = FIELDS[field]
    if value < lo or value > hi:
        return f"{field}: out_of_range"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DATA / "llm_demo" / "manual_scoring.xlsx"))
    parser.add_argument("--out-dir", default=str(DATA / "llm_demo"))
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    df = pd.read_excel(input_path, sheet_name="Manual Scoring")

    parsed_rows = []
    for _, row in df.iterrows():
        parsed, status = parse_one(row.get("model_json", ""))
        out = row.to_dict()
        out["parse_status"] = status
        issues = []
        if parsed is not None:
            for field in FIELDS:
                value = parsed.get(field)
                out[field] = value
                issue = validate_value(field, value, row.get("segment"))
                if issue:
                    issues.append(issue)
            out["short_rationale"] = parsed.get("short_rationale", "")
        else:
            for field in FIELDS:
                out[field] = None
            out["short_rationale"] = ""
        out["validation_issues"] = "; ".join(issues)
        parsed_rows.append(out)

    parsed_df = pd.DataFrame(parsed_rows)
    parsed_df.to_csv(out_dir / "llm_demo_features.csv", index=False)

    xlsx_path = out_dir / "llm_demo_features.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        parsed_df.to_excel(writer, sheet_name="Parsed Features", index=False)
        ws = writer.sheets["Parsed Features"]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            letter = col[0].column_letter
            ws.column_dimensions[letter].width = min(50, max(12, max(len(str(c.value or "")) for c in col[:20]) + 2))
        for row_cells in ws.iter_rows():
            for cell in row_cells:
                cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")

    scored = parsed_df[parsed_df["parse_status"].eq("ok")]
    report = []
    report.append("# LLM Demo Validation\n")
    report.append(f"Input file: `{input_path}`\n")
    report.append(f"Rows: {len(parsed_df):,}\n")
    report.append(f"Parsed OK: {len(scored):,}\n")
    report.append(f"Parse/missing failures: {len(parsed_df) - len(scored):,}\n")
    issue_count = parsed_df["validation_issues"].fillna("").ne("").sum()
    report.append(f"Rows with validation issues: {issue_count:,}\n")

    if not scored.empty:
        report.append("\n## Summary Statistics\n")
        stats_cols = [c for c in FIELDS if c in scored.columns]
        report.append(scored[stats_cols].describe().round(3).to_markdown())
        report.append("\n\n## Segment Counts\n")
        report.append(scored["segment"].value_counts(dropna=False).to_markdown())

    if issue_count:
        report.append("\n\n## Validation Issues\n")
        issues = parsed_df.loc[parsed_df["validation_issues"].fillna("").ne(""), ["task_id", "validation_issues"]]
        report.append(issues.to_markdown(index=False))

    report_path = RESULTS / "llm_demo_validation.md"
    report_path.write_text("\n".join(report))

    print(f"CSV: {out_dir / 'llm_demo_features.csv'}")
    print(f"XLSX: {xlsx_path}")
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()

