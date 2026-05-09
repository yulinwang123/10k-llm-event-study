"""
week2_llama_inference.py
────────────────────────
vLLM batch inference for Llama-3.1-8B-Instruct on Midway3.

Usage (on GPU node):
    python week2_llama_inference.py \
        --input  /scratch/midway3/$USER/llm_batches/batch_000.jsonl \
        --output /scratch/midway3/$USER/llm_out/results_000.jsonl \
        --model  /scratch/midway3/$USER/models/Meta-Llama-3.1-8B-Instruct

SLURM job arrays pass $SLURM_ARRAY_TASK_ID to select the shard.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True,  help="Input JSONL batch file")
    p.add_argument("--output", required=True,  help="Output JSONL results file")
    p.add_argument("--model",  required=True,  help="Path or HF model ID")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="Number of GPUs (use 2 for A100 pair)")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Max new tokens per response")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="vLLM request batch size")
    return p.parse_args()


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[INFO] Loaded {len(records)} records from {path}", flush=True)
    return records


def build_prompt(messages: list[dict], tokenizer) -> str:
    """Apply Llama-3 chat template."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# ── Prompt construction (call this in the notebook to build messages) ──────
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


def build_messages(mda_text: str, max_chars: int = 6000) -> list[dict]:
    """
    Build the chat messages list for a single MD&A filing.

    We use the FIRST 6000 chars because:
    - The opening of MD&A contains the executive summary and forward guidance
    - The tail is usually detailed segment tables and legal boilerplate
    - 6000 chars ≈ 1200 tokens, well within Llama's 8192 context

    For the full project you could experiment with:
    - First 6000 chars only (captures strategic narrative)
    - Middle 3000 + Last 3000 (captures both strategy and risk section)
    - Full text chunked + aggregated (expensive but most complete)
    """
    excerpt = mda_text[:max_chars].strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"MD&A excerpt (first ~6000 characters):\n\n{excerpt}"}
    ]


def parse_json_output(raw: str) -> dict | None:
    """Extract JSON object from model output, handling extra text."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def validate_scores(d: dict) -> dict:
    """Clamp all score values to [0, 10]."""
    for key in ["management_optimism", "guidance_specificity",
                "uncertainty_hedging", "risk_framing"]:
        val = d.get(key)
        try:
            d[key] = max(0, min(10, int(val)))
        except (TypeError, ValueError):
            d[key] = None
    return d


def main():
    args = parse_args()

    # ── Load vLLM ──────────────────────────────────────────────────────────
    print("[INFO] Importing vLLM …", flush=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[INFO] Loading model: {args.model}", flush=True)
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="half",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_model_len=4096,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[INFO] Model loaded in {time.time()-t0:.1f}s", flush=True)

    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["\n\n"],    # stop after JSON closes
    )

    # ── Load records ───────────────────────────────────────────────────────
    records  = load_records(args.input)
    prompts  = [build_prompt(rec["messages"], tokenizer) for rec in records]

    # ── Run inference ─────────────────────────────────────────────────────
    print(f"[INFO] Running inference on {len(prompts)} prompts …", flush=True)
    t1       = time.time()
    outputs  = llm.generate(prompts, sampling)
    elapsed  = time.time() - t1
    print(f"[INFO] Inference done in {elapsed:.1f}s  "
          f"({elapsed/len(prompts):.2f}s per filing)", flush=True)

    # ── Write results ─────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_parsed = 0
    n_failed = 0

    with open(args.output, "w") as fout:
        for rec, out in zip(records, outputs):
            raw_text = out.outputs[0].text if out.outputs else ""
            parsed   = parse_json_output(raw_text)

            result = {
                "id":     rec["id"],
                "gvkey":  rec["gvkey"],
                "fyear":  rec["fyear"],
                "ticker": rec["ticker"],
                "output": raw_text,
            }

            if parsed:
                parsed = validate_scores(parsed)
                result["scores"] = parsed
                n_parsed += 1
            else:
                result["scores"] = None
                n_failed += 1

            fout.write(json.dumps(result) + "\n")

    print(f"[INFO] Results written → {args.output}")
    print(f"       Parsed OK: {n_parsed}  |  Parse failed: {n_failed}")
    if n_failed > 0:
        fail_rate = n_failed / len(records) * 100
        print(f"  ⚠  Failure rate: {fail_rate:.1f}%  (check output field for raw text)")


if __name__ == "__main__":
    main()
