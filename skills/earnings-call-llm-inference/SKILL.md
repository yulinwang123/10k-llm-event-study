---
name: earnings-call-llm-inference
description: Use when preparing, running, validating, or integrating LLM inference for the earnings-announcement project: extracting semantic narrative measures from CIQ earnings call transcripts and comparing hard earnings surprise versus managerial narrative pricing weights.
---

# Earnings Call LLM Inference

Use this skill for the LLM stage of the earnings-announcement project. The goal is to replace baseline word-bag narrative proxies with structured semantic measures from earnings call transcripts, then re-run the relative-weighting pipeline.

## Core Question

Once narrative is measured more semantically, how much independent weight does the market place on hard earnings surprise versus managerial narrative, and does this weighting vary with the information environment?

## Guardrails

- Do not send WRDS/CIQ transcript text to external APIs without explicit user approval and confirmation that the data-use terms allow it.
- Do not include returns, CAR, post-announcement drift, analyst revisions after the event, or other future outcomes in LLM prompts.
- Default prompts should use transcript text only. If conditioning on SUE or reported EPS is requested, document it clearly and never include market outcomes.
- Start with a small audited pilot before full-scale inference.
- Output machine-readable JSON/JSONL first; convert to Parquet only after validation.
- Keep raw LLM outputs and parsed feature tables separate.

## Project Inputs

Expected local inputs:

```text
data/event_panel.parquet
data/transcripts_tmp/chunk_*.parquet
data/text_features.parquet
```

Key identifiers:

```text
transcriptid
gvkey
permno
ticker
call_date
fpedats
```

Transcript text usually lives in:

```text
data/transcripts_tmp/chunk_*.parquet
```

with columns like:

```text
transcriptid
prepared_text
qa_text
```

## Desired Output Schema

Create one row per transcript and segment when possible:

```text
transcriptid
segment                     prepared | qa | full
model_name
prompt_version
input_token_estimate
parse_status                ok | parse_error | skipped
contextual_sentiment         -1 to 1
uncertainty                  0 to 1
hedging                      0 to 1
forward_looking              0 to 1
defensiveness                0 to 1
informativeness              0 to 1
qa_evasiveness               0 to 1, mostly for Q&A
optimism                     0 to 1
short_rationale              <= 40 words; optional for audit, not for regressions
```

Recommended files:

```text
data/llm_out/raw_outputs/*.jsonl
data/llm_out/llm_narrative_features.parquet
results/llm_feature_validation.md
```

## Workflow

### 1. Confirm Runtime and Data Constraints

Before running inference:

1. Identify the intended inference backend: local model, vLLM, OpenAI API, cluster job, or professor PI account.
2. Confirm whether transcript text may be sent to that backend.
3. Confirm expected batch size and compute budget.
4. Run a small pilot first, e.g. 50-200 transcripts.

For a no-API demo using ChatGPT/Codex manually, use:

```bash
python scripts/llm_demo_build_batch.py --n 6 --max-chars 5000
```

This creates `data/llm_demo/manual_scoring.xlsx` and `data/llm_demo/demo_prompts.md`.
Paste model JSON responses into the `model_json` column, then parse with:

```bash
python scripts/llm_demo_parse_outputs.py
```

### 2. Build a Pilot Dataset

Use `event_panel.parquet` to select transcript IDs, then read only those rows from `transcripts_tmp/chunk_*.parquet`.

Pilot should include variation in:

- positive and negative SUE
- high and low earnings volatility
- high and low analyst coverage
- prepared remarks and Q&A availability
- short and long transcripts

### 3. Prompt Contract

Use strict JSON output. The model should score narrative dimensions, not predict returns.

For the full production prompt with finance-specific anchors and scoring rules, read `references/prompt_v1.md`.

Prompt requirements:

- Include the segment label (`prepared` or `qa`).
- Ask for numeric scores on fixed scales.
- Require valid JSON only.
- Tell the model not to infer market reaction.
- Keep rationales short and optional.

A minimal prompt pattern:

```text
You are extracting structured narrative features from an earnings call transcript segment.
Use only the text below. Do not infer stock returns, investor reaction, or future outcomes.

Segment: {segment}
Transcript text:
{text}

Return valid JSON only with these fields:
{
  "contextual_sentiment": number from -1 to 1,
  "uncertainty": number from 0 to 1,
  "hedging": number from 0 to 1,
  "forward_looking": number from 0 to 1,
  "defensiveness": number from 0 to 1,
  "informativeness": number from 0 to 1,
  "qa_evasiveness": number from 0 to 1,
  "optimism": number from 0 to 1,
  "short_rationale": string of at most 40 words
}
```

### 4. Parse and Validate

After inference:

- Parse all JSON records.
- Report parse failure rate.
- Check score ranges.
- Check missingness by segment.
- Check distributions and outliers.
- Compare LLM measures with LM dictionary measures from `data/text_features.parquet`.
- Compare prepared remarks vs Q&A; Q&A should plausibly show more unscripted uncertainty, defensiveness, or evasiveness.
- Manually audit at least 20-50 examples before full-scale runs.

Validation report should include:

```text
N transcripts processed
N prepared / Q&A segments
parse failure rate
summary statistics
correlation with LM tone / uncertainty / fog
examples of high and low scores
known failure modes
decision: scale / revise prompt / stop
```

### 5. Integrate with Empirical Pipeline

Merge LLM features back to the event panel by `transcriptid`.

Run relative-weighting specifications:

```text
CAR_i = theta * SUE_i + controls + FE + error_i
CAR_i = theta_LM * SUE_i + beta_LM * Narrative_LM_i + controls + FE + error_i
CAR_i = theta_LLM * SUE_i + beta_LLM * Narrative_LLM_i + controls + FE + error_i
```

Then test heterogeneity:

```text
CAR_i = theta * SUE_i
      + beta * Narrative_i
      + theta_X * SUE_i * X_i
      + beta_X * Narrative_i * X_i
      + controls + FE + error_i
```

where `X` can include:

```text
eps_vol
numest
io_pct
```

Use DML / cross-fitting only as robustness or adjustment, not as the main contribution.

## Completion Criteria

The LLM stage is complete only when:

- Pilot outputs parse cleanly.
- Validation report is written.
- LLM features merge to the event panel.
- At least one baseline relative-weighting table is produced.
- Prompt version, model name, and inference backend are documented.
