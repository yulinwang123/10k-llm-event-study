#!/bin/bash
#SBATCH --job-name=llama_10k
#SBATCH --account=macs30113
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100                    # 1 × A100-40GB is enough for 8B model
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/llama_%A_%a.out
#SBATCH --error=logs/llama_%A_%a.err
#SBATCH --array=0-2%3                   # 3 shards for pilot; adjust for full run

# ── Usage ────────────────────────────────────────────────────────────────────
# First run (pilot, 3 shards of ~20 filings):
#   sbatch submit_llama.sh
#
# Full run (adjust array range to match number of batch_NNN.jsonl files):
#   sbatch --array=0-249%10 submit_llama.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export VLLM_ATTENTION_BACKEND=FLASHINFER

module load cuda/12.3
mkdir -p logs

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH="/scratch/midway3/${USER}"
MODEL_DIR="${SCRATCH}/models/llama3-8b"
BATCH_DIR="${SCRATCH}/10k_data/10k-project/llm_batches"
OUT_DIR="${SCRATCH}/10k_data/10k-project/llm_out"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SHARD=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
INPUT="${BATCH_DIR}/batch_${SHARD}.jsonl"
OUTPUT="${OUT_DIR}/results_${SHARD}.jsonl"

# ── Environment ───────────────────────────────────────────────────────────────
VENV="/scratch/midway3/${USER}/vllm_env"
PYTHON="${VENV}/bin/python"

echo "================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}  →  shard ${SHARD}"
echo "Node:       ${SLURMD_NODENAME}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Input:      ${INPUT}"
echo "Output:     ${OUTPUT}"
echo "================================================"

# ── Sanity check ──────────────────────────────────────────────────────────────
if [[ ! -f "${INPUT}" ]]; then
    echo "ERROR: Input file not found: ${INPUT}"
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}"
    echo "Run: huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct"
    exit 1
fi

mkdir -p "${OUT_DIR}"

# ── Run inference ─────────────────────────────────────────────────────────────
"${PYTHON}" "${SCRATCH}/week2_llama_inference.py" \
    --input   "${INPUT}"      \
    --output  "${OUTPUT}"     \
    --model   "${MODEL_DIR}"  \
    --tensor-parallel-size 1  \
    --max-tokens 128          \
    --temperature 0.0         \
    --batch-size 8

echo "✓ Shard ${SHARD} complete"
