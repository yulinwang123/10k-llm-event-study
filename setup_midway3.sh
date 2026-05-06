#!/bin/bash
# setup_midway3.sh  —  Run ONCE on a Midway3 login node to set up everything
# Usage:  bash setup_midway3.sh

set -euo pipefail

SCRATCH="/scratch/midway3/${USER}"
MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_DIR="${SCRATCH}/models/Meta-Llama-3.1-8B-Instruct"
BATCH_DIR="${SCRATCH}/llm_batches"

echo "=== Midway3 Setup for LLM 10-K Inference ==="
echo "User:    ${USER}"
echo "Scratch: ${SCRATCH}"
echo ""

# ── Step 1: Create directory structure ────────────────────────────────────────
mkdir -p "${SCRATCH}/models"
mkdir -p "${BATCH_DIR}"
mkdir -p "${SCRATCH}/llm_out"
echo "✓ Directories created"

# ── Step 2: Conda environment ──────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -q "^vllm_env"; then
    echo "✓ vllm_env already exists"
else
    echo "Creating conda env: vllm_env …"
    conda create -y -n vllm_env python=3.11
    conda activate vllm_env
    pip install vllm transformers accelerate huggingface_hub
    echo "✓ vllm_env created"
fi

conda activate vllm_env

# ── Step 3: Download model ─────────────────────────────────────────────────────
if [[ -d "${MODEL_DIR}" && -f "${MODEL_DIR}/config.json" ]]; then
    echo "✓ Model already downloaded: ${MODEL_DIR}"
else
    echo "Downloading ${MODEL_ID} → ${MODEL_DIR}"
    echo "This takes ~15 min on Midway3's fast scratch I/O …"
    huggingface-cli download "${MODEL_ID}" \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False
    echo "✓ Model downloaded"
fi

# ── Step 4: Upload batch files ─────────────────────────────────────────────────
echo ""
echo "=== Next step: upload batch files from your laptop ==="
echo ""
echo "  scp 'data_pilot/llm_batches/batch_*.jsonl' \\"
echo "      midway3.rcc.uchicago.edu:${BATCH_DIR}/"
echo ""
echo "Then check how many shards you have:"
echo "  ls ${BATCH_DIR}/batch_*.jsonl | wc -l"
echo ""
echo "Update submit_llama.sh line:  #SBATCH --array=0-<N-1>%5"
echo ""
echo "Then submit:"
echo "  sbatch submit_llama.sh"
echo ""
echo "=== Setup complete ==="
