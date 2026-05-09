#!/bin/bash
#SBATCH --job-name=finbert_embed
#SBATCH --account=macs30113
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
VENV=/scratch/midway3/yulinwang/vllm_env
PYTHON=$VENV/bin/python

# ── HuggingFace cache ──────────────────────────────────────────────────────────
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_HOME=$SCRATCH/hf_cache
export SENTENCE_TRANSFORMERS_HOME=$SCRATCH/hf_cache
mkdir -p $SCRATCH/hf_cache

echo "========================================="
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "========================================="
nvidia-smi

$PYTHON $SCRATCH/midway3_finbert_embed.py \
    --data-root $SCRATCH/10k_data/10k-project \
    --batch-size 64

echo "========================================="
echo "Finished: $(date)"
echo "========================================="
