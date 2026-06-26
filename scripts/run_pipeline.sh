#!/usr/bin/env bash
# Run the full data pipeline in order.
# Each script saves its output to data/ before the next one runs.
# Usage: bash scripts/run_pipeline.sh  (from 10K/ root)

set -e   # exit on first error
cd "$(dirname "$0")/.."

echo "========================================"
echo "Step 1/4  pull_transcripts.py"
echo "========================================"
python -u scripts/pull_transcripts.py

echo ""
echo "========================================"
echo "Step 2/4  pull_ibes_sue.py"
echo "========================================"
python -u scripts/pull_ibes_sue.py

echo ""
echo "========================================"
echo "Step 3/4  pull_crsp_car.py"
echo "========================================"
python -u scripts/pull_crsp_car.py

echo ""
echo "========================================"
echo "Step 4a/4  pull_controls.py"
echo "========================================"
python -u scripts/pull_controls.py

echo ""
echo "========================================"
echo "Step 4b/4  merge_event_panel.py"
echo "========================================"
python -u scripts/merge_event_panel.py

echo ""
echo "========================================"
echo "Step 5/6  compute_text_features.py"
echo "========================================"
python -u scripts/compute_text_features.py

echo ""
echo "========================================"
echo "Step 6a/6  study_a_ols.py"
echo "========================================"
python -u scripts/study_a_ols.py

echo ""
echo "========================================"
echo "Step 6b/6  study_a_dml.py"
echo "========================================"
python -u scripts/study_a_dml.py

echo ""
echo "========================================"
echo "Pipeline complete → results/study_a_dml_summary.txt"
echo "========================================"
