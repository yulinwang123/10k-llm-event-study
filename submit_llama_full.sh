#!/bin/bash
# submit_llama_full.sh
# 自动分批提交llama job，每批12个，等跑完再提交下一批
# Usage: bash submit_llama_full.sh

TOTAL=226        # 总shard数 (0-225)
BATCH=12         # 每批最多提交数量
SCRIPT="/scratch/midway3/yulinwang/submit_llama.sh"
POLL=60          # 每隔多少秒检查一次

start=0
batch_num=1

while [ $start -lt $TOTAL ]; do
    end=$((start + BATCH - 1))
    if [ $end -ge $TOTAL ]; then
        end=$((TOTAL - 1))
    fi

    echo "================================================"
    echo "Batch $batch_num: submitting array ${start}-${end}"
    echo "================================================"

    JOBID=$(sbatch --array=${start}-${end}%${BATCH} "$SCRIPT" | awk '{print $4}')

    if [ -z "$JOBID" ]; then
        echo "ERROR: Failed to submit batch $batch_num"
        exit 1
    fi

    echo "Submitted job $JOBID (array ${start}-${end})"

    # 等待这批job全部完成
    while true; do
        RUNNING=$(squeue -j "$JOBID" -h 2>/dev/null | wc -l)
        if [ "$RUNNING" -eq 0 ]; then
            echo "Batch $batch_num complete."
            break
        fi
        echo "  $(date '+%H:%M:%S') — $RUNNING job(s) still running..."
        sleep $POLL
    done

    start=$((end + 1))
    batch_num=$((batch_num + 1))
done

echo "================================================"
echo "All $TOTAL shards submitted and completed!"
echo "Results in: /scratch/midway3/yulinwang/10k_data/10k-project/llm_out/"
echo "================================================"
