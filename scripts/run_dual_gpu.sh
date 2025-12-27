#!/usr/bin/env bash
set -euo pipefail

# Simple dual-GPU launcher.
# Default executable: bin/benchmark_full_cuda
# Optional: set EXEC=bin/benchmark_cuda to run CUDA-only benchmark (includes CUDA_Const / CUDA_Policy comparisons).
# Usage: EXEC=bin/benchmark_cuda ./scripts/run_dual_gpu.sh [image_size] [num_images] [num_threads] [iterations]
# Defaults: 4096 4 16 1

IMAGE_SIZE=${1:-4096}
TOTAL_IMAGES=${2:-4}
NUM_THREADS=${3:-16}
ITERATIONS=${4:-1}

EXEC=${EXEC:-./bin/benchmark_full_cuda}

# If using benchmark_cuda, it does not emit CSV; disable CSV handling.
USE_CSV=1
if [[ "$EXEC" == *benchmark_cuda ]]; then
  USE_CSV=0
fi
if [[ ! -x "$EXEC" ]]; then
  echo "error: $EXEC not found or not executable. Build it first." >&2
  exit 1
fi

# Split images between two GPUs (GPU0 gets the extra if odd)
GPU0_IMAGES=$(( (TOTAL_IMAGES + 1) / 2 ))
GPU1_IMAGES=$(( TOTAL_IMAGES - GPU0_IMAGES ))
if [[ $GPU1_IMAGES -le 0 ]]; then
  echo "warning: only one chunk to run; GPU1 will be skipped." >&2
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR=results
mkdir -p "$OUT_DIR"
OUT0="$OUT_DIR/benchmark_gpu0_${TS}.csv"
OUT1="$OUT_DIR/benchmark_gpu1_${TS}.csv"
MERGED="$OUT_DIR/benchmark_dual_${TS}.csv"

run_chunk() {
  local gpu_id=$1
  local imgs=$2
  local outfile=$3
  if [[ $imgs -le 0 ]]; then
    return
  fi
  echo "[GPU ${gpu_id}] Running ${imgs} images..."
  CUDA_VISIBLE_DEVICES=${gpu_id} $EXEC "$IMAGE_SIZE" "$imgs" "$NUM_THREADS" "$ITERATIONS" \
    >/tmp/benchmark_gpu${gpu_id}_${TS}.log
  if [[ $USE_CSV -eq 1 ]]; then
    # Move CSV emitted in CWD
    if [[ -f benchmark_full_results.csv ]]; then
      mv benchmark_full_results.csv "$outfile"
    else
      echo "error: benchmark_full_results.csv not found after GPU ${gpu_id} run" >&2
      exit 1
    fi
  fi
}

run_chunk 0 "$GPU0_IMAGES" "$OUT0" &
PID0=$!
run_chunk 1 "$GPU1_IMAGES" "$OUT1" &
PID1=$!

wait $PID0
if [[ $GPU1_IMAGES -gt 0 ]]; then
  wait $PID1
fi

if [[ $USE_CSV -eq 1 ]]; then
  # Merge CSV (keep header from first, append the rest)
  if [[ -f "$OUT0" ]]; then
    head -n 1 "$OUT0" > "$MERGED"
    tail -n +2 "$OUT0" >> "$MERGED"
  fi
  if [[ -f "$OUT1" ]]; then
    tail -n +2 "$OUT1" >> "$MERGED"
  fi
  echo "Done. Logs: /tmp/benchmark_gpu0_${TS}.log /tmp/benchmark_gpu1_${TS}.log"
  echo "Chunk CSVs: $OUT0 $OUT1"
  echo "Merged CSV: $MERGED"
else
  echo "Done (benchmark_cuda mode, no CSV emitted). Logs: /tmp/benchmark_gpu0_${TS}.log /tmp/benchmark_gpu1_${TS}.log"
fi