#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RACER_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${RACER_ROOT}/.." && pwd)}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${WORKSPACE_ROOT}/Megatron-LM-FT}"
DATA_PATH="${DATA_PATH:-${WORKSPACE_ROOT}/data/my_shakespeare_text_document}"
GPT2_VOCAB_FILE="${GPT2_VOCAB_FILE:-${VOCAB_FILE:-${WORKSPACE_ROOT}/gpt2_vocab/vocab.json}}"
GPT2_MERGE_FILE="${GPT2_MERGE_FILE:-${MERGE_FILE:-${WORKSPACE_ROOT}/gpt2_vocab/merges.txt}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RACER_ROOT}/results/megatron_csd_restart}"

cd "${RACER_ROOT}"

RACER_CSD_PROFILE_LOG="${RACER_CSD_PROFILE_LOG:-1}" \
RACER_CSD_CHECKSUM_TYPE="${RACER_CSD_CHECKSUM_TYPE:-sample64}" \
RACER_CSD_MANIFEST_UPDATE_MODE="${RACER_CSD_MANIFEST_UPDATE_MODE:-batch}" \
RACER_CSD_DIRECT_TENSOR_IPC="${RACER_CSD_DIRECT_TENSOR_IPC:-0}" \
python examples/run_megatron_csd_restart_test.py \
  --model 5.3b \
  --racer-root "${RACER_ROOT}" \
  --megatron-root "${MEGATRON_ROOT}" \
  --data-path "${DATA_PATH}" \
  --vocab-file "${GPT2_VOCAB_FILE}" \
  --merge-file "${GPT2_MERGE_FILE}" \
  --output-root "${OUTPUT_ROOT}" \
  --csd-backend "${CSD_BACKEND:-native_pinned}" \
  --csd-native-pinned-total-bytes "${CSD_NATIVE_PINNED_TOTAL_BYTES:-274877906944}" \
  --csd-native-pinned-segment-bytes "${CSD_NATIVE_PINNED_SEGMENT_BYTES:-1073741824}" \
  --racer-buffer-size "${RACER_BUFFER_SIZE:-1073741824}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}" \
  --save-interval "${SAVE_INTERVAL:-1}" \
  --kill-after-iter "${KILL_AFTER_ITER:-1}" \
  --resume-train-iters "${RESUME_TRAIN_ITERS:-2}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-8}" \
  --csd-ready-timeout-seconds "${CSD_READY_TIMEOUT_SECONDS:-180}" \
  --timeout-seconds "${TIMEOUT_SECONDS:-10800}"
