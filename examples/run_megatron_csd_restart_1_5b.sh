#!/usr/bin/env bash
set -euo pipefail

cd /workspace/racer

RACER_CSD_PROFILE_LOG="${RACER_CSD_PROFILE_LOG:-1}" \
RACER_CSD_CHECKSUM_TYPE="${RACER_CSD_CHECKSUM_TYPE:-sample64}" \
RACER_CSD_MANIFEST_UPDATE_MODE="${RACER_CSD_MANIFEST_UPDATE_MODE:-batch}" \
RACER_CSD_DIRECT_TENSOR_IPC="${RACER_CSD_DIRECT_TENSOR_IPC:-0}" \
python examples/run_megatron_csd_restart_test.py \
  --model 1.5b \
  --csd-backend "${CSD_BACKEND:-native_pinned}" \
  --csd-native-pinned-total-bytes "${CSD_NATIVE_PINNED_TOTAL_BYTES:-103079215104}" \
  --csd-native-pinned-segment-bytes "${CSD_NATIVE_PINNED_SEGMENT_BYTES:-1073741824}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}" \
  --save-interval "${SAVE_INTERVAL:-2}" \
  --kill-after-iter "${KILL_AFTER_ITER:-2}" \
  --resume-train-iters "${RESUME_TRAIN_ITERS:-4}" \
  --csd-ready-timeout-seconds "${CSD_READY_TIMEOUT_SECONDS:-120}" \
  --timeout-seconds "${TIMEOUT_SECONDS:-7200}"
