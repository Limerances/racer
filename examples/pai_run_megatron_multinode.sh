#!/usr/bin/env bash
set -euo pipefail

# 多节点 Megatron/RACER 手动启动模板。
#
# 同一个脚本要在所有节点上运行；所有参数保持一致，只有 NODE_RANK 不同。
# 例子，三个 PAI 节点：node0/node1 是 8 个 train ranks，node2 是 1 个 remote spare rank。
#
#   # node0
#   MODE=racer_pinned_remote_spare NODE_RANK=0 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=10.0.0.1 \
#     RACER_K=6 RACER_M=2 RACER_TRAIN_RANKS=0-7 RACER_SPARE_RANKS=8 RACER_RUNTIME_PORT=29610 \
#     bash examples/pai_run_megatron_multinode.sh
#
#   # node1
#   MODE=racer_pinned_remote_spare NODE_RANK=1 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=10.0.0.1 \
#     RACER_K=6 RACER_M=2 RACER_TRAIN_RANKS=0-7 RACER_SPARE_RANKS=8 RACER_RUNTIME_PORT=29610 \
#     bash examples/pai_run_megatron_multinode.sh
#
#   # node2
#   MODE=racer_pinned_remote_spare NODE_RANK=2 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=10.0.0.1 \
#     RACER_K=6 RACER_M=2 RACER_TRAIN_RANKS=0-7 RACER_SPARE_RANKS=8 RACER_RUNTIME_PORT=29610 \
#     bash examples/pai_run_megatron_multinode.sh
#
# MODE 可选：
#   baseline                 普通 Megatron 多节点训练，不启用 RACER。
#   racer_pinned_single_node 单节点 RACER pinned/CSD；torchrun 只启动 train ranks，spare 是额外 CUDA device。
#   racer_pinned_per_node    多节点 RACER pinned/CSD；每个节点启动一个本地 CSD。
#   racer_pinned_multinode   racer_pinned_per_node 的兼容别名。
#   racer_pinned_remote_spare 多节点 pinned/CSD，spare worker 在单独节点。
#   racer_egm                EGM CSD 后端；默认 local spare，可用 RACER_SPARE_LAUNCH_MODE=remote。
#   racer_egm_remote_spare   EGM CSD 后端，spare worker 在单独节点。
#
# 默认 RACER 拓扑是 8 个 train rank，k=6,m=2，另有 1 个 spare CUDA device：
#   RACER_K=6
#   RACER_M=2
#   RACER_TRAIN_RANKS=0,1,2,3,4,5,6,7
#   RACER_SPARE_RANKS=8
#
# local spare 模式下 RACER_SPARE_RANKS 仍表示 coordinator 节点可见的 spare CUDA device id。
# remote spare 模式下 RACER_SPARE_RANKS 表示外部 spare worker 的 RACER rank，例如 8。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RACER_ROOT="${RACER_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${RACER_ROOT}/.." && pwd)}"
MEGATRON_ROOT="${MEGATRON_ROOT:-${WORKSPACE_ROOT}/Megatron-LM-FT}"

MODE="${MODE:-baseline}"
MODEL_SIZE="${MODEL_SIZE:-1.5b}"
DRY_RUN="${DRY_RUN:-0}"
NODE_ROLE="${NODE_ROLE:-auto}"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
HOSTFILE="${HOSTFILE:-/mnt/workspace/pai_nodes.txt}"

DATA_PATH="${DATA_PATH:-${WORKSPACE_ROOT}/data/my_shakespeare_text_document}"
GPT2_VOCAB_FILE="${GPT2_VOCAB_FILE:-${VOCAB_FILE:-${WORKSPACE_ROOT}/gpt2_vocab/vocab.json}}"
GPT2_MERGE_FILE="${GPT2_MERGE_FILE:-${MERGE_FILE:-${WORKSPACE_ROOT}/gpt2_vocab/merges.txt}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/pai_runs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT}/checkpoints}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"

TRAIN_ITERS="${TRAIN_ITERS:-80}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"

TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-4}"

CSD_PORT="${CSD_PORT:-7007}"
CSD_START_TIMEOUT_SECONDS="${CSD_START_TIMEOUT_SECONDS:-300}"
CSD_NATIVE_PINNED_TOTAL_BYTES="${CSD_NATIVE_PINNED_TOTAL_BYTES:-}"
CSD_NATIVE_PINNED_SEGMENT_BYTES="${CSD_NATIVE_PINNED_SEGMENT_BYTES:-1073741824}"
CSD_NATIVE_PINNED_DEVICE="${CSD_NATIVE_PINNED_DEVICE:-0}"
RACER_CSD_MODE="${RACER_CSD_MODE:-managed}"
CSD_EGM_RUNTIME_FACTORY="${CSD_EGM_RUNTIME_FACTORY:-racer.egm_runtime:create_runtime}"
CSD_EGM_RUNTIME_CONFIG="${CSD_EGM_RUNTIME_CONFIG:-}"
CSD_EGM_POOL_ID="${CSD_EGM_POOL_ID:-}"
CSD_EGM_OWNER_NODE="${CSD_EGM_OWNER_NODE:-}"
CSD_EGM_OWNER_TRAY="${CSD_EGM_OWNER_TRAY:-}"
CSD_EGM_HOME_DEVICE="${CSD_EGM_HOME_DEVICE:-0}"
CSD_EGM_NUMA_ID="${CSD_EGM_NUMA_ID:-}"
CSD_EGM_ACCESSING_DEVICES="${CSD_EGM_ACCESSING_DEVICES:-}"
RACER_CSD_PROFILE_LOG="${RACER_CSD_PROFILE_LOG:-1}"
RACER_CSD_CHECKSUM_TYPE="${RACER_CSD_CHECKSUM_TYPE:-sample64}"
RACER_CSD_MANIFEST_UPDATE_MODE="${RACER_CSD_MANIFEST_UPDATE_MODE:-batch}"

RACER_K="${RACER_K:-6}"
RACER_M="${RACER_M:-2}"
RACER_TRAIN_RANKS="${RACER_TRAIN_RANKS:-0,1,2,3,4,5,6,7}"
RACER_SPARE_RANKS="${RACER_SPARE_RANKS:-${RACER_SPARE_CUDA_DEVICES:-8}}"
RACER_CSD_PER_NODE="${RACER_CSD_PER_NODE:-1}"
RACER_CSD_LOCAL_RANKS="${RACER_CSD_LOCAL_RANKS:-}"
RACER_CSD_LOCAL_COORDINATOR_RANK="${RACER_CSD_LOCAL_COORDINATOR_RANK:-}"
RACER_SPARE_LAUNCH_MODE="${RACER_SPARE_LAUNCH_MODE:-}"
RACER_RUNTIME_PORT="${RACER_RUNTIME_PORT:-29610}"
RACER_REMOTE_SPARE_CUDA_DEVICE="${RACER_REMOTE_SPARE_CUDA_DEVICE:-0}"

detect_node_rank() {
  if [[ -n "${NODE_RANK:-}" ]]; then
    echo "${NODE_RANK}"
    return
  fi
  for name in PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX PAI_WORKER_INDEX WORKER_INDEX DLC_WORKER_INDEX; do
    value="${!name:-}"
    if [[ -n "${value}" ]]; then
      echo "${value}"
      return
    fi
  done
  if [[ -f "${HOSTFILE}" ]]; then
    local host_short host_fqdn host_ips line index
    host_short="$(hostname)"
    host_fqdn="$(hostname -f 2>/dev/null || true)"
    host_ips="$(hostname -I 2>/dev/null || true)"
    index=0
    while read -r line; do
      [[ -z "${line}" ]] && continue
      if [[ "${line}" == "${host_short}" || "${line}" == "${host_fqdn}" || " ${host_ips} " == *" ${line} "* ]]; then
        echo "${index}"
        return
      fi
      index=$((index + 1))
    done < "${HOSTFILE}"
  fi
  echo "ERROR: 无法自动判断 NODE_RANK。请手动设置 NODE_RANK=0/1/2。" >&2
  exit 2
}

resolve_master_addr() {
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    echo "${MASTER_ADDR}"
    return
  fi
  if [[ -f "${HOSTFILE}" ]]; then
    sed -n '1p' "${HOSTFILE}"
    return
  fi
  echo "ERROR: 未设置 MASTER_ADDR，且 ${HOSTFILE} 不存在。" >&2
  exit 2
}

resolve_node_role() {
  local role="$1"
  if [[ "${role}" != "auto" ]]; then
    case "${role}" in
      train|spare)
        echo "${role}"
        return
        ;;
      *)
        echo "ERROR: NODE_ROLE 只能是 auto/train/spare，当前是 ${role}" >&2
        exit 3
        ;;
    esac
  fi
  case "${MODE}" in
    racer_pinned_remote_spare|racer_egm_remote_spare)
      if (( NODE_RANK >= NNODES )); then
        echo spare
      else
        echo train
      fi
      ;;
    *)
      echo train
      ;;
  esac
}

require_path() {
  local path="$1"
  local name="$2"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: ${name} 不存在: ${path}" >&2
    exit 3
  fi
}

require_indexed_dataset_prefix() {
  local prefix="$1"
  local name="$2"
  local missing=0
  if [[ ! -f "${prefix}.bin" ]]; then
    echo "ERROR: ${name}.bin 不存在: ${prefix}.bin" >&2
    missing=1
  fi
  if [[ ! -f "${prefix}.idx" ]]; then
    echo "ERROR: ${name}.idx 不存在: ${prefix}.idx" >&2
    missing=1
  fi
  if (( missing != 0 )); then
    echo "提示: ${name} 是 Megatron indexed dataset 前缀，不需要存在无后缀文件。" >&2
    echo "      当前 ${name}=${prefix}" >&2
    exit 3
  fi
}

expand_rank_list() {
  local value="$1"
  local -a out=()
  local item left right step rank
  IFS=',' read -ra parts <<< "${value}"
  for item in "${parts[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -z "${item}" ]] && continue
    if [[ "${item}" == *-* ]]; then
      left="${item%%-*}"
      right="${item#*-}"
      if [[ -z "${left}" || -z "${right}" ]]; then
        echo "ERROR: rank range 格式错误: ${item}" >&2
        exit 3
      fi
      if (( right >= left )); then
        step=1
      else
        step=-1
      fi
      rank="${left}"
      while true; do
        out+=("${rank}")
        [[ "${rank}" == "${right}" ]] && break
        rank=$((rank + step))
      done
    else
      out+=("${item}")
    fi
  done
  local IFS=,
  echo "${out[*]}"
}

rank_count() {
  local expanded="$1"
  if [[ -z "${expanded}" ]]; then
    echo 0
    return
  fi
  local -a ranks
  IFS=',' read -ra ranks <<< "${expanded}"
  echo "${#ranks[@]}"
}

first_rank() {
  local expanded="$1"
  local -a ranks
  IFS=',' read -ra ranks <<< "${expanded}"
  echo "${ranks[0]}"
}

validate_remote_spare_topology() {
  case "${MODE}" in
    racer_pinned_remote_spare|racer_egm_remote_spare)
      ;;
    *)
      return
      ;;
  esac
  if (( RACER_SPARE_RANK_COUNT != 1 )); then
    echo "ERROR: 当前 PAI remote spare 脚本一次只启动 1 个 remote spare rank。" >&2
    echo "       当前 RACER_SPARE_RANKS=${RACER_SPARE_RANKS_EXPANDED}" >&2
    exit 4
  fi
  local expected_spare_rank
  expected_spare_rank="${RACER_TRAIN_RANK_COUNT}"
  if [[ "${RACER_SPARE_RANKS_EXPANDED}" != "${expected_spare_rank}" ]]; then
    cat >&2 <<MSG
ERROR: remote spare rank 必须紧跟 train rank 连续编号。

当前：
  train rank 数=${RACER_TRAIN_RANK_COUNT}
  RACER_SPARE_RANKS=${RACER_SPARE_RANKS_EXPANDED}

应设置：
  RACER_SPARE_RANKS=${expected_spare_rank}

例如 8 个 train rank 时，remote spare rank 必须是 8。
MSG
    exit 4
  fi
}

default_local_rank_range() {
  local start end
  start=$((NODE_RANK * NPROC_PER_NODE))
  end=$((start + NPROC_PER_NODE - 1))
  if (( start == end )); then
    echo "${start}"
  else
    echo "${start}-${end}"
  fi
}

case "${MODEL_SIZE}" in
  1.5b)
    NUM_LAYERS="${NUM_LAYERS:-48}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-1600}"
    FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-6400}"
    NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-25}"
    DEFAULT_GLOBAL_BATCH_SIZE=16
    DEFAULT_CSD_NATIVE_PINNED_TOTAL_BYTES=103079215104
    ;;
  5.3b)
    NUM_LAYERS="${NUM_LAYERS:-64}"
    HIDDEN_SIZE="${HIDDEN_SIZE:-2560}"
    FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-10240}"
    NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-40}"
    DEFAULT_GLOBAL_BATCH_SIZE=8
    DEFAULT_CSD_NATIVE_PINNED_TOTAL_BYTES=274877906944
    ;;
  *)
    echo "ERROR: MODEL_SIZE 只能是 1.5b 或 5.3b，当前是 ${MODEL_SIZE}" >&2
    exit 3
    ;;
esac

NODE_RANK="$(detect_node_rank)"
MASTER_ADDR="$(resolve_master_addr)"
NODE_ROLE="$(resolve_node_role "${NODE_ROLE}")"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${DEFAULT_GLOBAL_BATCH_SIZE}}"
if [[ -z "${CSD_NATIVE_PINNED_TOTAL_BYTES}" ]]; then
  case "${MODE}" in
    racer_pinned_remote_spare|racer_egm_remote_spare)
      if [[ "${NODE_ROLE}" == "spare" ]]; then
        CSD_NATIVE_PINNED_TOTAL_BYTES=0
      else
        CSD_NATIVE_PINNED_TOTAL_BYTES="${DEFAULT_CSD_NATIVE_PINNED_TOTAL_BYTES}"
      fi
      ;;
    *)
      CSD_NATIVE_PINNED_TOTAL_BYTES="${DEFAULT_CSD_NATIVE_PINNED_TOTAL_BYTES}"
      ;;
  esac
fi
RACER_TRAIN_RANKS_EXPANDED="$(expand_rank_list "${RACER_TRAIN_RANKS}")"
RACER_SPARE_RANKS_EXPANDED="$(expand_rank_list "${RACER_SPARE_RANKS}")"
RACER_TRAIN_RANK_COUNT="$(rank_count "${RACER_TRAIN_RANKS_EXPANDED}")"
RACER_SPARE_RANK_COUNT="$(rank_count "${RACER_SPARE_RANKS_EXPANDED}")"
TORCHRUN_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
RACER_RUNTIME_WORLD_SIZE=$((RACER_TRAIN_RANK_COUNT + RACER_SPARE_RANK_COUNT))
validate_remote_spare_topology

require_path "${RACER_ROOT}" "RACER_ROOT"
require_path "${MEGATRON_ROOT}/pretrain_gpt.py" "Megatron pretrain_gpt.py"
if [[ "${DRY_RUN}" != "1" ]]; then
  require_indexed_dataset_prefix "${DATA_PATH}" "DATA_PATH"
  require_path "${GPT2_VOCAB_FILE}" "GPT2_VOCAB_FILE"
  require_path "${GPT2_MERGE_FILE}" "GPT2_MERGE_FILE"
fi

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}"

RUN_ID="${RUN_ID:-${MODEL_SIZE}_${MODE}_${NNODES}n${NPROC_PER_NODE}g}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/${RUN_ID}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard/${RUN_ID}}"
RACER_MANIFEST_DIR="${RACER_MANIFEST_DIR:-${OUTPUT_ROOT}/racer_manifests/${RUN_ID}}"
RACER_PROFILE_DIR="${RACER_PROFILE_DIR:-${OUTPUT_ROOT}/racer_profiles/${RUN_ID}}"
LOG_FILE="${LOG_ROOT}/${RUN_ID}.node${NODE_RANK}.log"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export PYTHONPATH="${RACER_ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"
export RACER_CSD_PROFILE_LOG
export RACER_CSD_CHECKSUM_TYPE
export RACER_CSD_MANIFEST_UPDATE_MODE

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

RACER_ARGS=()
CSD_PID=""

launch_csd_process() {
  if [[ "${RACER_CSD_MODE}" == "persistent" ]]; then
    setsid "$@" > "${CSD_DIR}/csd.log" 2>&1 &
  else
    "$@" > "${CSD_DIR}/csd.log" 2>&1 &
  fi
}

wait_for_local_csd() {
  local expected_backend="$1"
  local deadline output rc
  deadline=$((SECONDS + CSD_START_TIMEOUT_SECONDS))
  while true; do
    if [[ -n "${CSD_PID}" ]] && ! kill -0 "${CSD_PID}" >/dev/null 2>&1; then
      echo "ERROR: 本地 CSD 启动后已退出，backend=${expected_backend}，日志: ${CSD_DIR}/csd.log" >&2
      sed -n '1,160p' "${CSD_DIR}/csd.log" >&2 || true
      exit 4
    fi

    set +e
    output="$(
      EXPECTED_CSD_BACKEND="${expected_backend}" CSD_PORT="${CSD_PORT}" PYTHONPATH="${PYTHONPATH}" \
      python - <<'PY' 2>&1
import json
import os

from racer.csd import CheckpointStorageDaemonClient

expected = os.environ["EXPECTED_CSD_BACKEND"]
client = CheckpointStorageDaemonClient(("127.0.0.1", int(os.environ["CSD_PORT"])), authkey="racer-csd")
caps = dict(client.capabilities())
actual = str(caps.get("backend", caps.get("storage_backend", ""))).lower()
if actual != expected:
    raise RuntimeError(f"expected CSD backend {expected}, got {actual or '<unknown>'}: {caps}")
if expected == "native_pinned" and not (
    bool(caps.get("supports_cuda_ipc")) and bool(caps.get("supports_async_copy"))
):
    raise RuntimeError(f"native_pinned CSD lacks CUDA IPC async copy support: {caps}")
if expected == "egm" and not bool(caps.get("supports_egm_native_transport")):
    raise RuntimeError(f"EGM CSD lacks native EGM transport support: {caps}")
print(json.dumps({
    "backend": actual,
    "supports_cuda_ipc": bool(caps.get("supports_cuda_ipc")),
    "supports_async_copy": bool(caps.get("supports_async_copy")),
    "supports_egm_native_transport": bool(caps.get("supports_egm_native_transport")),
}, sort_keys=True))
PY
    )"
    rc="$?"
    set -e
    if [[ "${rc}" == "0" ]]; then
      echo "CSD ready: ${output}"
      return
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: 等待本地 CSD ready 超时，backend=${expected_backend}，last_error=${output}" >&2
      if [[ -f "${CSD_DIR}/csd.log" ]]; then
        sed -n '1,160p' "${CSD_DIR}/csd.log" >&2 || true
      fi
      exit 4
    fi
    sleep 2
  done
}

start_local_csd() {
  local backend="$1"
  CSD_DIR="${OUTPUT_ROOT}/csd/${RUN_ID}/node${NODE_RANK}"
  mkdir -p "${CSD_DIR}"
  case "${RACER_CSD_MODE}" in
    managed|persistent)
      ;;
    existing)
      echo "RACER_CSD_MODE=existing: 复用已启动的本地 CSD，backend=${backend}"
      if [[ "${DRY_RUN}" != "1" ]]; then
        wait_for_local_csd "${backend}"
      fi
      return
      ;;
    *)
      echo "ERROR: RACER_CSD_MODE 只能是 managed/persistent/existing，当前是 ${RACER_CSD_MODE}" >&2
      exit 4
      ;;
  esac
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: 跳过启动本地 CSD，backend=${backend}，计划目录是 ${CSD_DIR}"
    return
  fi
  if [[ "${backend}" == "native_pinned" ]]; then
    launch_csd_process python -m racer.csd \
      --host 127.0.0.1 \
      --port "${CSD_PORT}" \
      --backend native_pinned \
      --metadata-dir "${CSD_DIR}/metadata" \
      --native-pinned-total-bytes "${CSD_NATIVE_PINNED_TOTAL_BYTES}" \
      --native-pinned-segment-bytes "${CSD_NATIVE_PINNED_SEGMENT_BYTES}" \
      --native-pinned-device "${CSD_NATIVE_PINNED_DEVICE}"
  elif [[ "${backend}" == "egm" ]]; then
    EGM_ARGS=(
      --host 127.0.0.1
      --port "${CSD_PORT}"
      --backend egm
      --metadata-dir "${CSD_DIR}/metadata"
      --egm-runtime-factory "${CSD_EGM_RUNTIME_FACTORY}"
      --egm-home-device "${CSD_EGM_HOME_DEVICE}"
    )
    if [[ -n "${CSD_EGM_RUNTIME_CONFIG}" ]]; then
      EGM_ARGS+=(--egm-runtime-config "${CSD_EGM_RUNTIME_CONFIG}")
    fi
    if [[ -n "${CSD_EGM_POOL_ID}" ]]; then
      EGM_ARGS+=(--egm-pool-id "${CSD_EGM_POOL_ID}")
    fi
    if [[ -n "${CSD_EGM_OWNER_NODE}" ]]; then
      EGM_ARGS+=(--egm-owner-node "${CSD_EGM_OWNER_NODE}")
    fi
    if [[ -n "${CSD_EGM_OWNER_TRAY}" ]]; then
      EGM_ARGS+=(--egm-owner-tray "${CSD_EGM_OWNER_TRAY}")
    fi
    if [[ -n "${CSD_EGM_NUMA_ID}" ]]; then
      EGM_ARGS+=(--egm-numa-id "${CSD_EGM_NUMA_ID}")
    fi
    if [[ -n "${CSD_EGM_ACCESSING_DEVICES}" ]]; then
      EGM_ARGS+=(--egm-accessing-devices "${CSD_EGM_ACCESSING_DEVICES}")
    fi
    launch_csd_process python -m racer.csd "${EGM_ARGS[@]}"
  else
    echo "ERROR: unknown CSD backend=${backend}" >&2
    exit 4
  fi
  CSD_PID="$!"
  echo "${CSD_PID}" > "${CSD_DIR}/csd.pid"
  wait_for_local_csd "${backend}"
  if [[ "${RACER_CSD_MODE}" == "persistent" ]]; then
    disown "${CSD_PID}" >/dev/null 2>&1 || true
    CSD_PID=""
  fi
}

configure_racer_args() {
  local csd_backend="$1"
  local storage_backend="$2"
  local launch_mode="$3"
  RACER_SPARE_LAUNCH_MODE="${launch_mode}"
  if (( RACER_K + RACER_M != RACER_TRAIN_RANK_COUNT )); then
    echo "ERROR: RACER_K + RACER_M 必须等于 train rank 数。" >&2
    echo "       RACER_K=${RACER_K}, RACER_M=${RACER_M}, RACER_TRAIN_RANKS=${RACER_TRAIN_RANKS_EXPANDED}" >&2
    exit 4
  fi
  if (( RACER_SPARE_RANK_COUNT < 1 )); then
    echo "ERROR: RACER_SPARE_RANKS 至少需要一个 spare rank。" >&2
    exit 4
  fi
  if (( TORCHRUN_WORLD_SIZE != RACER_TRAIN_RANK_COUNT )); then
    cat >&2 <<MSG
ERROR: 当前 Megatron adapter 要求 torchrun world size 等于 train rank 数。

当前：
  NNODES=${NNODES}
  NPROC_PER_NODE=${NPROC_PER_NODE}
  torchrun world size=${TORCHRUN_WORLD_SIZE}
  RACER_TRAIN_RANKS=${RACER_TRAIN_RANKS_EXPANDED}

如果要测 8 train + 1 spare：
  - 同节点 spare：用 NNODES=1,NPROC_PER_NODE=8，并让 coordinator 节点可见 spare CUDA device。
  - remote spare：用两个 train 节点 NNODES=2,NPROC_PER_NODE=4，第三节点 NODE_ROLE=spare。
MSG
    exit 4
  fi

  if [[ -z "${RACER_CSD_LOCAL_RANKS}" ]]; then
    RACER_CSD_LOCAL_RANKS="$(default_local_rank_range)"
  fi
  RACER_CSD_LOCAL_RANKS_EXPANDED="$(expand_rank_list "${RACER_CSD_LOCAL_RANKS}")"
  if [[ -z "${RACER_CSD_LOCAL_COORDINATOR_RANK}" ]]; then
    RACER_CSD_LOCAL_COORDINATOR_RANK="$(first_rank "${RACER_CSD_LOCAL_RANKS_EXPANDED}")"
  fi
  export RACER_CSD_PER_NODE
  export RACER_CSD_LOCAL_RANKS
  export RACER_CSD_LOCAL_COORDINATOR_RANK

  start_local_csd "${csd_backend}"
  RACER_ARGS=(
    --racer-checkpoint
    --racer-path "${RACER_ROOT}"
    --racer-k "${RACER_K}"
    --racer-m "${RACER_M}"
    --racer-train-ranks "${RACER_TRAIN_RANKS_EXPANDED}"
    --racer-spare-ranks "${RACER_SPARE_RANKS_EXPANDED}"
    --racer-spare-launch-mode "${launch_mode}"
    --racer-buffer-size "${RACER_BUFFER_SIZE:-1073741824}"
    --racer-payload-pool-prewarm-chunks "${RACER_PAYLOAD_POOL_PREWARM_CHUNKS:-0}"
    --racer-retain-checkpoints 1
    --racer-distributed-store
    --racer-storage-backend "${storage_backend}"
    --racer-csd-host 127.0.0.1
    --racer-csd-port "${CSD_PORT}"
    --racer-csd-authkey racer-csd
    --racer-manifest-dir "${RACER_MANIFEST_DIR}"
    --racer-profile-dir "${RACER_PROFILE_DIR}"
  )
  if [[ "${launch_mode}" == "remote" ]]; then
    RACER_ARGS+=(--racer-runtime-port "${RACER_RUNTIME_PORT}")
  fi
}

run_remote_spare_worker() {
  local csd_backend="$1"
  local storage_backend="$2"
  if [[ -z "${RACER_CSD_LOCAL_RANKS}" ]]; then
    RACER_CSD_LOCAL_RANKS="${RACER_SPARE_RANKS_EXPANDED}"
  fi
  RACER_CSD_LOCAL_RANKS_EXPANDED="$(expand_rank_list "${RACER_CSD_LOCAL_RANKS}")"
  if [[ -z "${RACER_CSD_LOCAL_COORDINATOR_RANK}" ]]; then
    RACER_CSD_LOCAL_COORDINATOR_RANK="$(first_rank "${RACER_CSD_LOCAL_RANKS_EXPANDED}")"
  fi
  export RACER_CSD_PER_NODE
  export RACER_CSD_LOCAL_RANKS
  export RACER_CSD_LOCAL_COORDINATOR_RANK
  start_local_csd "${csd_backend}"

  SPARE_LOG_FILE="${LOG_ROOT}/${RUN_ID}.node${NODE_RANK}.remote_spare.log"
  echo "RUN_ID=${RUN_ID}"
  echo "MODE=${MODE}"
  echo "NODE_ROLE=${NODE_ROLE}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "RACER_RUNTIME_PORT=${RACER_RUNTIME_PORT}"
  echo "RACER_RUNTIME_WORLD_SIZE=${RACER_RUNTIME_WORLD_SIZE}"
  echo "RACER_SPARE_RANKS=${RACER_SPARE_RANKS_EXPANDED}"
  echo "RACER_REMOTE_SPARE_CUDA_DEVICE=${RACER_REMOTE_SPARE_CUDA_DEVICE}"
  echo "RACER_CSD_MODE=${RACER_CSD_MODE}"
  echo "CSD_NATIVE_PINNED_TOTAL_BYTES=${CSD_NATIVE_PINNED_TOTAL_BYTES}"
  echo "CSD_NATIVE_PINNED_SEGMENT_BYTES=${CSD_NATIVE_PINNED_SEGMENT_BYTES}"
  echo "CSD_NATIVE_PINNED_DEVICE=${CSD_NATIVE_PINNED_DEVICE}"
  echo "RACER_CSD_CHECKSUM_TYPE=${RACER_CSD_CHECKSUM_TYPE}"
  echo "RACER_CSD_MANIFEST_UPDATE_MODE=${RACER_CSD_MANIFEST_UPDATE_MODE}"
  echo "CSD_EGM_RUNTIME_FACTORY=${CSD_EGM_RUNTIME_FACTORY}"
  echo "CSD_EGM_RUNTIME_CONFIG=${CSD_EGM_RUNTIME_CONFIG}"
  echo "CSD_EGM_POOL_ID=${CSD_EGM_POOL_ID}"
  echo "CSD_EGM_OWNER_NODE=${CSD_EGM_OWNER_NODE}"
  echo "CSD_EGM_OWNER_TRAY=${CSD_EGM_OWNER_TRAY}"
  echo "CSD_EGM_HOME_DEVICE=${CSD_EGM_HOME_DEVICE}"
  echo "CSD_EGM_NUMA_ID=${CSD_EGM_NUMA_ID}"
  echo "CSD_EGM_ACCESSING_DEVICES=${CSD_EGM_ACCESSING_DEVICES}"
  echo "RACER_CSD_LOCAL_RANKS=${RACER_CSD_LOCAL_RANKS}"
  echo "RACER_CSD_LOCAL_COORDINATOR_RANK=${RACER_CSD_LOCAL_COORDINATOR_RANK}"
  echo "LOG_FILE=${SPARE_LOG_FILE}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1: 参数校验完成，跳过 remote spare worker。"
    exit 0
  fi
  python "${RACER_ROOT}/examples/racer_remote_spare_worker.py" \
    --racer-root "${RACER_ROOT}" \
    --megatron-root "${MEGATRON_ROOT}" \
    --store-host "${MASTER_ADDR}" \
    --runtime-port "${RACER_RUNTIME_PORT}" \
    --world-size "${RACER_RUNTIME_WORLD_SIZE}" \
    --spare-rank "$(first_rank "${RACER_SPARE_RANKS_EXPANDED}")" \
    --spare-cuda-device "${RACER_REMOTE_SPARE_CUDA_DEVICE}" \
    --racer-k "${RACER_K}" \
    --racer-m "${RACER_M}" \
    --racer-train-ranks "${RACER_TRAIN_RANKS_EXPANDED}" \
    --racer-spare-ranks "${RACER_SPARE_RANKS_EXPANDED}" \
    --racer-buffer-size "${RACER_BUFFER_SIZE:-1073741824}" \
    --racer-storage-backend "${storage_backend}" \
    --racer-csd-host 127.0.0.1 \
    --racer-csd-port "${CSD_PORT}" \
    --racer-csd-authkey racer-csd \
    2>&1 | tee "${SPARE_LOG_FILE}"
  exit "${PIPESTATUS[0]}"
}

case "${MODE}" in
  baseline)
    ;;
  racer_pinned_single_node)
  if [[ "${NNODES}" != "1" ]]; then
    echo "ERROR: racer_pinned_single_node 只能 NNODES=1。多节点请用 MODE=racer_pinned_per_node。" >&2
    exit 4
  fi
  configure_racer_args native_pinned csd_native_pinned local
    ;;
  racer_pinned_per_node|racer_pinned_multinode)
    configure_racer_args native_pinned csd_native_pinned "${RACER_SPARE_LAUNCH_MODE:-local}"
    ;;
  racer_pinned_remote_spare)
    if [[ "${NODE_ROLE}" == "spare" ]]; then
      run_remote_spare_worker native_pinned csd_native_pinned
    fi
    configure_racer_args native_pinned csd_native_pinned remote
    ;;
  racer_egm)
    configure_racer_args egm csd_egm "${RACER_SPARE_LAUNCH_MODE:-local}"
    ;;
  racer_egm_remote_spare)
    if [[ "${NODE_ROLE}" == "spare" ]]; then
      run_remote_spare_worker egm csd_egm
    fi
    configure_racer_args egm csd_egm remote
    ;;
  *)
    echo "ERROR: unknown MODE=${MODE}" >&2
    exit 3
    ;;
esac

cleanup() {
  if [[ -n "${CSD_PID}" ]]; then
    kill "${CSD_PID}" >/dev/null 2>&1 || true
    wait "${CSD_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "RUN_ID=${RUN_ID}"
echo "MODE=${MODE}"
echo "NODE_ROLE=${NODE_ROLE}"
echo "DRY_RUN=${DRY_RUN}"
echo "MODEL_SIZE=${MODEL_SIZE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "TORCHRUN_WORLD_SIZE=${TORCHRUN_WORLD_SIZE}"
echo "RACER_RUNTIME_PORT=${RACER_RUNTIME_PORT}"
echo "RACER_RUNTIME_WORLD_SIZE=${RACER_RUNTIME_WORLD_SIZE}"
echo "RACER_K=${RACER_K}"
echo "RACER_M=${RACER_M}"
echo "RACER_TRAIN_RANKS=${RACER_TRAIN_RANKS_EXPANDED}"
echo "RACER_SPARE_RANKS=${RACER_SPARE_RANKS_EXPANDED}"
echo "RACER_SPARE_LAUNCH_MODE=${RACER_SPARE_LAUNCH_MODE:-}"
echo "RACER_CSD_PER_NODE=${RACER_CSD_PER_NODE}"
echo "RACER_CSD_MODE=${RACER_CSD_MODE}"
echo "CSD_NATIVE_PINNED_TOTAL_BYTES=${CSD_NATIVE_PINNED_TOTAL_BYTES}"
echo "CSD_NATIVE_PINNED_SEGMENT_BYTES=${CSD_NATIVE_PINNED_SEGMENT_BYTES}"
echo "CSD_NATIVE_PINNED_DEVICE=${CSD_NATIVE_PINNED_DEVICE}"
echo "RACER_CSD_CHECKSUM_TYPE=${RACER_CSD_CHECKSUM_TYPE}"
echo "RACER_CSD_MANIFEST_UPDATE_MODE=${RACER_CSD_MANIFEST_UPDATE_MODE}"
echo "CSD_EGM_RUNTIME_FACTORY=${CSD_EGM_RUNTIME_FACTORY}"
echo "CSD_EGM_RUNTIME_CONFIG=${CSD_EGM_RUNTIME_CONFIG}"
echo "CSD_EGM_POOL_ID=${CSD_EGM_POOL_ID}"
echo "CSD_EGM_OWNER_NODE=${CSD_EGM_OWNER_NODE}"
echo "CSD_EGM_OWNER_TRAY=${CSD_EGM_OWNER_TRAY}"
echo "CSD_EGM_HOME_DEVICE=${CSD_EGM_HOME_DEVICE}"
echo "CSD_EGM_NUMA_ID=${CSD_EGM_NUMA_ID}"
echo "CSD_EGM_ACCESSING_DEVICES=${CSD_EGM_ACCESSING_DEVICES}"
echo "RACER_CSD_LOCAL_RANKS=${RACER_CSD_LOCAL_RANKS:-}"
echo "RACER_CSD_LOCAL_COORDINATOR_RANK=${RACER_CSD_LOCAL_COORDINATOR_RANK:-}"
echo "LOG_FILE=${LOG_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: 参数校验完成，跳过 torchrun。"
  exit 0
fi

cd "${MEGATRON_ROOT}"

set +e
torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${MEGATRON_ROOT}/pretrain_gpt.py" \
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --tensor-model-parallel-size "${TP_SIZE}" \
  --pipeline-model-parallel-size "${PP_SIZE}" \
  --num-layers "${NUM_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  --attention-backend auto \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --train-iters "${TRAIN_ITERS}" \
  --lr 1.5e-4 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --lr-warmup-iters 1 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --bf16 \
  --no-bias-dropout-fusion \
  --use-distributed-optimizer \
  --ckpt-format torch \
  --data-path "${DATA_PATH}" \
  --vocab-file "${GPT2_VOCAB_FILE}" \
  --merge-file "${GPT2_MERGE_FILE}" \
  --split 949,50,1 \
  --save "${CHECKPOINT_PATH}" \
  --load "${CHECKPOINT_PATH}" \
  --tensorboard-dir "${TENSORBOARD_DIR}" \
  --log-interval 1 \
  --save-interval "${SAVE_INTERVAL}" \
  --eval-interval 100000 \
  --eval-iters 2 \
  "${RACER_ARGS[@]}" \
  ${MEGATRON_EXTRA_ARGS:-} \
  2>&1 | tee "${LOG_FILE}"
rc="${PIPESTATUS[0]}"
set -e

echo "torchrun exit code: ${rc}"
exit "${rc}"
