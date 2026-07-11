#!/usr/bin/env bash
set -euo pipefail

# 多节点 RACER 重启测试 driver。
#
# 用法：在 PAI 的每个节点上运行同一个脚本，只有 NODE_RANK 不同。
# 目标默认是 2 个训练节点 * 4 GPU + 1 个 remote spare 节点：
#   k=6,m=2, train ranks=0-7, spare rank=8
#
# 这个脚本会：
#   1. 第 1 轮启动 CSD，并让 CSD 脱离训练进程组；
#   2. 每 20 个 iteration 等 RACER checkpoint 保存完成后杀训练进程；
#   3. 连续杀 3 次；
#   4. 最后再跑到 80 iteration；
#   5. 在 node0 生成 iteration_times.csv 和 summary.md。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RACER_ROOT="${RACER_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${RACER_ROOT}/.." && pwd)}"

MODE="${MODE:-racer_pinned_remote_spare}"
MODEL_SIZE="${MODEL_SIZE:-1.5b}"
BASE_RUN_ID="${BASE_RUN_ID:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/pai_runs/${MODEL_SIZE}_${MODE}_restart}"
RESTART_OVERWRITE="${RESTART_OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
MEGATRON_EXTRA_ARGS="${MEGATRON_EXTRA_ARGS:-}"
RACER_DEBUG_PAYLOAD_CHECKSUM="${RACER_DEBUG_PAYLOAD_CHECKSUM:-}"
RACER_DEBUG_STORAGE_READ_CHECKSUM="${RACER_DEBUG_STORAGE_READ_CHECKSUM:-}"
RESTART_STANDALONE="${RESTART_STANDALONE:-0}"
RACER_CSD_CLEANUP_AFTER="${RACER_CSD_CLEANUP_AFTER:-1}"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PAI_TOTAL_NODES="${PAI_TOTAL_NODES:-}"
# PAI injects MASTER_PORT for its own rendezvous; do not reuse it for Megatron by default.
MASTER_PORT_BASE="${MASTER_PORT_BASE:-${MEGATRON_MASTER_PORT:-29500}}"
RACER_RUNTIME_PORT_BASE="${RACER_RUNTIME_PORT_BASE:-${RACER_RUNTIME_PORT:-29610}}"
CSD_PORT="${CSD_PORT:-7007}"

SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
RACER_RETAIN_CHECKPOINTS="${RACER_RETAIN_CHECKPOINTS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"
KILL_INTERVAL_ITERS="${KILL_INTERVAL_ITERS:-20}"
KILL_COUNT="${KILL_COUNT:-3}"
POST_KILL_TRAIN_ITERS="${POST_KILL_TRAIN_ITERS:-20}"
FINAL_TRAIN_ITERS=$((KILL_INTERVAL_ITERS * KILL_COUNT + POST_KILL_TRAIN_ITERS))
PHASE_COUNT=$((KILL_COUNT + 1))

RACER_K="${RACER_K:-6}"
RACER_M="${RACER_M:-2}"
RACER_TRAIN_RANKS="${RACER_TRAIN_RANKS:-0-7}"
RACER_SPARE_RANKS="${RACER_SPARE_RANKS:-8}"
RACER_CSD_PROFILE_LOG="${RACER_CSD_PROFILE_LOG:-1}"
RACER_CSD_CUDA_EVENT_TIMING="${RACER_CSD_CUDA_EVENT_TIMING:-1}"
RACER_CSD_CHECKSUM_TYPE="${RACER_CSD_CHECKSUM_TYPE:-sample64}"
RACER_CSD_MANIFEST_UPDATE_MODE="${RACER_CSD_MANIFEST_UPDATE_MODE:-batch}"
RACER_DISTRIBUTED_DIRECT_STORAGE_LOAD="${RACER_DISTRIBUTED_DIRECT_STORAGE_LOAD:-auto}"
RACER_CSD_DIRECT_TENSOR_IPC="${RACER_CSD_DIRECT_TENSOR_IPC:-0}"
RACER_EGM_DIRECT_IPC="${RACER_EGM_DIRECT_IPC:-1}"
case "${MODE}" in
  racer_egm|racer_egm_remote_spare)
    if [[ "${RACER_EGM_DIRECT_IPC}" == "1" ]]; then
      RACER_CSD_DIRECT_WRITE_IPC="${RACER_CSD_DIRECT_WRITE_IPC:-1}"
      RACER_CSD_DIRECT_READ_IPC="${RACER_CSD_DIRECT_READ_IPC:-1}"
    else
      RACER_CSD_DIRECT_WRITE_IPC="${RACER_CSD_DIRECT_WRITE_IPC:-0}"
      RACER_CSD_DIRECT_READ_IPC="${RACER_CSD_DIRECT_READ_IPC:-0}"
    fi
    ;;
  *)
    RACER_CSD_DIRECT_WRITE_IPC="${RACER_CSD_DIRECT_WRITE_IPC:-0}"
    RACER_CSD_DIRECT_READ_IPC="${RACER_CSD_DIRECT_READ_IPC:-0}"
    ;;
esac
if [[ -z "${RACER_CSD_STRICT_DIRECT_IPC:-}" ]]; then
  case "${MODE}" in
    racer_egm|racer_egm_remote_spare)
      RACER_CSD_STRICT_DIRECT_IPC="${RACER_EGM_DIRECT_IPC}"
      ;;
    *) RACER_CSD_STRICT_DIRECT_IPC=0 ;;
  esac
fi
RACER_CSD_SERIALIZE_READ_IPC="${RACER_CSD_SERIALIZE_READ_IPC:-0}"
RACER_ASYNC_OFFLOAD="${RACER_ASYNC_OFFLOAD:-1}"
RACER_BUFFER_SIZE="${RACER_BUFFER_SIZE:-1073741824}"
if [[ -z "${RACER_PAYLOAD_POOL_PREWARM_CHUNKS:-}" ]]; then
  case "${MODE}" in
    racer_egm|racer_egm_remote_spare) RACER_PAYLOAD_POOL_PREWARM_CHUNKS=4 ;;
    *) RACER_PAYLOAD_POOL_PREWARM_CHUNKS=0 ;;
  esac
fi
CSD_NATIVE_PINNED_TOTAL_BYTES="${CSD_NATIVE_PINNED_TOTAL_BYTES:-}"
CSD_NATIVE_PINNED_SEGMENT_BYTES="${CSD_NATIVE_PINNED_SEGMENT_BYTES:-1073741824}"
CSD_NATIVE_PINNED_DEVICE="${CSD_NATIVE_PINNED_DEVICE:-0}"
CSD_EGM_RUNTIME_FACTORY="${CSD_EGM_RUNTIME_FACTORY:-racer.egm_runtime:create_runtime}"
CSD_EGM_RUNTIME_CONFIG="${CSD_EGM_RUNTIME_CONFIG:-}"
CSD_EGM_POOL_ID="${CSD_EGM_POOL_ID:-}"
CSD_EGM_OWNER_NODE="${CSD_EGM_OWNER_NODE:-}"
CSD_EGM_OWNER_TRAY="${CSD_EGM_OWNER_TRAY:-}"
CSD_EGM_HOME_DEVICE="${CSD_EGM_HOME_DEVICE:-0}"
CSD_EGM_NUMA_ID="${CSD_EGM_NUMA_ID:-}"
CSD_EGM_ACCESSING_DEVICES="${CSD_EGM_ACCESSING_DEVICES:-}"
CSD_EGM_TOTAL_BYTES="${CSD_EGM_TOTAL_BYTES:-}"
CSD_EGM_SEGMENT_BYTES="${CSD_EGM_SEGMENT_BYTES:-}"
CSD_EGM_MAX_POOL_BYTES="${CSD_EGM_MAX_POOL_BYTES:-}"
RACER_EXTENSION_TRUST_READY="${RACER_EXTENSION_TRUST_READY:-1}"

HOSTFILE="${HOSTFILE:-/mnt/workspace/pai_nodes.txt}"
PHASE_TIMEOUT_SECONDS="${PHASE_TIMEOUT_SECONDS:-7200}"
MARKER_POLL_SECONDS="${MARKER_POLL_SECONDS:-2}"

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

node_role() {
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

terminate_group() {
  local pid="$1"
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    return
  fi
  kill -TERM "-${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  kill -KILL "-${pid}" >/dev/null 2>&1 || true
}

wait_for_cluster_marker() {
  local marker="$1"
  local timeout="$2"
  local start now
  start="$(date +%s)"
  while [[ ! -e "${marker}" ]]; do
    now="$(date +%s)"
    if (( now - start > timeout )); then
      echo "ERROR: 等待 marker 超时: ${marker}" >&2
      exit 5
    fi
    sleep "${MARKER_POLL_SECONDS}"
  done
}

wait_for_done_files() {
  local phase="$1"
  local expected="$2"
  local timeout="$3"
  local start now count
  start="$(date +%s)"
  while true; do
    count="$(find "${STATE_DIR}" -maxdepth 1 -name "${phase}.node*.done" 2>/dev/null | wc -l)"
    if (( count >= expected )); then
      return
    fi
    now="$(date +%s)"
    if (( now - start > timeout )); then
      echo "ERROR: phase ${phase} 等待节点完成超时，当前 ${count}/${expected}" >&2
      exit 6
    fi
    sleep "${MARKER_POLL_SECONDS}"
  done
}

wait_for_save_then_kill() {
  local phase="$1"
  local target_iter="$2"
  local pid="$3"
  local log_path="$4"
  local marker="$5"
  local start now done_pattern
  if [[ "${RACER_ASYNC_OFFLOAD}" == "1" ]]; then
    done_pattern="RACER async checkpoint committed: iteration=${target_iter}"
  else
    done_pattern="RACER save blocking time: iteration=${target_iter}"
  fi
  start="$(date +%s)"
  while true; do
    if grep -q "${done_pattern}" "${log_path}" 2>/dev/null; then
      date -u +"%Y-%m-%dT%H:%M:%SZ" > "${marker}"
      terminate_group "${pid}"
      return
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "ERROR: phase ${phase} 在 iteration ${target_iter} 保存完成前退出，日志: ${log_path}" >&2
      exit 7
    fi
    now="$(date +%s)"
    if (( now - start > PHASE_TIMEOUT_SECONDS )); then
      echo "ERROR: phase ${phase} 等待 iteration ${target_iter} 保存完成超时，日志: ${log_path}" >&2
      terminate_group "${pid}"
      exit 8
    fi
    sleep "${MARKER_POLL_SECONDS}"
  done
}

summarize_cluster_logs() {
  python - "${LOG_ROOT}" "${BASE_RUN_ID}" "${SAVE_INTERVAL}" "${STATE_DIR}" <<'PY'
import csv
import re
import statistics
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
base_run_id = sys.argv[2]
save_interval = int(sys.argv[3])
state_dir = Path(sys.argv[4])

iter_re = re.compile(r"iteration\s+([0-9]+)\s*/\s*([0-9]+).*elapsed time per iteration \(ms\):\s*([0-9.]+)")
store_re = re.compile(r"RACER distributed tensor-tree checkpoint stored: tag=([^,]+), store=([0-9.]+) ms")
load_re = re.compile(r"RACER distributed memory checkpoint loaded: tag=([^,]+), total=([0-9.]+) ms")

rows_by_key = {}
stores_by_key = {}
loads_by_key = {}
for path in sorted(log_root.glob(f"{base_run_id}_phase*.driver.node*.log")):
    phase = path.stem.split(".")[0].replace(base_run_id + "_", "")
    node = path.stem.rsplit("node", 1)[-1]
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = iter_re.search(line)
        if m:
            iteration = int(m.group(1))
            rows_by_key[(phase, iteration)] = {
                "phase": phase,
                "node": node,
                "iteration": iteration,
                "train_iters": int(m.group(2)),
                "elapsed_time_per_iteration_ms": float(m.group(3)),
                "checkpoint_iteration": iteration % save_interval == 0,
            }
        m = store_re.search(line)
        if m:
            stores_by_key[(phase, m.group(1))] = {"phase": phase, "tag": m.group(1), "store_ms": float(m.group(2))}
        m = load_re.search(line)
        if m:
            loads_by_key[(phase, m.group(1))] = {"phase": phase, "tag": m.group(1), "load_total_ms": float(m.group(2))}

def stats(items):
    if not items:
        return {"count": 0, "avg": "", "p50": "", "p95": ""}
    ordered = sorted(items)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "count": len(items),
        "avg": f"{statistics.mean(items):.3f}",
        "p50": f"{statistics.median(items):.3f}",
        "p95": f"{ordered[p95_index]:.3f}",
    }

state_dir.mkdir(parents=True, exist_ok=True)
rows = [rows_by_key[key] for key in sorted(rows_by_key)]
stores = [stores_by_key[key] for key in sorted(stores_by_key)]
loads = [loads_by_key[key] for key in sorted(loads_by_key)]
with (state_dir / "iteration_times.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["phase", "node", "iteration", "train_iters", "elapsed_time_per_iteration_ms", "checkpoint_iteration"])
    writer.writeheader()
    writer.writerows(rows)

normal = [r["elapsed_time_per_iteration_ms"] for r in rows if not r["checkpoint_iteration"]]
all_iters = [r["elapsed_time_per_iteration_ms"] for r in rows]
unique_iterations = sorted({int(r["iteration"]) for r in rows})
unique_checkpoint_iterations = [iteration for iteration in unique_iterations if iteration % save_interval == 0]
unique_normal_iterations = [iteration for iteration in unique_iterations if iteration % save_interval != 0]
first_iteration_by_phase = {}
for row in rows:
    phase = row["phase"]
    iteration = int(row["iteration"])
    first_iteration_by_phase[phase] = min(iteration, first_iteration_by_phase.get(phase, iteration))
steady_normal = [
    r["elapsed_time_per_iteration_ms"]
    for r in rows
    if not r["checkpoint_iteration"] and int(r["iteration"]) > first_iteration_by_phase[r["phase"]] + 1
]
normal_stats = stats(normal)
steady_normal_stats = stats(steady_normal)
all_stats = stats(all_iters)

summary = [
    "# GB200 多节点 RACER 重启测试摘要",
    "",
    f"- 原始日志目录: `{log_root}`",
    f"- iteration 明细: `{state_dir / 'iteration_times.csv'}`",
    f"- 唯一训练 iteration 数: `{len(unique_iterations)}`",
    f"- 唯一非 checkpoint iteration 数: `{len(unique_normal_iterations)}`",
    f"- 唯一 checkpoint iteration 数: `{len(unique_checkpoint_iterations)}`",
    f"- iteration 时间日志样本数: `{len(rows)}`",
    f"- 非 checkpoint 日志样本数: `{normal_stats['count']}`",
    f"- 非 checkpoint 样本平均耗时: `{normal_stats['avg']} ms`",
    f"- 非 checkpoint 样本 p50/p95: `{normal_stats['p50']} / {normal_stats['p95']} ms`",
    f"- 稳态非 checkpoint 样本数: `{steady_normal_stats['count']}`",
    f"- 稳态训练每 iteration 平均耗时: `{steady_normal_stats['avg']} ms`",
    f"- 稳态训练 p50/p95: `{steady_normal_stats['p50']} / {steady_normal_stats['p95']} ms`",
    f"- 全部日志样本平均耗时: `{all_stats['avg']} ms`",
    "- 说明: 日志样本数可能大于唯一 iteration 数；kill marker 从 node0 传播前，其他节点可能多打印少量边界 iteration。",
    f"- 观察到 RACER store 次数: `{len(stores)}`",
    f"- 观察到 RACER restart load 次数: `{len(loads)}`",
]
(state_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
PY
}

SUMMARY_WRITTEN=0

write_summary_once() {
  local status="$1"
  local reason="${2:-}"
  if (( NODE_RANK != 0 )); then
    return
  fi
  if [[ "${SUMMARY_WRITTEN}" == "1" ]]; then
    return
  fi
  SUMMARY_WRITTEN=1
  set +e
  summarize_cluster_logs
  local rc="$?"
  if [[ "${rc}" != "0" ]]; then
    mkdir -p "${STATE_DIR}"
    {
      echo "# GB200 多节点 RACER 重启测试摘要"
      echo
      echo "- 摘要生成失败: \`summarize_cluster_logs exit ${rc}\`"
      echo "- 原始日志目录: \`${LOG_ROOT}\`"
    } > "${STATE_DIR}/summary.md"
  fi
  if [[ -n "${reason}" ]]; then
    {
      echo "- 状态: \`${status}\`"
      echo "- 退出原因: \`${reason}\`"
    } >> "${STATE_DIR}/summary.md"
  else
    echo "- 状态: \`${status}\`" >> "${STATE_DIR}/summary.md"
  fi
  set -e
}

write_failure_summary_on_exit() {
  local rc="$?"
  if [[ "${rc}" == "0" ]]; then
    return
  fi
  write_summary_once "未完成" "driver exit ${rc}"
}

NODE_RANK="$(detect_node_rank)"
MASTER_ADDR="$(resolve_master_addr)"
NODE_ROLE="$(node_role)"

if [[ -z "${CSD_EGM_TOTAL_BYTES}" ]]; then
  case "${MODE}" in
    racer_egm|racer_egm_remote_spare)
      if [[ "${NODE_ROLE}" == "spare" ]]; then
        CSD_EGM_TOTAL_BYTES=0
      else
        case "${MODEL_SIZE}" in
          1.5b)
            CSD_EGM_TOTAL_BYTES=103079215104
            ;;
          5.3b)
            CSD_EGM_TOTAL_BYTES=274877906944
            ;;
          *)
            echo "ERROR: EGM 预分配无法识别 MODEL_SIZE=${MODEL_SIZE}" >&2
            exit 2
            ;;
        esac
      fi
      ;;
  esac
fi
if [[ -z "${CSD_EGM_SEGMENT_BYTES}" ]]; then
  CSD_EGM_SEGMENT_BYTES="${CSD_NATIVE_PINNED_SEGMENT_BYTES}"
fi
if [[ -z "${CSD_EGM_MAX_POOL_BYTES}" ]]; then
  case "${MODE}" in
    racer_egm|racer_egm_remote_spare)
      CSD_EGM_MAX_POOL_BYTES="${CSD_EGM_TOTAL_BYTES}"
      ;;
  esac
fi

if [[ -z "${BASE_RUN_ID}" ]]; then
  echo "ERROR: 请显式设置 BASE_RUN_ID，三个节点必须完全一致，例如 BASE_RUN_ID=gb200_pinned_1_5b_001。" >&2
  exit 2
fi

if [[ -z "${PAI_TOTAL_NODES}" ]]; then
  case "${MODE}" in
    racer_pinned_remote_spare|racer_egm_remote_spare)
      PAI_TOTAL_NODES=$((NNODES + 1))
      ;;
    *)
      PAI_TOTAL_NODES="${NNODES}"
      ;;
  esac
fi

STATE_DIR="${RESTART_STATE_DIR:-${OUTPUT_ROOT}/restart_state/${BASE_RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_ROOT}/checkpoints/${BASE_RUN_ID}}"
RACER_MANIFEST_DIR="${RACER_MANIFEST_DIR:-${OUTPUT_ROOT}/racer_manifests/${BASE_RUN_ID}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard/${BASE_RUN_ID}}"

mkdir -p "${LOG_ROOT}" "${CHECKPOINT_PATH}" "${RACER_MANIFEST_DIR}" "${TENSORBOARD_DIR}"
trap write_failure_summary_on_exit EXIT

if (( NODE_RANK == 0 )) || [[ "${RESTART_STANDALONE}" == "1" ]]; then
  if [[ -e "${STATE_DIR}/ready" && "${RESTART_OVERWRITE}" != "1" ]]; then
    echo "ERROR: ${STATE_DIR} 已存在。换 BASE_RUN_ID，或设置 RESTART_OVERWRITE=1。" >&2
    exit 2
  fi
  if [[ "${RESTART_OVERWRITE}" == "1" ]]; then
    rm -rf "${STATE_DIR}"
  fi
  mkdir -p "${STATE_DIR}"
  cat > "${STATE_DIR}/config.txt" <<EOF
BASE_RUN_ID=${BASE_RUN_ID}
MODE=${MODE}
MODEL_SIZE=${MODEL_SIZE}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
MEGATRON_EXTRA_ARGS=${MEGATRON_EXTRA_ARGS}
RACER_DEBUG_PAYLOAD_CHECKSUM=${RACER_DEBUG_PAYLOAD_CHECKSUM}
RACER_DEBUG_STORAGE_READ_CHECKSUM=${RACER_DEBUG_STORAGE_READ_CHECKSUM}
MASTER_ADDR=${MASTER_ADDR}
MASTER_PORT_BASE=${MASTER_PORT_BASE}
RACER_RUNTIME_PORT_BASE=${RACER_RUNTIME_PORT_BASE}
CSD_PORT=${CSD_PORT}
NNODES=${NNODES}
NPROC_PER_NODE=${NPROC_PER_NODE}
PAI_TOTAL_NODES=${PAI_TOTAL_NODES}
SAVE_INTERVAL=${SAVE_INTERVAL}
RACER_RETAIN_CHECKPOINTS=${RACER_RETAIN_CHECKPOINTS}
KILL_INTERVAL_ITERS=${KILL_INTERVAL_ITERS}
KILL_COUNT=${KILL_COUNT}
POST_KILL_TRAIN_ITERS=${POST_KILL_TRAIN_ITERS}
FINAL_TRAIN_ITERS=${FINAL_TRAIN_ITERS}
RACER_K=${RACER_K}
RACER_M=${RACER_M}
RACER_TRAIN_RANKS=${RACER_TRAIN_RANKS}
RACER_SPARE_RANKS=${RACER_SPARE_RANKS}
RACER_CSD_PROFILE_LOG=${RACER_CSD_PROFILE_LOG}
RACER_CSD_CUDA_EVENT_TIMING=${RACER_CSD_CUDA_EVENT_TIMING}
RACER_CSD_CHECKSUM_TYPE=${RACER_CSD_CHECKSUM_TYPE}
RACER_CSD_MANIFEST_UPDATE_MODE=${RACER_CSD_MANIFEST_UPDATE_MODE}
RACER_DISTRIBUTED_DIRECT_STORAGE_LOAD=${RACER_DISTRIBUTED_DIRECT_STORAGE_LOAD}
RACER_CSD_DIRECT_TENSOR_IPC=${RACER_CSD_DIRECT_TENSOR_IPC}
RACER_CSD_DIRECT_WRITE_IPC=${RACER_CSD_DIRECT_WRITE_IPC}
RACER_CSD_DIRECT_READ_IPC=${RACER_CSD_DIRECT_READ_IPC}
RACER_CSD_STRICT_DIRECT_IPC=${RACER_CSD_STRICT_DIRECT_IPC}
RACER_CSD_SERIALIZE_READ_IPC=${RACER_CSD_SERIALIZE_READ_IPC}
RACER_EGM_DIRECT_IPC=${RACER_EGM_DIRECT_IPC}
RACER_ASYNC_OFFLOAD=${RACER_ASYNC_OFFLOAD}
RACER_BUFFER_SIZE=${RACER_BUFFER_SIZE}
RACER_PAYLOAD_POOL_PREWARM_CHUNKS=${RACER_PAYLOAD_POOL_PREWARM_CHUNKS}
CSD_NATIVE_PINNED_TOTAL_BYTES=${CSD_NATIVE_PINNED_TOTAL_BYTES}
CSD_NATIVE_PINNED_SEGMENT_BYTES=${CSD_NATIVE_PINNED_SEGMENT_BYTES}
CSD_NATIVE_PINNED_DEVICE=${CSD_NATIVE_PINNED_DEVICE}
CSD_EGM_RUNTIME_FACTORY=${CSD_EGM_RUNTIME_FACTORY}
CSD_EGM_RUNTIME_CONFIG=${CSD_EGM_RUNTIME_CONFIG}
CSD_EGM_POOL_ID=${CSD_EGM_POOL_ID}
CSD_EGM_OWNER_NODE=${CSD_EGM_OWNER_NODE}
CSD_EGM_OWNER_TRAY=${CSD_EGM_OWNER_TRAY}
CSD_EGM_HOME_DEVICE=${CSD_EGM_HOME_DEVICE}
CSD_EGM_NUMA_ID=${CSD_EGM_NUMA_ID}
CSD_EGM_ACCESSING_DEVICES=${CSD_EGM_ACCESSING_DEVICES}
CSD_EGM_TOTAL_BYTES=${CSD_EGM_TOTAL_BYTES}
CSD_EGM_SEGMENT_BYTES=${CSD_EGM_SEGMENT_BYTES}
CSD_EGM_MAX_POOL_BYTES=${CSD_EGM_MAX_POOL_BYTES}
RACER_EXTENSION_TRUST_READY=${RACER_EXTENSION_TRUST_READY}
EOF
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STATE_DIR}/ready"
else
  wait_for_cluster_marker "${STATE_DIR}/ready" "${PHASE_TIMEOUT_SECONDS}"
fi

echo "BASE_RUN_ID=${BASE_RUN_ID}"
echo "NODE_RANK=${NODE_RANK}"
echo "NODE_ROLE=${NODE_ROLE}"
echo "MODE=${MODE}"
echo "MODEL_SIZE=${MODEL_SIZE}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "DRY_RUN=${DRY_RUN}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT_BASE=${MASTER_PORT_BASE}"
echo "RACER_RUNTIME_PORT_BASE=${RACER_RUNTIME_PORT_BASE}"
echo "CSD_PORT=${CSD_PORT}"
echo "FINAL_TRAIN_ITERS=${FINAL_TRAIN_ITERS}"
echo "STATE_DIR=${STATE_DIR}"

for phase in $(seq 0 $((PHASE_COUNT - 1))); do
  phase_name="$(printf 'phase%02d' "${phase}")"
  phase_run_id="${BASE_RUN_ID}_${phase_name}"
  phase_log="${LOG_ROOT}/${phase_run_id}.driver.node${NODE_RANK}.log"
  phase_done="${STATE_DIR}/${phase_name}.node${NODE_RANK}.done"
  kill_marker="${STATE_DIR}/${phase_name}.kill"
  master_port=$((MASTER_PORT_BASE + phase))
  runtime_port=$((RACER_RUNTIME_PORT_BASE + phase))
  if (( phase == 0 )); then
    csd_mode=persistent
  else
    csd_mode=existing
  fi

  echo "启动 ${phase_name}，日志 ${phase_log}"
  setsid env \
    MODE="${MODE}" \
    MODEL_SIZE="${MODEL_SIZE}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    MEGATRON_EXTRA_ARGS="${MEGATRON_EXTRA_ARGS}" \
    RACER_DEBUG_PAYLOAD_CHECKSUM="${RACER_DEBUG_PAYLOAD_CHECKSUM}" \
    RACER_DEBUG_STORAGE_READ_CHECKSUM="${RACER_DEBUG_STORAGE_READ_CHECKSUM}" \
    NODE_RANK="${NODE_RANK}" \
    NNODES="${NNODES}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    MASTER_ADDR="${MASTER_ADDR}" \
    MASTER_PORT="${master_port}" \
    RACER_RUNTIME_PORT="${runtime_port}" \
    TRAIN_ITERS="${FINAL_TRAIN_ITERS}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    RUN_ID="${phase_run_id}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    TENSORBOARD_DIR="${TENSORBOARD_DIR}/${phase_name}" \
    RACER_MANIFEST_DIR="${RACER_MANIFEST_DIR}" \
    RACER_PROFILE_DIR="${OUTPUT_ROOT}/racer_profiles/${BASE_RUN_ID}/${phase_name}" \
    RACER_CSD_MODE="${csd_mode}" \
    RACER_K="${RACER_K}" \
    RACER_M="${RACER_M}" \
    RACER_TRAIN_RANKS="${RACER_TRAIN_RANKS}" \
    RACER_SPARE_RANKS="${RACER_SPARE_RANKS}" \
    RACER_CSD_PROFILE_LOG="${RACER_CSD_PROFILE_LOG}" \
    RACER_CSD_CUDA_EVENT_TIMING="${RACER_CSD_CUDA_EVENT_TIMING}" \
    RACER_CSD_CHECKSUM_TYPE="${RACER_CSD_CHECKSUM_TYPE}" \
    RACER_CSD_MANIFEST_UPDATE_MODE="${RACER_CSD_MANIFEST_UPDATE_MODE}" \
    RACER_DISTRIBUTED_DIRECT_STORAGE_LOAD="${RACER_DISTRIBUTED_DIRECT_STORAGE_LOAD}" \
    RACER_CSD_DIRECT_TENSOR_IPC="${RACER_CSD_DIRECT_TENSOR_IPC}" \
    RACER_CSD_DIRECT_WRITE_IPC="${RACER_CSD_DIRECT_WRITE_IPC}" \
    RACER_CSD_DIRECT_READ_IPC="${RACER_CSD_DIRECT_READ_IPC}" \
    RACER_CSD_STRICT_DIRECT_IPC="${RACER_CSD_STRICT_DIRECT_IPC}" \
    RACER_CSD_SERIALIZE_READ_IPC="${RACER_CSD_SERIALIZE_READ_IPC}" \
    RACER_EGM_DIRECT_IPC="${RACER_EGM_DIRECT_IPC}" \
    RACER_ASYNC_OFFLOAD="${RACER_ASYNC_OFFLOAD}" \
    RACER_BUFFER_SIZE="${RACER_BUFFER_SIZE}" \
    RACER_PAYLOAD_POOL_PREWARM_CHUNKS="${RACER_PAYLOAD_POOL_PREWARM_CHUNKS}" \
    RACER_RETAIN_CHECKPOINTS="${RACER_RETAIN_CHECKPOINTS}" \
    CSD_NATIVE_PINNED_TOTAL_BYTES="${CSD_NATIVE_PINNED_TOTAL_BYTES}" \
    CSD_NATIVE_PINNED_SEGMENT_BYTES="${CSD_NATIVE_PINNED_SEGMENT_BYTES}" \
    CSD_NATIVE_PINNED_DEVICE="${CSD_NATIVE_PINNED_DEVICE}" \
    CSD_EGM_RUNTIME_FACTORY="${CSD_EGM_RUNTIME_FACTORY}" \
    CSD_EGM_RUNTIME_CONFIG="${CSD_EGM_RUNTIME_CONFIG}" \
    CSD_EGM_POOL_ID="${CSD_EGM_POOL_ID}" \
    CSD_EGM_OWNER_NODE="${CSD_EGM_OWNER_NODE}" \
    CSD_EGM_OWNER_TRAY="${CSD_EGM_OWNER_TRAY}" \
    CSD_EGM_HOME_DEVICE="${CSD_EGM_HOME_DEVICE}" \
    CSD_EGM_NUMA_ID="${CSD_EGM_NUMA_ID}" \
    CSD_EGM_ACCESSING_DEVICES="${CSD_EGM_ACCESSING_DEVICES}" \
    CSD_EGM_TOTAL_BYTES="${CSD_EGM_TOTAL_BYTES}" \
    CSD_EGM_SEGMENT_BYTES="${CSD_EGM_SEGMENT_BYTES}" \
    CSD_EGM_MAX_POOL_BYTES="${CSD_EGM_MAX_POOL_BYTES}" \
    RACER_EXTENSION_TRUST_READY="${RACER_EXTENSION_TRUST_READY}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${SCRIPT_DIR}/pai_run_megatron_multinode.sh" \
    > "${phase_log}" 2>&1 &
  child_pid="$!"

  if (( phase < KILL_COUNT )); then
    target_iter=$((KILL_INTERVAL_ITERS * (phase + 1)))
    if [[ "${DRY_RUN}" == "1" ]]; then
      if (( NODE_RANK == 0 )) || [[ "${RESTART_STANDALONE}" == "1" ]]; then
        date -u +"%Y-%m-%dT%H:%M:%SZ" > "${kill_marker}"
      fi
      wait "${child_pid}" >/dev/null 2>&1 || true
    elif (( NODE_RANK == 0 )); then
      wait_for_save_then_kill "${phase_name}" "${target_iter}" "${child_pid}" "${phase_log}" "${kill_marker}"
    else
      while kill -0 "${child_pid}" >/dev/null 2>&1; do
        if [[ -e "${kill_marker}" ]]; then
          terminate_group "${child_pid}"
          break
        fi
        sleep "${MARKER_POLL_SECONDS}"
      done
    fi
    wait "${child_pid}" >/dev/null 2>&1 || true
  else
    set +e
    wait "${child_pid}"
    rc="$?"
    set -e
    if [[ "${rc}" != "0" ]]; then
      echo "ERROR: final phase 退出码 ${rc}，日志 ${phase_log}" >&2
      exit "${rc}"
    fi
  fi

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${phase_done}"
  wait_for_done_files "${phase_name}" "${PAI_TOTAL_NODES}" "${PHASE_TIMEOUT_SECONDS}"
done

if [[ "${RACER_CSD_CLEANUP_AFTER}" == "1" && "${DRY_RUN}" != "1" ]]; then
  csd_pid_file="${OUTPUT_ROOT}/csd/${BASE_RUN_ID}_phase00/node${NODE_RANK}/csd.pid"
  if [[ -f "${csd_pid_file}" ]]; then
    csd_pid="$(cat "${csd_pid_file}")"
    if [[ -n "${csd_pid}" ]]; then
      terminate_group "${csd_pid}"
    fi
  fi
fi

if (( NODE_RANK == 0 )); then
  write_summary_once "完成"
  echo "summary: ${STATE_DIR}/summary.md"
  echo "iteration_times: ${STATE_DIR}/iteration_times.csv"
fi
