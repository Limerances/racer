#!/usr/bin/env bash
set -euo pipefail

# 本地模拟两个 torchrun 节点。
#
# 默认用 CPU/gloo，所以没有 GPU 也能验证 rendezvous 和 rank 编号。
# 如果要试 NCCL，可以设置：
#   BACKEND=nccl CUDA_VISIBLE_DEVICES_NODE0=0,1 CUDA_VISIBLE_DEVICES_NODE1=2,3 bash examples/local_torchrun_multinode_smoke.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29617}"
BACKEND="${BACKEND:-gloo}"
LOG_ROOT="${LOG_ROOT:-/tmp/racer_local_torchrun_smoke}"

if [[ "${NNODES}" != "2" ]]; then
  echo "ERROR: 这个 smoke 脚本只模拟两个节点。要测更多节点，复制下面两个 torchrun 块即可。" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"
rm -f "${LOG_ROOT}/node0.log" "${LOG_ROOT}/node1.log"

run_node() {
  local node_rank="$1"
  local log_file="$2"
  local cvd_var="CUDA_VISIBLE_DEVICES_NODE${node_rank}"
  local cvd="${!cvd_var:-}"

  if [[ -n "${cvd}" ]]; then
    CUDA_VISIBLE_DEVICES="${cvd}" BACKEND="${BACKEND}" torchrun \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --nnodes "${NNODES}" \
      --node_rank "${node_rank}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${SCRIPT_DIR}/local_torchrun_multinode_smoke.py" \
      > "${log_file}" 2>&1
  else
    BACKEND="${BACKEND}" torchrun \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --nnodes "${NNODES}" \
      --node_rank "${node_rank}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${SCRIPT_DIR}/local_torchrun_multinode_smoke.py" \
      > "${log_file}" 2>&1
  fi
}

echo "启动本地 torchrun 多节点 smoke test"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "BACKEND=${BACKEND}"
echo "LOG_ROOT=${LOG_ROOT}"

run_node 0 "${LOG_ROOT}/node0.log" &
pid0="$!"

sleep 2

run_node 1 "${LOG_ROOT}/node1.log" &
pid1="$!"

rc=0
wait "${pid0}" || rc="$?"
wait "${pid1}" || rc="$?"

echo
echo "===== node0.log ====="
cat "${LOG_ROOT}/node0.log"
echo
echo "===== node1.log ====="
cat "${LOG_ROOT}/node1.log"

exit "${rc}"
