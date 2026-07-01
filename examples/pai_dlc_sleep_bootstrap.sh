#!/usr/bin/env bash
set -euo pipefail

# PAI DLC 启动页用的占位脚本。
#
# 用法：
# 1. 如果启动页能选择代码目录，就把本仓库挂进去，然后启动命令填：
#      bash /mnt/workspace/racer/examples/pai_dlc_sleep_bootstrap.sh
# 2. 如果启动页暂时没有代码，就先在启动命令里填：
#      bash -lc 'mkdir -p /mnt/workspace/pai_bootstrap; env | sort > /mnt/workspace/pai_bootstrap/env.$(hostname).txt; hostname -I > /mnt/workspace/pai_bootstrap/ip.$(hostname).txt; sleep infinity'
#    进入节点后再把仓库放到 /mnt/workspace/racer 和 /mnt/workspace/Megatron-LM-FT。
#
# 这个脚本会在每个 Worker 上执行一次。它不启动训练，只负责让容器保持运行，
# 方便你逐个登录节点、检查环境、再运行 RACER 多节点 restart driver。

BOOT_ROOT="${BOOT_ROOT:-/mnt/workspace/pai_bootstrap}"
KEEPALIVE_SECONDS="${KEEPALIVE_SECONDS:-0}"

mkdir -p "${BOOT_ROOT}"

HOSTNAME_VALUE="$(hostname)"
NOW="$(date -u +%Y%m%d_%H%M%S)"

{
  echo "timestamp_utc=${NOW}"
  echo "hostname=${HOSTNAME_VALUE}"
  echo "hostname_f=$(hostname -f 2>/dev/null || true)"
  echo "ip=$(hostname -I 2>/dev/null || true)"
  echo "pwd=$(pwd)"
  echo "user=$(id 2>/dev/null || true)"
  echo
  echo "重要环境变量："
  env | sort | grep -E '^(MASTER|NODE|RANK|WORLD|LOCAL|PAI|DLC|WORKER|POD|HOST|CUDA|NVIDIA|NCCL|TORCH)_' || true
} | tee "${BOOT_ROOT}/node.${HOSTNAME_VALUE}.${NOW}.txt"

env | sort > "${BOOT_ROOT}/env.${HOSTNAME_VALUE}.${NOW}.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L > "${BOOT_ROOT}/gpu.${HOSTNAME_VALUE}.${NOW}.txt" || true
fi

cat <<'MSG'

PAI Worker 已启动并保持运行。

下一步：
1. 登录每个节点，确认 /mnt/workspace 是否是所有节点共享目录。
2. 把 racer 放到 /mnt/workspace/racer，把 Megatron-LM-FT 放到 /mnt/workspace/Megatron-LM-FT。
3. 在 /mnt/workspace/pai_nodes.txt 写入节点地址，第一行必须是 0 号节点，例如：
     10.0.0.1
     10.0.0.2
     10.0.0.3
4. 正式 RACER pinned restart 测试。node0/node1 是 train，node2 是 remote spare；每个节点各起一个本地 CSD。
   三个节点都运行同一个 driver，只改 NODE_RANK：

   node0：
     cd /mnt/workspace/racer
     MODE=racer_pinned_remote_spare MODEL_SIZE=1.5b BASE_RUN_ID=gb200_pinned_1_5b_001 \
       RESTART_OVERWRITE=1 NODE_RANK=0 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=10.0.0.1 \
       OUTPUT_ROOT=/mnt/workspace/pai_runs/gb200_pinned_1_5b_001 \
       bash examples/pai_run_megatron_restart_driver.sh

   node1 只改 NODE_RANK=1：
     cd /mnt/workspace/racer
     MODE=racer_pinned_remote_spare MODEL_SIZE=1.5b BASE_RUN_ID=gb200_pinned_1_5b_001 \
       RESTART_OVERWRITE=1 NODE_RANK=1 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=10.0.0.1 \
       OUTPUT_ROOT=/mnt/workspace/pai_runs/gb200_pinned_1_5b_001 \
       bash examples/pai_run_megatron_restart_driver.sh

   node2 只改 NODE_RANK=2：
     cd /mnt/workspace/racer
     MODE=racer_pinned_remote_spare MODEL_SIZE=1.5b BASE_RUN_ID=gb200_pinned_1_5b_001 \
       RESTART_OVERWRITE=1 NODE_RANK=2 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=10.0.0.1 \
       OUTPUT_ROOT=/mnt/workspace/pai_runs/gb200_pinned_1_5b_001 \
       bash examples/pai_run_megatron_restart_driver.sh

   driver 会自动：每 5 个 iteration 保存一次，每 20 个 iteration 杀一次，一共杀 3 次，最后跑到 80。

5. 5.3B 测试只改 MODEL_SIZE、BASE_RUN_ID、OUTPUT_ROOT：
     MODEL_SIZE=5.3b BASE_RUN_ID=gb200_pinned_5_3b_001 OUTPUT_ROOT=/mnt/workspace/pai_runs/gb200_pinned_5_3b_001

6. EGM 后端使用同样三节点 driver 命令，把 MODE 改成 racer_egm_remote_spare。
   默认 factory 是 racer.egm_runtime:create_runtime。
   如果 CUDA Driver 不能自动检测 GPU 对应 host NUMA，额外设置：
     CSD_EGM_NUMA_ID=<NUMA_ID>
   如果你们已有更专门的 EGM runtime，可以覆盖：
     CSD_EGM_RUNTIME_FACTORY=your_module:create_runtime

7. node0 完成后查看：
     cat /mnt/workspace/pai_runs/gb200_pinned_1_5b_001/restart_state/gb200_pinned_1_5b_001/summary.md

注意：
- remote spare 模式下 RACER_SPARE_RANKS=8 表示外部 spare worker 的 RACER rank。
- local spare 模式下 RACER_SPARE_RANKS 才表示 coordinator 节点可见的 spare CUDA device id。
- NNODES=2 表示 torchrun 训练节点数，不是 PAI 总节点数。

MSG

if [[ "${KEEPALIVE_SECONDS}" == "0" ]]; then
  sleep infinity
else
  sleep "${KEEPALIVE_SECONDS}"
fi
