# GB200 多节点 RACER 测试步骤

目标形态：3 个 PAI Worker。node0/node1 各跑 4 个 train rank，node2 跑 1 个 remote spare rank。RACER 配置为 `k=6,m=2,train=0-7,spare=8`。每个节点各有一个本地 CSD。

注意：这里 `NNODES=2` 表示 torchrun 的训练节点数，不是 PAI 总节点数。`NODE_RANK=0/1` 会自动作为 train 节点，`NODE_RANK=2` 会自动作为 spare 节点。

`OUTPUT_ROOT` 必须放在三个节点都能看到的共享目录里。restart driver 用 `OUTPUT_ROOT/restart_state/<BASE_RUN_ID>` 下的 marker 文件协调杀进程和进入下一轮；如果这个目录不是共享的，多节点会互相等不到。

remote spare 模式下，spare rank 必须紧跟 train rank 连续编号。8 个 train rank 时只能填 `RACER_SPARE_RANKS=8`；如果以后改成 4 个 train rank，就填 `RACER_SPARE_RANKS=4`。

## 1. 先启动 PAI 容器

PAI 启动命令先填：

```bash
bash -lc 'mkdir -p /mnt/data/luohaonan/workspace/pai_bootstrap; env | sort > /mnt/data/luohaonan/workspace/pai_bootstrap/env.$(hostname).txt; hostname -I > /mnt/data/luohaonan/workspace/pai_bootstrap/ip.$(hostname).txt; sleep infinity'
```

进入每个节点后确认：

```bash
nvidia-smi -L
ls /mnt/data/luohaonan/workspace/racer
ls /mnt/data/luohaonan/workspace/Megatron-LM-FT
```

`MASTER_ADDR` 用 node0 的内网 IP。

## 2. 正式测 pinned memory

三台节点都运行 `examples/pai_run_megatron_restart_driver.sh`。它会自动执行：

- 每 `5` 个 iteration 保存一次；
- 每 `20` 个 iteration 杀一次训练进程；
- 一共杀 `3` 次；
- 最后一次恢复后再跑到 `80` iteration；
- node0 汇总正常训练每 iteration 时间。

脚本默认使用 `RACER_CSD_CHECKSUM_TYPE=sample64` 和 `RACER_CSD_MANIFEST_UPDATE_MODE=batch`，用于避免大 checkpoint 默认走全量 `sha256` 和逐 chunk SQLite seal。需要强校验时再显式改成 `RACER_CSD_CHECKSUM_TYPE=sha256`。

`CSD_NATIVE_PINNED_TOTAL_BYTES` 默认会按模型和节点角色设置：1.5B train 节点约 `96 GiB`，5.3B train 节点约 `256 GiB`，remote spare 独占节点默认 `0`，避免第三节点无意义预分配 pinned memory。内存不够或想预热更大的池时可以手动覆盖。

每次正式测试都换一个新的 `BASE_RUN_ID`。如果复用同一个 `BASE_RUN_ID`，三个节点都保留 `RESTART_OVERWRITE=1`，并先启动 node0。

node0:

```bash
cd /mnt/data/luohaonan/workspace/racer
MODE=racer_pinned_remote_spare MODEL_SIZE=1.5b BASE_RUN_ID=gb200_pinned_1_5b_001 \
  RESTART_OVERWRITE=1 NODE_RANK=0 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=<MASTER_IP> \
  OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/gb200_pinned_1_5b_001 \
  bash examples/pai_run_megatron_restart_driver.sh
```
```bash
cd /mnt/data/luohaonan/workspace/racer
MODE=racer_pinned_remote_spare MODEL_SIZE=1.5b BASE_RUN_ID=gb200_pinned_1_5b_001 \
  RESTART_OVERWRITE=1 NODE_RANK=0 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=10.101.226.155 \
  OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/gb200_pinned_1_5b_001 \
  bash examples/pai_run_megatron_restart_driver.sh
```

node1:

```bash
cd /mnt/data/luohaonan/workspace/racer
MODE=racer_pinned_remote_spare MODEL_SIZE=1.5b BASE_RUN_ID=gb200_pinned_1_5b_001 \
  RESTART_OVERWRITE=1 NODE_RANK=1 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=<MASTER_IP> \
  OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/gb200_pinned_1_5b_001 \
  bash examples/pai_run_megatron_restart_driver.sh
```
```bash
cd /mnt/data/luohaonan/workspace/racer
MODE=racer_pinned_remote_spare MODEL_SIZE=1.5b BASE_RUN_ID=gb200_pinned_1_5b_001 \
  RESTART_OVERWRITE=1 NODE_RANK=1 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=10.101.226.155 \
  OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/gb200_pinned_1_5b_001 \
  bash examples/pai_run_megatron_restart_driver.sh
```

node2:

```bash
cd /mnt/data/luohaonan/workspace/racer
MODE=racer_pinned_remote_spare MODEL_SIZE=1.5b BASE_RUN_ID=gb200_pinned_1_5b_001 \
  RESTART_OVERWRITE=1 NODE_RANK=2 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=<MASTER_IP> \
  OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/gb200_pinned_1_5b_001 \
  bash examples/pai_run_megatron_restart_driver.sh
```
```bash
cd /mnt/data/luohaonan/workspace/racer
MODE=racer_pinned_remote_spare MODEL_SIZE=1.5b BASE_RUN_ID=gb200_pinned_1_5b_001 \
  RESTART_OVERWRITE=1 NODE_RANK=2 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=10.101.226.155 \
  OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/gb200_pinned_1_5b_001 \
  bash examples/pai_run_megatron_restart_driver.sh
```

5.3B 只改 `MODEL_SIZE`、`BASE_RUN_ID`、`OUTPUT_ROOT`：

```bash
MODEL_SIZE=5.3b BASE_RUN_ID=gb200_pinned_5_3b_001 OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/gb200_pinned_5_3b_001
```

## 3. 正式测 EGM

默认使用 repo 内置 runtime factory：`racer.egm_runtime:create_runtime`。先在每个节点确认能 import：

```bash
PYTHONPATH=/mnt/data/luohaonan/workspace/racer:/mnt/data/luohaonan/workspace/Megatron-LM-FT:$PYTHONPATH \
python -c 'import racer.egm_runtime as r; print(r.create_runtime)'
```

再做 EGM runtime preflight：

```bash
PYTHONPATH=/mnt/data/luohaonan/workspace/racer:/mnt/data/luohaonan/workspace/Megatron-LM-FT:$PYTHONPATH \
python - <<'PY'
from racer.egm_runtime import create_runtime
runtime = create_runtime(total_bytes=0)
print(runtime.capabilities())
PY
```

如果这里报 `CSD_EGM_NUMA_ID is required`，说明容器里不能自动查询 GPU 对应的 host NUMA，需要在每个节点设置该节点本地的 NUMA id。三台节点不要求相同，以实际机器拓扑为准：

```bash
CSD_EGM_NUMA_ID=<NUMA_ID>
```

如果需要限制 EGM pool 的 home device 或访问设备，可以加：

```bash
CSD_EGM_HOME_DEVICE=0
CSD_EGM_ACCESSING_DEVICES=0,1,2,3
```

preflight 通过后，把 pinned 命令里的 `MODE` 改成 `racer_egm_remote_spare`。

脚本启动 CSD 后会先检查 daemon capabilities；如果 EGM mempool、NUMA 或 backend 不匹配，训练开始前就会失败。这个失败不是 fallback，说明 EGM 后端没有真正起来。

restart driver 会把 `CSD_EGM_*`、`CSD_NATIVE_PINNED_*`、`RACER_BUFFER_SIZE` 等关键参数写入 `restart_state/<BASE_RUN_ID>/config.txt`，每个 phase 的日志也会打印 resolved 配置。

如果你们已有更专门的 EGM runtime，可以覆盖默认 factory：

```bash
CSD_EGM_RUNTIME_FACTORY=your_module:create_runtime
```

这个 factory 需要返回一个 runtime 对象，至少实现：

```python
write_from_cuda_ipc(tag: str, chunk_id: str, view: dict, metadata: dict) -> dict | str
read_to_cuda_ipc(tag: str, chunk_id: str, view: dict) -> dict | str
```

可选实现：

```python
wait(op_id)
poll(op_id)
metadata(tag, chunk_id)
list_chunks(tag)
checksum(tag, chunk_id)
capabilities()
```

如果 EGM mempool 创建失败，`MODE=racer_egm_remote_spare` 会直接失败，不会 fallback 到 pinned memory。

## 4. 结果位置

node0 完成后看：

```bash
cat /mnt/data/luohaonan/workspace/pai_runs/gb200_pinned_1_5b_001/restart_state/gb200_pinned_1_5b_001/summary.md
```

关键文件：

```text
restart_state/<BASE_RUN_ID>/summary.md
restart_state/<BASE_RUN_ID>/iteration_times.csv
logs/<BASE_RUN_ID>_phase*.node*.log
csd/<BASE_RUN_ID>_phase00/node*/csd.log
```

成功时应看到：

```text
RACER_RUNTIME_WORLD_SIZE=9
node0: RACER_CSD_LOCAL_RANKS=0-3
node1: RACER_CSD_LOCAL_RANKS=4-7
node2: RACER_CSD_LOCAL_RANKS=8
```

并且 `summary.md` 里 `观察到 RACER restart load 次数` 至少是 `3`。

## 5. 只做 smoke test

如果只是先确认多节点能启动，不杀进程，用：

```bash
MODE=racer_pinned_remote_spare NODE_RANK=0 NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=<MASTER_IP> \
  MODEL_SIZE=1.5b TRAIN_ITERS=20 SAVE_INTERVAL=5 \
  RACER_K=6 RACER_M=2 RACER_TRAIN_RANKS=0-7 RACER_SPARE_RANKS=8 \
  OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/smoke_pinned_1_5b \
  bash examples/pai_run_megatron_multinode.sh
```

node1/node2 同理只改 `NODE_RANK`。

## 6. spare 和 train 在同一台机器

如果要测单机 5 卡形态，例如 4 个 train rank + 1 个 local spare：

```bash
cd /mnt/data/luohaonan/workspace/racer
MODE=racer_pinned_single_node NODE_RANK=0 NNODES=1 NPROC_PER_NODE=4 MASTER_ADDR=127.0.0.1 \
  MODEL_SIZE=1.5b TRAIN_ITERS=80 SAVE_INTERVAL=5 \
  RACER_K=3 RACER_M=1 RACER_TRAIN_RANKS=0-3 RACER_SPARE_RANKS=4 \
  OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/local_spare_1_5b \
  bash examples/pai_run_megatron_multinode.sh
```

这里 `RACER_SPARE_RANKS=4` 表示 coordinator 节点可见的 CUDA device id。remote spare 模式下它才表示外部 RACER rank。
