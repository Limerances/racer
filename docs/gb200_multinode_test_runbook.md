# GB200 三节点 RACER EGM 测试手册

本文档描述当前论文主路径：两台训练节点各运行 4 个 Megatron rank，第三台节点运行 1 个 remote spare rank，每台节点各有一个常驻 CSD。本文的标准验收配置为：

```text
model=1.5B
global_batch_size=16
train_ranks=0-7
spare_ranks=8
k=5, m=3
```

因此 `k + m = 8 = n_train`；spare rank 8 不属于编码矩阵。脚本默认值仍是 `k=6,m=2`，所有正式命令必须显式传 `RACER_K=5 RACER_M=3`，并以 `restart_state/<run-id>/config.txt` 为准确认实际参数。

当前节点映射：

| IP | `NODE_RANK` | 角色 |
| --- | ---: | --- |
| `10.101.226.175` | 0 | train ranks 0-3 / master |
| `10.101.226.154` | 1 | train ranks 4-7 |
| `10.101.226.178` | 2 | remote spare rank 8 |

这里 `NNODES=2` 是 torchrun 的训练节点数，不是物理节点总数。

## 1. 启动前检查

三个节点都应能看到共享目录：

```bash
ls /mnt/data/luohaonan/workspace/racer
ls /mnt/data/luohaonan/workspace/Megatron-LM-FT
nvidia-smi -L
```

清理旧进程并确认 GPU 空闲：

```bash
pgrep -af 'racer\.csd|torchrun|pretrain_gpt|racer_remote_spare_worker'

pkill -TERM -f 'python -m racer.csd'
pkill -TERM -f 'racer_remote_spare_worker.py'
pkill -TERM -f 'pretrain_gpt.py'
pkill -TERM -f 'torchrun'
```

不要无条件清理不属于本次实验的进程。正式实验使用一组唯一端口，并尽量保持在常见 ephemeral port 起点 32768 以下；历史运行曾因 34012 被占用而失败。

## 2. EGM preflight

在每个节点运行：

```bash
cd /mnt/data/luohaonan/workspace/racer
PYTHONPATH=/mnt/data/luohaonan/workspace/racer:/mnt/data/luohaonan/workspace/Megatron-LM-FT:$PYTHONPATH \
python -c 'from racer.egm_runtime import create_runtime; r=create_runtime(total_bytes=0); print(r.capabilities())'
```

内置 runtime 当前是真实 CUDA Host-NUMA mempool 后端，不是 fake runtime。默认会为每个可见 CUDA device 建立 topology-aware EGM pool，并打印：

```text
egm_runtime=cuda_mempool_host_numa
egm_topology=per_device_host_numa
topology_aware=true
pool_count=4
device_numa_map={0:0,1:0,2:1,3:1}  # 以实际机器为准
```

`direct IPC` 表示 CSD 从 producer CUDA IPC view 直接复制到 EGM allocation，绕过中间 CUDA staging slab；它仍包含 GPU→EGM 或 EGM→GPU copy，不代表训练进程直接零拷贝访问 EGM。

如果自动查询 NUMA 失败，`CSD_EGM_NUMA_ID` 只覆盖 `CSD_EGM_HOME_DEVICE`，不会伪造其他 device 的 NUMA。可以将 `CSD_EGM_ACCESSING_DEVICES` 限制为能够确定拓扑的设备，或提供支持 per-device NUMA 配置的自定义 runtime；不要把一个 scalar NUMA id 当成所有 GPU 的映射。

## 3. 标准四阶段性能/重启实验

为每次实验选择全新的、能表达参数和实现版本的 ID，例如：

```text
egm_1p5b_gbs16_k5m3_s1_postfix_<git-sha>_<date>_<sequence>
```

建议端口组：

```text
MASTER_PORT_BASE=30700        # phase00-03: 30700-30703
RACER_RUNTIME_PORT_BASE=30810 # phase00-03: 30810-30813
CSD_PORT=7397
```

在三台节点分别执行同一个命令，只修改 `NODE_RANK`：

```bash
cd /mnt/data/luohaonan/workspace/racer

MODE=racer_egm_remote_spare \
MODEL_SIZE=1.5b \
GLOBAL_BATCH_SIZE=16 \
BASE_RUN_ID=<RUN_ID> \
OUTPUT_ROOT=/mnt/data/luohaonan/workspace/pai_runs/<RUN_ID> \
RESTART_OVERWRITE=1 \
MASTER_ADDR=10.101.226.175 \
MASTER_PORT_BASE=30700 \
RACER_RUNTIME_PORT_BASE=30810 \
CSD_PORT=7397 \
NODE_RANK=<0|1|2> \
NNODES=2 \
NPROC_PER_NODE=4 \
RACER_K=5 \
RACER_M=3 \
RACER_TRAIN_RANKS=0-7 \
RACER_SPARE_RANKS=8 \
RACER_EGM_DIRECT_IPC=1 \
RACER_CSD_DIRECT_WRITE_IPC=1 \
RACER_CSD_DIRECT_READ_IPC=1 \
RACER_CSD_STRICT_DIRECT_IPC=1 \
RACER_CSD_CUDA_EVENT_TIMING=1 \
RACER_ASYNC_OFFLOAD=1 \
RACER_PAYLOAD_POOL_PREWARM_CHUNKS=4 \
bash examples/pai_run_megatron_restart_driver.sh
```

先启动 node0，随后立即启动 node1 和 node2。`OUTPUT_ROOT` 必须是共享目录；driver 使用其中的 marker 协调三个节点。

四个阶段的语义：

1. phase00：新建 persistent CSD，训练到 iteration 20，等待 checkpoint 20 异步 commit 后 kill 训练进程。
2. phase01：保留 CSD/EGM，恢复 iteration 20，训练到 40，commit 后再次 kill。
3. phase02：恢复 iteration 40，训练到 60，commit 后再次 kill。
4. phase03：恢复 iteration 60，训练到 80，正常退出。

每 5 个 iteration 保存一次，共应有 16 次 store、3 次 restart load。故障 kill 后出现 `failed to recv` 或 `TCPStore closed` 通常是预期退出噪声；它们不能单独作为失败依据。

## 4. 硬成功判据

结果目录：

```text
restart_state/<RUN_ID>/config.txt
restart_state/<RUN_ID>/summary.md
restart_state/<RUN_ID>/iteration_times.csv
logs/<RUN_ID>_phase*.driver.node*.log
logs/<RUN_ID>_phase*.node*.log
csd/<RUN_ID>_phase00/node*/csd.log
racer_profiles/<RUN_ID>/phase*/rank_*.jsonl
racer_manifests/<RUN_ID>/
```

必须同时满足：

- `summary.md` 状态为“完成”；
- unique iteration 为 80；store 为 16；restart load 为 3；
- `phase00/01/02.kill` 共 3 个，四阶段三节点 `.done` 共 12 个；
- 最终在 storage-bearing node0/node1 CSD 均存在 iteration 80 的 metadata-only generation marker，
  并存在 rank 0-7 的 CSD tensor-tree manifest；共享目录中的对应文件是原子缓存，
  正常运行应存在，但其单独缺失不代表 CSD generation 未提交；
- `config.txt` 明确记录 GBS16、k5、m3、spare8 及三组端口；
- node0/node1 CSD 是 topology-aware EGM，node2 没有正式 checkpoint PUT/GET；
- strict direct PUT 均为 `direct_ipc_enabled=true`、`staging_kind=none`，没有 `direct_ipc_error`；
- 不出现 `EADDRINUSE`、CSD `ConnectionRefusedError`、反序列化错误、CUDA error 或 strict-direct failure。

连续 3 分钟没有新 iteration、store/commit 或 marker 时应主动检查，而不是等待默认 7200 秒超时。

iteration 性能必须按 checkpoint 时间线分类，不能简单使用
`checkpoint_iteration=false`：

- Megatron 的 `elapsed time per iteration` 在调用 checkpoint 之前输出，所以触发保存的 iteration 本身仍是纯训练耗时；save blocking 需要单独统计。
- 从 `RACER async checkpoint scheduled` 到 `RACER async checkpoint committed` 期间与训练相交的 iteration 是 `async_checkpoint_overlap`，它会受到 EC、CUDA IPC 和 EGM 流量影响，不能计入普通训练基线。
- restart/load 后的第一轮是 `restart_first` 冷启动样本，应与普通训练和 async-overlap 分开。
- 新版 driver 会在 `iteration_times.csv` 中输出 `sample_class`、`async_checkpoint_overlap` 和 `clean_ordinary_iteration`。

## 5. 性能比较

同配置的即时未修改基线与本轮接受结果：

```text
pai_runs/egm_1p5b_gbs16_k5m3_s1_baseline_67f3003_20260711_0637
pai_runs/egm_1p5b_gbs16_k5m3_s1_finalreview_20260711_04
```

后一个 run 已完整通过 80 iterations / 16 stores / 3 restart loads。后续同机型
初筛以接受结果为参考：

| 指标 | 未修改基线 | 接受结果 | 后续单次疑似回归线（相对接受结果 5%） |
| --- | ---: | ---: | ---: |
| 稳态非 checkpoint iteration mean | 696.567 ms | 646.767 ms | > 679.1 ms |
| checkpoint caller timer mean | 205.076 ms | 193.959 ms | > 203.7 ms |
| store makespan mean | 442.691 ms | 444.802 ms | > 467.0 ms |
| logical store BW | 60.094 GB/s | 60.642 GB/s | < 57.61 GB/s |
| encoded store BW | 112.146 GB/s | 113.169 GB/s | < 107.51 GB/s |
| async commit wall mean | 1025.061 ms | 863.091 ms | > 906.2 ms |

`racer_distributed_store_wall_ms_max`、logical/encoded store BW 保留为纯 child-data
路径的同比指标。新实现还必须报告 `racer_checkpoint_commit_wall_ms_max`、
`racer_checkpoint_metadata_publish_ms_max` 和 logical/encoded commit BW；这些字段
包含 CSD tree manifest、所有 per-node generation marker 及 collective，不能用旧
store makespan 代替。旧参考 run 没有该新字段，因此第一轮用已有的 async commit
wall 与 checkpoint caller timer 约束真实开销，后续以新字段建立同口径基线。

5% 只用于单轮初筛。最终结论应重复运行并比较中位数，同时核对 store 的 payload bytes、encoded bytes、direct IPC 计数完全一致。`racer_egm_copy_bandwidth_gbps` 使用 CUDA event 累计时长，是介质 copy 的诊断指标，不应当作端到端吞吐。

最终代码 run 的 store p95 为 665.729 ms，来自两次真实 CSD 尾延迟；它与前一
干净 run 的 663.796 ms 基本一致。store 均值仅比即时基线高 0.48%，caller p95、
训练 p95 和 async-commit p95 均优于基线，因此不把单次后台 store 尾样本解释为
热路径回归。

## 6. 显式纠删恢复实验

标准 restart driver 只证明“训练进程重启后仍能看到 CSD-owned EGM checkpoint”；无故障 fast path 会直接读取系统型 data row，不能单独证明 m=3 解码。

在 correctness 专用 run 中加入：

```bash
MEGATRON_EXTRA_ARGS='--racer-recover-ranks 0,1,2 --racer-replacement-mapping 0:0,1:1,2:2'
RACER_DEBUG_PAYLOAD_CHECKSUM=sha256
RACER_DEBUG_STORAGE_READ_CHECKSUM=1
RACER_RETAIN_CHECKPOINTS=2
```

这会把 codeword rows 0、1、2 视为同时丢失，并用剩余 5 行触发 GPU decode；self replacement mapping 让恢复结果回到当前测试进程。日志中应出现非零 `load_decode_request_count` / `load_route_decode_ms`，每个 rank 的 payload SHA-256 必须通过，并且 Megatron 至少继续训练两步、确认 optimizer step 后的 loss/grad 仍为有限值。保留两个 checkpoint 可避免一次失败恢复产生的新保存点立刻剪掉最后一个已知健康 generation。

store 会把每个 reduction group 补齐到 manifest 中记录的对齐
`group_nbytes`。decode 的所有 survivor NCCL send/recv 必须使用这一精确宽度，不能
用该组最大的 `valid_nbytes`；只有 decode 完成后才能按目标 rank 的
`packet_nbytes_by_rank` 截短。尾组长度不一致不会可靠地触发 NCCL 错误，却会静默
损坏主要位于 checkpoint 尾部的 distributed optimizer state。

本轮通过的最大擦除结果位于：

```text
pai_runs/egm_1p5b_gbs16_k5m3_s1_erasure3_sha256_20260711_02
```

该 run 为 10 iterations / 2 stores / 1 restart load，实际触发 24 次 decode；52 个
CSD storage reads 的 sample64 校验和 8 ranks 共 28 个 payload chunks 的 SHA-256
均通过，iter 6～10 的 `number of nan iterations` 均为 0。

这是最大 `m=3` 的逻辑 erasure 注入，验证矩阵、survivor 选择、GPU decode 和 payload 重建。它仍不等同于物理下线整台节点：真实 node-loss 还要求弹性重建训练 world、替换进程和跨节点 CSD survivor discovery；当前实现尚未完整自动化这部分。

## 7. Pinned 对照与快速 smoke

测 pinned 对照时只修改：

```text
MODE=racer_pinned_remote_spare
BASE_RUN_ID=<包含 pinned/k5m3/gbs16 的新 ID>
OUTPUT_ROOT=<对应新目录>
```

只验证启动、不做 kill 时，可直接运行 `pai_run_megatron_multinode.sh`，但仍必须显式传 GBS16、k5、m3、train0-7、spare8。不要复用正式结果目录或端口。
