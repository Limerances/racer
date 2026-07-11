# RACER 接入 Megatron 说明

这份文档给后续接 Megatron 的 agent 看，只保留目标、边界和接入步骤。

## 1. 项目目标

RACER 不是普通 Reed-Solomon 库，也不是把 Jerasure 接进 Megatron。它的目标是：在大模型 checkpoint 保存和恢复时，让 spare GPU 承担 GF(2^8) 编码、校验和恢复计算。

核心语义：

- `k + m == len(train_ranks)` 必须成立。
- `spare_ranks` 不进入编码矩阵。
- spare GPU 只做计算和恢复承接，不是 data/parity owner。
- 大 buffer 的 GF 计算必须走 RACER CUDA kernel。
- Jerasure 只做 CPU baseline 和正确性对比，不是 fallback。

## 2. 创新点

一句话：RACER 是一个 spare-GPU-assisted checkpoint erasure coding 原型，在保持 train rank 的 RS/Cauchy 语义不变的前提下，把编码和恢复中的 GF 计算卸载到 spare GPU。

具体创新：

- spare GPU 不改变编码布局，只提供计算资源。
- checkpoint store/load 的 GF 计算从 CPU/Jerasure 转到 CUDA kernel。
- 用 `data_group` 和 `reduction_group` 组织 train-rank payload，spare rank 始终在布局外。
- failed rank 的恢复结果可以直接落到 spare GPU。
- Jerasure 保留为对比路线，用来验证 RACER CUDA 结果。

## 3. 当前接口

单进程接口：

```python
import racer

ctx = racer.init(k=3, m=1, train_ranks=[0, 1, 2, 3], spare_ranks=[4])
handle = racer.store(rank_states, tag="iter_100", context=ctx)
handle.wait()
recovered = racer.load(tag="iter_100", failed_train_ranks=[0], context=ctx)
```

`store` 支持两类输入：

- `dict[int, torch.uint8 Tensor]`：每个 train rank 一个 CUDA byte payload。
- `dict[int, dict[str, Tensor]]`：每个 train rank 一个 state_dict，内部会 flatten 成 byte payload。

注意：输入 key 只能是 train rank；spare rank 不能出现在 checkpoint 输入里。单进程 `RacerContext` 默认 rank id 等于 CUDA device id。

## 4. data_group 和 reduction_group

`data_group` 是编码矩阵里的数据列，编号 `0 .. k-1`。

`reduction_group` 是一次一起编码/恢复的一组 train-rank payload，最多包含 `k` 个真实数据块。

如果 `len(train_ranks)` 不能被 `k` 整除，会补 `virtual_zero`，但不会把 spare rank 塞进去。

例子：

```text
k=3, m=1
train_ranks=[0,1,2,3]
spare_ranks=[4]

reduction_group 0: rank 0, rank 1, rank 2
reduction_group 1: rank 3, zero, zero
```

spare rank 4 不属于任何 data_group/reduction_group，只负责计算。

## 5. store/load 数据流

store：校验输入 rank；state_dict 输入先 flatten 成 CUDA `uint8` payload；按 reduction_group 分组；data row 保存到 train-rank owner；chunk 拷到 spare GPU staging buffer；调用 `codec_cuda.apply_matrix_cuda` 生成 parity；parity 回写到 train-rank parity owner；写 manifest 和 chunk storage。

load：读取 manifest/chunks；把 failed rank 映射到 reduction_group 和 data_group；选择 survivor rows；小矩阵求逆在 CPU 上做；survivor chunks 拷到 spare GPU；调用 CUDA GF kernel 恢复目标 data column；failed rank 的恢复 payload 默认放到第一个 spare GPU；如果原输入是 state_dict，再 unflatten 回 tensor。

CPU 只能做小矩阵和元数据计算，大 checkpoint buffer 不能走 CPU/Jerasure。

## 6. CUDA 与 Jerasure 边界

CUDA runtime 关键文件：

- `racer/codec_cuda.py`
- `racer/csrc/racer_cuda.cu`
- `racer/csrc/binding.cpp`

关键函数：`apply_matrix_cuda`、`encode_blocks`、`decode_blocks`、`gf256_mul`、`gf256_mul_xor`。

Jerasure 相关文件：`racer/jerasure.py`、`examples/bench_jerasure_baseline.py`。

接 Megatron 时必须保持：runtime 不调用 Jerasure；Jerasure 只用于 baseline、测试对齐和实验对比；不新增 CPU fallback。

## 7. 接 Megatron 从哪里开始

优先看：

- `racer/distributed.py`
- `examples/bench_distributed_5gpu.py`

不要优先从单进程 `RacerContext` 接。单进程路径更适合本地验证和 benchmark。

当前 distributed prototype：train rank 传本地 CUDA `uint8` payload，spare rank 传 `None`，NCCL send/recv 把数据送到 spare，spare 调 CUDA kernel 算 parity/recovery，再把 parity 或 recovered payload 发回对应 train rank。

## 8. 建议的 Megatron adapter

建议先做薄 adapter，不要改 RACER 核心逻辑：

```python
class RacerMegatronCheckpointManager:
    def save(tag, local_state_dict): ...
    def load(tag, failed_train_ranks): ...
```

第一版流程：

1. 在 Megatron checkpoint save 边界拿到每个 rank 的 local state_dict。
2. train rank flatten 成 CUDA `uint8` payload，spare rank 传 `None`。
3. 所有参与 rank 调 `racer.distributed.distributed_store`。
4. adapter 将 data/parity chunks 和 child manifest 两阶段提交到 per-node CSD；
   metadata-only tensor-tree/top marker 通过 CSD 单 RPC、单 SQLite transaction
   原子提交，shared-file 副本只作索引/缓存。
5. load 时读 manifest/chunks，调用 distributed load。
6. 恢复出的 byte payload 按 metadata unflatten 回 Megatron state_dict。

当前实现已经采用 tensor-tree chunk packing 和 train-side pinned arena，不再把
整个 rank checkpoint 拼成单个常驻 CUDA payload。async store-many 会按 child
chunk 数保留 CUDA reload/receive slots；显存随 inflight child tags 增长，后续
需要可配置 window，而不是回退到整 rank flatten。

## 9. 最容易踩错的点

- 不要把 spare rank 加进 `k + m`。
- 不要让 spare rank 成为 checkpoint 输入 owner。
- 不要把 Jerasure 接进 runtime。
- 不要新增 CPU fallback。
- 不要把 `data_group/reduction_group` 改名成别的概念。
- 不要默认 Megatron global rank 等于 CUDA device id。
- 不要用 CPU/Jerasure benchmark 代表 RACER runtime 性能。
- 不要以为 `buffer_size` 能解决 state_dict flatten 的整包显存问题。
- 不要用 valid payload 最大值作为 decode survivor 的 NCCL count；send/recv 必须
  使用 manifest 的对齐 `group_nbytes`，decode 后才能截短。

## 10. 当前实现状态与剩余边界

已经完成：

- per-node CSD-owned native pinned / topology-aware EGM；
- strict direct CUDA IPC 与两阶段 child-tag commit；
- Megatron physical rank 到独立 RACER process-group rank 的映射；
- tensor-tree metadata、chunk packing、pinned arena 和 async store-many；
- 两训练节点加 remote spare 的多节点路径；
- CSD 保持存活时的训练进程 restart/load。
- manifest v3 的 CSD-authoritative tensor-tree manifest 与 storage-bearing
  train-node generation marker；同 generation 重试幂等，shared
  文件只作原子兼容缓存；
- metadata-only tree/top marker 使用 `commit_metadata` 单 RPC 原子提交；新 client
  连接旧 daemon 时回退 legacy begin/put/commit，不改变 data-bearing child 协议；
- 每个 per-node CSD coordinator 在 load 前验证 child tag 的 committed、
  daemon-owned 和完整 residency，并 collective 一致失败。

仍缺少：

- 物理 failed rank 缺席后的 elastic Megatron world rebuild 和真正 replacement process 接管；
- distributed repair；
- multi-spare 负载均衡（执行路径当前固定使用第一个 spare）；
- TrainingLocal / Hybrid planner 和基于真实 topology 的 cost model；
- store abort、失败 generation 的确定性资源回收；
- async child-tag 显存窗口上限。

第三方 EGM runtime 还必须自己保证同 key payload 的 copy-before-publish；wrapper
只能延迟 metadata/list 可见性，不能隔离一个 custom runtime 已经提前暴露给 read
路径的底层 bytes。内置 Host-NUMA runtime 满足该语义。

当前 repair 只具备单 chunk copy-before-publish/reader lease；不要把它描述成
多 chunk repair transaction 原子性。`RacerContext.repair` 与 distributed repair
仍缺 generation rollback 和跨节点冗余确认。

注意：标准 restart driver 的 3 次 load 是无 failed-row fast load，不能单独证明
纠删码恢复。EC correctness 必须额外显式注入 1～m 个 failed codeword rows。
最大擦除测试还应打开 payload SHA-256，并在恢复后的 optimizer step 之后至少再跑
一轮；只看到 load 成功或第一轮 forward loss 正常不足以排除 optimizer tail 损坏。

## 11. 验证入口

```bash
pytest -q
python examples/verify_5gpu.py

torchrun --nproc_per_node=5 examples/bench_distributed_5gpu.py \
  --train-ranks 0,1,2,3 --spare-ranks 4 --k 3 --m 1 --sizes 1M --verify

python examples/bench_gpt2_checkpoint.py \
  --profile megatron-test-tp4 --train-ranks 0,1,2,3 --spare-ranks 4 \
  --k 3 --m 1 --iters 1 --warmup 0 --verify
```
