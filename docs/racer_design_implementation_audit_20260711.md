# RACER 设计—实现审计（2026-07-11）

审计对象：

```text
/mnt/data/luohaonan/workspace/doc/RACER_设计文档_v0.2.md
/mnt/data/luohaonan/workspace/racer                 (base HEAD 67f3003)
/mnt/data/luohaonan/workspace/Megatron-LM-FT        (base HEAD 73dfe2814)
```

目标实验：1.5B、GBS16、`k=5,m=3`、8 train ranks、1 remote spare、三台 GB200、CSD-owned EGM。

## 1. 总体结论

RACER 已经不是早期单进程 codec prototype。当前主路径已经具备：systematic
Cauchy RS、GPU GF(2^8)、virtual-zero、tensor-tree chunking、train-side pinned
arena、remote spare GPU、per-node CSD、两阶段提交、topology-aware CUDA
Host-NUMA EGM、strict direct CUDA IPC、async store-many 和跨训练进程 restart
load。

当前最重要的边界是：标准 restart 实验验证的是“CSD/EGM 在训练进程退出后仍
可见”，不是“物理 train node 缺席后自动 replacement”。核心库能做逻辑 row
erasure decode，但 elastic Megatron world rebuild、distributed repair 和跨节点
CSD survivor directory 尚未闭环。

## 2. 本轮决策

| 分类 | 项目 | 处理 |
| --- | --- | --- |
| 当前实现优于旧文档 | per-device/per-NUMA EGM runtime、strict direct IPC | 保留实现；更新设计文档 13.2/13.4，明确 direct IPC 不是 zero-copy。 |
| 当前实现优于旧文档 | tensor-tree + async store-many | 保留实现；用 `(2+C+Cq)*chunk_size` 保守模型替换固定 `4*chunk_size`。 |
| 当前实现优于旧文档 | CUDA bitmatrix 路径已存在 | 文档从“后续实现”改为“实验路径已实现，table kernel 仍为主路径”。 |
| 实现弱于设计且设计合理 | replacement payload 取值 | 修复 adapter 按 replacement target 反查 original source key，并拒绝多 source→同 target。 |
| 实现弱于设计且设计合理 | repair overwrite 可见性 | 修复单 chunk async copy 完成前过早发布和旧 native location 的 reader race；并发 reader 使用 lease 后再回收 retired location。多 chunk repair 仍不具备 transaction rollback。 |
| 实现弱于设计且设计合理 | short/empty payload slot 复用 | 恢复原本被死分支绕过的 caller-owned zero/group slot 复用；不再写出任意 narrow view 边界。 |
| 旧实现/描述都不够精确 | EGM capability metadata | `supports_zero_copy_region` 改为 runtime 显式 opt-in，内置 runtime 为 false。 |
| 旧实现/描述都不够精确 | CSD CLI topology | home device、NUMA、accessing devices 同时传给 allocator runtime 与 wrapper。 |
| 旧实现/描述都不够精确 | `data_resident` | 从“存在任意一个 backend chunk”改为“全部 expected published/sealed chunk 均 resident”，包括无 SQLite 路径。 |
| 旧实现/描述都不够精确 | routing max cost | 按同一 rank 的 compute+xor 求 max，避免拼接两个不同 rank 的最大值。 |
| 旧实现/描述都不够精确 | 顶层 generation commit | 增加每 CSD metadata-only commit record；tree/top shared files 降为原子缓存，load 在各 per-node coordinator 一致验证 residency。 |
| 旧实现/描述都不够精确 | metadata 重试/可用性 | manifest v3 严格验证同代本地 CSD tree/top/children；同 generation 按完整 payload 幂等；响应丢失只标为 uncertain-owned；selection/transport RPC 错误 fail-closed，不能回退陈旧 v2。 |
| 性能回归驱动的协议改进 | metadata-only commit | 首轮强一致实现使 async commit mean 超基线 8.0%；改为 CSD 单 RPC、SQLite 显式单事务、exact-payload 幂等提交，旧 daemon 自动回退 legacy 协议。 |
| 旧实现/描述都不够精确 | PUT/delete 失败生命周期 | 注册前失败完整回收 CUDA/IPC/allocation；delete drain pending PUT，阻止已删除 tag 被晚完成 op 重新发布。 |
| 旧实现/描述都不够精确 | async PUT poll | dispatch future 未产生 backend op id 时非阻塞返回 RUNNING；terminal poll 与 wait 共用 completion、清理 daemon op 和 client IPC view。 |
| 旧实现/描述都不够精确 | EGM eager visibility | wrapper metadata/list 只在 op 成功后发布并与 runtime residency 取交集；设计文档补充 custom runtime 的底层 payload copy-before-publish 契约。 |
| 旧实现/描述都不够精确 | batch seal retry | SQLite seal 批次中途失败时，将失败项与未处理尾项重新入队；commit 保持不可见，重试可自愈且不重放成功前缀。 |
| 测试/文档漂移 | driver 默认 k6m2、端口未记录 | 正式命令显式 k5m3；config 记录 master/runtime/CSD ports；重写 GB200 runbook。 |

## 3. 实现覆盖

| 设计项 | 状态 | 说明 |
| --- | --- | --- |
| `k+m=n_train`，spare 不进入 E | 已实现 | config/layout 双重 fail-fast。 |
| Cauchy systematic matrix、GF inverse | 已实现 | CPU 小矩阵；byte exact。 |
| CUDA table GF encode/decode | 已实现 | 主热路径；无 CPU/Jerasure fallback。 |
| CUDA bitmatrix | 已实现（实验） | 未自动替换 table kernel。 |
| W%k virtual zero | 已实现 | 不计算、不传输、不存储。 |
| serialization-free tensor-tree/chunking | 已实现 | model/optimizer tensor bytes 与小 metadata 分离。 |
| pinned arena 双槽生成 | 已实现 | packing D2H 双 stream/slot。 |
| async store-many | 已实现 | foreground NCCL/EC/enqueue，background wait/commit。 |
| CSD two-phase commit | 已实现 | child tag committed-only load。 |
| CSD-authoritative generation commit | 已实现 | 覆盖所有 storage-bearing train-node CSD；shared file 只作 v2/控制面兼容缓存。 |
| atomic metadata-only commit | 已实现 | tree/top marker 不再执行 get+begin+put+commit，不重复写 CSD `.pt` cache。 |
| native pinned / real EGM | 已实现 | 均 daemon-owned；内置 EGM 与 CSD 同生命周期。 |
| process restart load | 已实现并实测 | CSD 保持存活，训练进程重新建 CUDA/NCCL。 |
| 1～m logical row erasure decode | 核心已实现 | 本轮增加 k5m3 全部 56 种三行擦除 CUDA 测试和远端逻辑故障实验。 |
| replacement payload routing | 已修复 | 仍不等于 replacement Megatron process 自动接管。 |
| repair | 单 chunk publish/reader lease 已实现 | 单进程多 chunk repair transaction、distributed/Megatron background repair 未实现。 |
| TrainingLocal / Hybrid planner | 未实现 | 执行路径仅 spare-compute。 |
| multi-spare balance | 未实现 | 当前固定 `spare_ranks[0]`。 |
| elastic physical node replacement | 未实现 | 需要 launcher/world/CSD directory 联合设计。 |
| ECCHECK-style network idle scheduler | 未实现 | 当前 P2P 顺序和 async storage overlap。 |
| bounded inflight child window | 未实现 | 当前以 GB200 显存换取 caller latency；应后续加可配置窗口。 |

## 4. 测试口径

标准 restart run 必须满足：80 unique iterations、16 stores、3 loads、3 kill
markers、12 phase/node done markers、iteration 80 committed generation。它只验证
restart visibility 和性能。

EC correctness 另行验证：

```text
k=5,m=3,n=8,q=2
枚举 C(8,3)=56 种最大三行擦除
以及 remote-spare Megatron run 中显式 failed rows 0,1,2
```

## 5. 即时未修改基线

Run：

```text
pai_runs/egm_1p5b_gbs16_k5m3_s1_baseline_67f3003_20260711_0637
```

| 指标 | 基线 |
| --- | ---: |
| unique iteration / stores / restart loads | 80 / 16 / 3 |
| 稳态非 checkpoint mean / p50 / p95 | 696.567 / 613.300 / 875.800 ms |
| save function / caller timer | 170.116 / 205.076 ms |
| store makespan | 442.691 ms |
| logical / encoded BW | 60.094 / 112.146 GB/s |
| EGM event normalized aggregate BW | 671.018 GB/s |
| restart fetch mean / p50 / p95 | 288.820 / 291.275 / 299.058 ms |

## 6. 修改后回归

首轮强一致实现 run：

```text
pai_runs/egm_1p5b_gbs16_k5m3_s1_postaudit_20260711_01
```

正确完成 80 iterations / 16 stores / 3 restart loads，但新增 tree/top CSD
authority 后 `async commit mean=1107.565 ms`，相对基线 `+8.049%`，超过 5%
初筛门槛。profile 显示每棵约 129 KiB tree 执行
`get + begin + put_manifest + commit`，重复传输、SQLite 写入和 `.pt` cache
持久化。因而没有接受该结果，而是加入 metadata-only 单 RPC/单事务协议。

metadata fast-path 优化后的首个无并发完整 run：

```text
pai_runs/egm_1p5b_gbs16_k5m3_s1_atomicmeta_clean_20260711_03
```

| 指标 | 基线 | 最终 | 变化 |
| --- | ---: | ---: | ---: |
| unique iteration / stores / restart loads | 80 / 16 / 3 | 80 / 16 / 3 | 一致 |
| 稳态 mean | 696.567 ms | 652.666 ms | -6.30% |
| 稳态 p50 / p95 | 613.300 / 875.800 ms | 603.500 / 822.900 ms | -1.60% / -6.04% |
| save function | 170.116 ms | 148.143 ms | -12.92% |
| caller timer | 205.076 ms | 184.852 ms | -9.86% |
| store makespan mean / p50 | 442.691 / 427.661 ms | 448.260 / 408.396 ms | +1.26% / -4.50% |
| logical / encoded store BW | 60.094 / 112.146 GB/s | 60.668 / 113.217 GB/s | +0.95% |
| EGM event BW | 671.018 GB/s | 668.830 GB/s | -0.33% |
| async commit | 1025.061 ms | 885.864 ms | -13.58% |
| restart fetch mean / p50 | 288.820 / 291.275 ms | 242.880 / 185.781 ms | -15.91% / -36.22% |

该 run 中两次真实 CSD 尾延迟使 store p95 达 663.796 ms，但 16 次 store
均值相对即时基线只增加 1.26%，caller、训练和 p50 均改善。新提交口径为：

| atomic authority 指标 | 首轮强一致实现 | 最终 | 变化 |
| --- | ---: | ---: | ---: |
| checkpoint commit wall | 1182.163 ms | 959.266 ms | -18.85% |
| metadata publish | 311.780 ms | 95.396 ms | -69.40% |
| tree publish | 240.671 ms | 81.671 ms | -66.07% |
| per-CSD top marker | 59.926 ms | 1.151 ms | -98.08% |
| logical / encoded commit BW | 22.430 / 41.859 GB/s | 28.572 / 53.320 GB/s | +27.38% |

该 run 的 iteration 80 在 node0/node1 各有同 generation 的 committed top marker，
rank 0～7 共 8 个 CSD tree manifest，共享 cache generation 一致；所有正式 PUT
均 strict direct IPC、无 staging/error，三次 load 均报告 manifest v3。

`atomicmeta_20260711_02` 与远端 CUDA pytest 重叠，phase00 受到 GPU0/CSD 竞争，
并且事后静态 marker 不完整；该 run 明确作废，不用于性能或正确性结论。

所有后续 correctness/lifecycle 修复合入后的最终代码 run：

```text
pai_runs/egm_1p5b_gbs16_k5m3_s1_finalreview_20260711_04
```

| 指标 | 即时基线 | 最终代码 | 变化 |
| --- | ---: | ---: | ---: |
| unique iteration / stores / restart loads | 80 / 16 / 3 | 80 / 16 / 3 | 一致 |
| 稳态 mean | 696.567 ms | 646.767 ms | -7.15% |
| 稳态 p50 / p95 | 613.300 / 875.800 ms | 624.000 / 838.100 ms | +1.74% / -4.30% |
| save function / caller | 170.116 / 205.076 ms | 160.539 / 193.959 ms | -5.63% / -5.42% |
| store mean / p50 | 442.691 / 427.661 ms | 444.802 / 412.186 ms | +0.48% / -3.62% |
| logical / encoded store BW | 60.094 / 112.146 GB/s | 60.642 / 113.169 GB/s | +0.91% |
| EGM event BW | 671.018 GB/s | 672.102 GB/s | +0.16% |
| async commit | 1025.061 ms | 863.091 ms | -15.80% |
| restart fetch mean / p50 / p95 | 288.820 / 291.275 / 299.058 ms | 182.372 / 182.568 / 187.944 ms | mean -36.86% |

相对 `atomicmeta_clean_03`，最终代码的 checkpoint commit 为 951.096 ms
（-0.85%），metadata/tree publish 为 63.989/50.188 ms（-32.92%/-38.55%），
logical/encoded commit BW 为 28.799/53.745 GB/s（+0.80%）。CSD marker 均值
1.221 ms 比前 run 高 0.070 ms，但 p50 略优，属于亚毫秒噪声。正式 runbook 六项
5% gate 全部通过；其中 caller `+4.927%` 最接近门槛。save-function mean 相对前
run 的 +8.37% 来自唯一 iteration 45 的 423.70 ms 尾样本，其 p50 反而下降
2.10%，并且相对原始基线仍改善 5.63%。store p95 665.729 ms 与前一干净 run 的
663.796 ms 一致；均值、p50、caller、训练 p95 和 async p95 均未退化。

最终 generation `18c12fcc471609d7-16d0e:3:megatron:iter_0000080` 在共享 marker、
rank 0～7 的 v3 tree 及 node0/node1 CSD top/tree 完全一致。832 次正式 PUT 全部
strict direct IPC、无 staging/error，156 次 GET 均有 daemon IPC profile，node2
无正式 PUT/GET；三个节点退出后无残留进程或端口。

最终回归测试：RACER 本地 `124 passed, 45 skipped`；Megatron adapter 本地
`58 passed`；`.175` 远端 CSD/manifest/adapter 合并回归
`106 passed, 1 skipped`；远端 distributed/codec 最新组合 `29 passed`。另在
`.175` 上通过 k5m3 全部 `C(8,3)=56` 种最大擦除 CUDA 枚举测试。

最大三行逻辑擦除的首轮 Megatron 实验：

```text
pai_runs/egm_1p5b_gbs16_k5m3_s1_erasure3_20260711_01
```

确实触发了 24 次 GPU decode，load 和 iteration 6 forward 成功，但 iteration 7
出现 NaN。根因不是 RS 矩阵或 optimizer adapter：store 将两个尾 reduction group
分别补齐到 387,973,120 / 375,390,208 bytes，旧 load 却按真实 packet 最大值
386,894,848 / 373,813,253 bytes 建立 survivor recv；owner 仍发送完整存储宽度，
NCCL count 相差 1,078,272 / 1,576,955 bytes。被静默损坏的尾 chunk 恰好全部是
distributed optimizer state，因此第一轮 forward 正常，optimizer step 后才传播
NaN。修复后 load 以 manifest `group_nbytes` 统一 CSD read、survivor send/recv 和
decode 输入，只在 decode 后按 valid payload 截短，并增加非对齐尾组单测。

修复后的端到端最大擦除 run：

```text
pai_runs/egm_1p5b_gbs16_k5m3_s1_erasure3_sha256_20260711_02
```

完成 10 iterations / 2 stores / 1 restart load / 1 kill / 6 node-phase done
markers；实际 `load_decode_request_count=24`、decode route 非零。52 个从 EGM/CSD
读出的 storage chunks 均通过 sample64，rank 0～7 共 28 个恢复 payload chunks
均通过端到端 SHA-256；iteration 6～10 的 loss/grad 有限且 nan count 为 0。该结果
证明当前代码能恢复 `m=3` 个逻辑 codeword row，但仍不代表物理节点缺席时的
elastic world replacement。

## 7. 仍需后续处理

以下不应在缺少独立实验的情况下直接混入当前热路径：

1. `max_inflight_child_tags`：可降低 staging 显存，但可能增加 foreground
   prepare/finalize 次数和 checkpoint caller time。
2. TrainingLocal/Hybrid/multi-spare：需要真实 topology bandwidth 和训练干扰
   cost，不能只补 planner 类而不改执行器。
3. elastic replacement：需要让 replacement Megatron process 以原逻辑 rank
   加入新 world；当前 remote spare worker 只是 EC compute worker。
4. distributed repair：单 chunk 已有 copy-before-publish 和 reader lease，但仍
   需要全局 location directory、多 chunk generation rollback 和容错状态机；
   恢复成功后目前不会自动补回完整冗余。
5. CSD abort：状态枚举已有 `ABORTED`，但 public abort API 和失败 generation
   的跨 rank 协调仍需完整实现。
