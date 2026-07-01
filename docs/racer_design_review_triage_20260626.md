# RACER 设计审阅逐条结论（2026-06-26）

审阅对象：`/workspace/RACER_设计文档_v0.2.md`、`/workspace/racer`、`/workspace/Megatron-LM-FT` 的 RACER 集成。

结论分三类：

- 已修复：本轮做了代码修改和测试。
- 存在但暂不强改：问题属实，但属于设计补齐、较大重构或性能/安全取舍，本轮不应混进热路径。
- 不构成当前 bug：描述有依据，但当前调用契约或已有保护使它不是现路径阻塞问题。

## 功能性 / 正确性

| 编号 | 结论 | 处理 |
| --- | --- | --- |
| C1 `_async_ops` / native `_ops` 只增不删 | 属实，严重。会导致长跑内存增长和 poll 开销恶化。 | 已修复。CSD `wait()` 收尾删除 `_async_ops`；native `profile()` 后删除完成 op；补单测。 |
| C2 native `wait` 与 poller 销毁 event 竞态 | 属实，严重。`wait()` 锁外同步 event 时 poller 可能已销毁。 | 已修复。op 增加 `in_completion` 认领状态，poller 跳过 wait 正在处理的 op；补单测。 |
| C3 `_ipc_mem_cache` refcount 不下降 | 属实。长期 producer 变化会积累 IPC handle。 | 已修复。完成 op 后递减 refcount，并按 `RACER_CSD_IPC_MEM_CACHE_MAX_ENTRIES` 驱逐 idle handle；默认保留小 cache 以避免性能回退。 |
| C4 store reload 单槽复用 | 部分属实。当前 `distributed_store` 是同步 wait 后返回，现路径不构成串扰 bug；强行改双槽会扰动热路径。 | 暂不改。保留单槽；未来如果 `distributed_store` 变成真正异步返回，再改成双槽或显式 slot 复用同步。 |
| C5 load 跨流 scatter 缺少同步 | 属实。materialize stream 读 payload 前缺少显式顺序。 | 已修复。distributed/local tensor-tree load 在 scatter 前执行 `materialize_stream.wait_stream(current_stream)`。 |
| C6 failed rank 无 replacement 时发回失败 rank | 属实，严重。真实故障会向失效 rank 回传。 | 已修复。无 replacement 时默认发到第一个 spare rank；没有 spare 时显式报错；补单测。 |
| C7 slot3/slot4 异构尾块容量 | 当前不构成已测路径 bug。`_payload_store_slot` 要求短尾块来自 full-size slot；Megatron distributed store 现在传入的是 pinned payload full-size slot。 | 暂不改。后续若允许任意短 tensor 直接进入 distributed store，应把接收槽容量契约改成显式 `valid_nbytes + capacity_nbytes`。 |
| C8 load 缺 committed / daemon_owned / data_resident 校验 | 属实，严重。manifest 标记不满足时仍可能继续读 chunk。 | 已修复。新增 `validate_committed_daemon_manifest()`，核心 load 和 distributed storage load 都校验三标志；补单测。 |

## 设计一致性

| 编号 | 结论 | 处理 |
| --- | --- | --- |
| D1 Megatron tree manifest 只落本地文件 | 属实。chunk bytes/metadata 在 CSD，state_dict skeleton/metas/chunk layout 之前只在本地文件。 | 已修复一层。保留本地缓存，同时把每 rank tree manifest 作为 metadata-only wrapper 写入 CSD；load 时本地缺失会从 CSD 取。 |
| D2 per-rank 本地 manifest 阻断 replacement | 属实。replacement 进程需要被接管 rank 的 tree manifest。 | 已修复。replacement 目标 rank 会加载 source train rank 的 tree manifest；本地没有时走 CSD 副本。 |
| D3 train-side pinned arena / 4 * chunk_size 上界 | 部分属实。Megatron 已有 pinned payload chunks，但 reload 当前仍按同步 store 的单槽路径走；`racer/context.py` 单进程路径仍不是完整 arena 状态机，也没有严格 4 * chunk_size 上界。 | 暂不强改。这是较大架构重构，不能和 correctness 修复混做。 |
| D4 缺 TrainingLocalPlanner / HybridPlanner | 属实，是设计未完成项，不是当前 spare-compute 路径 bug。 | 暂不改。当前 `make_planner()` 仍只返回 `SpareComputePlanner`。 |
| D5 CostModel 字段不对齐设计 | 属实。缺 `final_result_return_bytes`，spare 字段命名偏离文档。 | 已修复。补设计字段，同时保留旧 `accelerators` 字段兼容现有报告；补单测。 |
| D6 readiness bitmap 未实现 | 属实，但当前实现按 group 同步顺序处理，不是当前同步路径 bug。 | 暂不改。多 window 并发时再补 bitmap/状态机。 |
| D7 `_select_tag` 绕过 committed 门控 | 属实。session tag 可能绕过 chunk committed 检查。 | 已修复。session tag 命中后也必须通过 `_manifest_chunks_committed()`。 |
| D8 RACER load 返回 None 静默 fallback | 部分属实。无 checkpoint 冷启动返回 None 是合理的；显式请求某 iteration/release 时静默 fallback 不合理。 | 已修复显式请求路径：指定 iteration/release 但 tag 不存在或未 commit 时直接 raise。 |
| D9 EGM 禁止 fd/socket 但 CSD 仍保留分支 | 属实但当前不是可达传输面。相关 op 已硬 raise，不能完成 fd/socket 读写。 | 暂不清理。属于死代码/维护面收缩，可单独做删除 PR。 |

## 性能

| 编号 | 结论 | 处理 |
| --- | --- | --- |
| P1 CUDA 扩展热路径 `cudaFree` 同步 | 属实。三处 kernel wrapper 在 launch 后 raw `cudaFree`，会形成同步点。 | 已修复。改为同 stream 的 `cudaMallocAsync/cudaFreeAsync` 管理 device pointer table，避免 host 同步释放；`tests/test_codec_cuda.py` 通过。 |
| P2 context 热路径大量 `torch.cuda.synchronize` | 属实，主要在 `racer/context.py` 单进程路径。 | 暂不改。需要 event 化重写，风险高；Megatron 当前主测路径更多走 distributed/CSD。 |
| P3 CSD client / capabilities 重复创建 | 属实。Megatron 每次 `_racer_chunk_storage()` 都会建 client 并做 capabilities RPC。 | 已修复。按 backend/address/authkey/register 复用 session client，`reset_session_state()` 会 close。 |
| P4 sampled checksum 弱 | 属实但属于性能/完整性取舍。强制 sha256 会明显影响当前训练保存性能。 | 暂不改默认。保留 `sample64` 快速路径；需要强校验时用配置/环境切到 sha256 或专门验证任务。 |

## 冗余 / 死代码

| 项 | 结论 | 处理 |
| --- | --- | --- |
| `storage.py` 多个不可达 backend | 基本属实。 | 暂不删。删除会影响导入兼容和测试假对象，单独清理。 |
| `FdMmapHostBackend` 死实现 | 属实，入口已 hard fail。 | 暂不删。与 D9 一起单独清理。 |
| `stats.py` 基本未使用 | 属实。 | 暂不删。低风险但和本轮 correctness 无关。 |
| `context.py` 本地 checkpoint index 死分支 | 属实。 | 暂不删。 |
| `distributed.py` 重复/死分支 | 部分属实。 | 暂不删，避免扰动 distributed 测试面。 |
| Megatron 死参数 | 属实。 | 暂不删，CLI 兼容优先。 |
| 实验报告入库 | 属实，属于仓库卫生。 | 暂不处理。 |

## 其他隐患

| 编号 | 结论 | 处理 |
| --- | --- | --- |
| S1 unflatten 视图别名/对齐 | 部分属实。dtype 对齐已有 clone 保护；返回 payload 视图仍是隔离性/性能取舍。 | 暂不改，避免 load 时全量 clone 造成性能回退。 |
| ShardedTensor 还原裸 tensor | 属实，非 legacy 分布式 checkpoint 语义未完全覆盖。 | 暂不改，需结合 Megatron dist-checkpoint 类型系统做。 |
| chained optimizer state 长度不匹配静默跳过 | 属实。 | 已修复。长度不匹配直接 `ValueError`。 |
| spare_ranks 默认 `[len(train)]` | 属实但有历史默认行为。 | 暂不改。直接改为必填可能破坏现有脚本。 |
| release tag 映射 iteration 0 | 属实，低风险命名问题。 | 暂不改。 |
| CSD close 资源释放不彻底 | 部分属实。daemon shutdown 路径会 close manifest store；backend 资源更多依赖对象析构。 | 暂不改，建议单独补 deterministic close。 |
| manifest 两阶段 commit 软探测 | 部分属实。当前 CSD path 有 begin/commit；通用 storage adapter 仍偏软。 | 暂不改。 |
| `data_resident` 用 list_chunks 判定 | 属实。对 metadata-only tree manifest 不应套三标志；对 chunk checkpoint 仍需更强 resident 证明。 | 已对 chunk load 加三标志门控；resident 证明增强另做。 |
| wait 捕获 `BaseException` | 属实。会吞 `KeyboardInterrupt/SystemExit` 类型。 | 暂不改。本轮不动 daemon 错误协议；建议后续改为 `Exception` 并保留 traceback。 |

## 本轮验证

- `pytest -q tests/test_csd.py tests/test_manifest.py tests/test_distributed_storage.py tests/test_routing.py`：31 passed。
- `pytest -q tests/test_csd_manifest.py tests/test_manifest_helpers.py`：7 passed。
- `pytest -q tests/test_codec_cuda.py`：8 passed。
- Python AST 编译检查：`racer/csd.py`、`racer/manifest.py`、`racer/distributed.py`、`racer/routing.py`、Megatron `racer_checkpointing.py`、`session_state.py`、`optimizer_state.py` 均通过。
- Megatron restart 实机脚本：每 5 iter 保存一次、每 20 iter 杀一次、共杀 3 次、最后一次恢复后再跑 20 iter；1.5B 和 5.3B 都完成 80 iter，并观察到 3 次从 RACER/CSD 恢复。

| 模型 | 结果目录 | 稳定非 checkpoint 迭代 p50 | 稳定非 checkpoint 迭代 mean | store mean | load total |
| --- | --- | ---: | ---: | ---: | --- |
| 1.5B 本轮 | `results/megatron_csd_restart/gpt2_1.5b_20260626_182221` | 854.25 ms | 866.18 ms | 657.08 ms | 235.73 / 229.65 / 223.43 ms |
| 1.5B 修改前同脚本 | `results/megatron_csd_restart/gpt2_1.5b_20260626_162536` | 856.65 ms | 857.04 ms | 662.04 ms | 234.39 / 230.95 / 235.33 ms |
| 5.3B 本轮 | `results/megatron_csd_restart/gpt2_5.3b_20260626_182755` | 970.85 ms | 971.25 ms | 2436.25 ms | 931.06 / 931.28 / 935.16 ms |
| 5.3B 修改前同脚本 | `results/megatron_csd_restart/gpt2_5.3b_20260626_163107` | 970.10 ms | 970.13 ms | 2007.20 ms | 560.37 / 551.06 / 539.08 ms |

训练主循环稳定迭代时间基本未退化。1.5B checkpoint store/load 也回到修改前同脚本水平。5.3B checkpoint store/load 比 `20260626_163107` 那次快基线慢，分段看主要是 CSD native pinned copy wall time 变长：同样 18.13 GiB payload、1976 次 CSD op，`daemon_memcpy_ms_wall_sum` 从 61180.28 ms 增到 90531.32 ms。独立 CSD 1 GiB * 19 benchmark 的 copy wall P50 约 19.88 ms，说明 CSD 本体 copy 不是固定慢；当前 5.3B 结果与同日 `20260626_153523`、`20260626_180703` 的 store/load 档位接近。这个问题不影响普通训练每迭代耗时，但不能把 5.3B checkpoint 路径说成相对最快基线无退化；后续应继续定位 Megatron store 阶段的 slot reload、CSD wait 与 copy 带宽波动。
