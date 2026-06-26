# Megatron RACER CSD 重启测试操作说明

这份说明用于在没有 Codex 的机器上跑 1.5B / 5.3B RACER checkpoint 测试，并读出真实的训练、保存、重启加载耗时。

## 1. 测试目标

测试脚本会做完整 restart-aware 流程：

1. 启动 CSD daemon，后端默认 `native_pinned`。
2. 启动 Megatron 训练进程。
3. 训练到指定 iteration 后保存 RACER checkpoint。
4. 杀掉第一轮训练进程，CSD daemon 保持存活。
5. 重新启动 Megatron 训练进程。
6. 新训练进程从 CSD 读取 committed checkpoint 并恢复训练。
7. 继续训练并再次保存 checkpoint。

这里的 chunk 固定为 `1073741824` bytes，也就是 1GiB。

## 2. 直接运行

默认假设 `racer` 和 `Megatron-LM-FT` 在同一个父目录下，例如：

```text
some_parent/
  racer/
  Megatron-LM-FT/
  data/my_shakespeare_text_document
  gpt2_vocab/vocab.json
  gpt2_vocab/merges.txt
```

进入实际的 `racer` 目录运行：

```bash
cd /path/to/racer
SAVE_INTERVAL=1 KILL_AFTER_ITER=1 RESUME_TRAIN_ITERS=2 examples/run_megatron_csd_restart_1_5b.sh
```

```bash
cd /path/to/racer
SAVE_INTERVAL=1 KILL_AFTER_ITER=1 RESUME_TRAIN_ITERS=2 examples/run_megatron_csd_restart_5_3b.sh
```

常用参数通过环境变量控制：

- `SAVE_INTERVAL`: 每多少个 iteration 保存一次 checkpoint。
- `KILL_AFTER_ITER`: 第一轮训练保存到哪个 iteration 后杀掉训练进程，用来模拟重启。
- `RESUME_TRAIN_ITERS`: 重启后的训练总 iteration 数。
- `GLOBAL_BATCH_SIZE`: 全局 batch size，5.3B wrapper 默认是 `8`。
- `CUDA_VISIBLE_DEVICES`: 默认 `0,1,2,3,4`，其中 4 个 train rank，1 个 spare rank。
- `RACER_BUFFER_SIZE`: RACER chunk 大小，默认固定 `1073741824`。
- `CSD_NATIVE_PINNED_TOTAL_BYTES`: CSD native pinned pool 总大小。
- `TIMEOUT_SECONDS`: 整个测试超时时间。
- `WORKSPACE_ROOT`: `racer` 和 `Megatron-LM-FT` 的共同父目录；默认自动取 `racer/..`。
- `MEGATRON_ROOT`: Megatron repo 路径；默认 `$WORKSPACE_ROOT/Megatron-LM-FT`。
- `DATA_PATH`: Megatron indexed dataset 前缀；默认 `$WORKSPACE_ROOT/data/my_shakespeare_text_document`。这里填的是前缀，不带 `.bin` / `.idx` 后缀；实际文件应为 `DATA_PATH.bin` 和 `DATA_PATH.idx`。
- `GPT2_VOCAB_FILE` 或 `VOCAB_FILE`: GPT-2 vocab 文件；默认 `$WORKSPACE_ROOT/gpt2_vocab/vocab.json`。
- `GPT2_MERGE_FILE` 或 `MERGE_FILE`: GPT-2 merges 文件；默认 `$WORKSPACE_ROOT/gpt2_vocab/merges.txt`。
- `OUTPUT_ROOT`: 结果输出目录；默认 `$RACER_ROOT/results/megatron_csd_restart`。

如果目标机器不是上述默认布局，显式指定路径：

```bash
cd /path/to/racer
MEGATRON_ROOT=/path/to/Megatron-LM-FT \
DATA_PATH=/path/to/data/my_shakespeare_text_document \
GPT2_VOCAB_FILE=/path/to/gpt2_vocab/vocab.json \
GPT2_MERGE_FILE=/path/to/gpt2_vocab/merges.txt \
SAVE_INTERVAL=1 KILL_AFTER_ITER=1 RESUME_TRAIN_ITERS=2 \
examples/run_megatron_csd_restart_1_5b.sh
```

## 3. 输出目录

每次运行会生成一个目录：

```text
$RACER_ROOT/results/megatron_csd_restart/gpt2_1.5b_YYYYMMDD_HHMMSS
$RACER_ROOT/results/megatron_csd_restart/gpt2_5.3b_YYYYMMDD_HHMMSS
```

关键文件：

- `first_run.log`: 第一轮训练日志。
- `resume_run.log`: 重启后训练日志。
- `csd.log`: CSD daemon 日志。
- `parsed_log_events.csv`: 最重要的汇总事件表。
- `per_rank_profile_events.csv`: 每个 rank 的更细 profile。
- `csd_profile_summary.csv`: CSD 后端 put/get 的分解汇总。
- `summary.json`: 机器可读 summary。
- `report.md`: 脚本自动生成的简表。

## 4. 一键读结果

跑完后执行：

```bash
python scripts/summarize_megatron_csd_restart.py results/megatron_csd_restart/gpt2_1.5b_YYYYMMDD_HHMMSS
```

输出里重点看四块：

1. `Normal training iteration time from raw Megatron logs`
   - 这是 Megatron 原始日志里的正常 iteration 时间。
   - 字段 `elapsed_per_iter` 是每个 iteration 的 wall time。

2. `Megatron checkpoint blocking events`
   - `save_fn_total` 是训练进程真正被 checkpoint 阻塞的总时间。
   - `racer_adapter_save` 在 `save_fn_total` 里面，不要和它相加。
   - `state_dict` 和 `optimizer_capture` 也是 `save_fn_total` 的内部阶段。

3. `RACER store events`
   - `adapter_store` 是 RACER adapter 的 store 总耗时，属于 `save_fn_total` 内部。
   - `racer_calls` 是所有 RACER distributed store 调用的累计时间。
   - `max_chunk_store` 是最慢的单个 logical chunk store。
   - `data_rows` 是 data row 传输/归位阶段。
   - `parity` 是 spare GPU 计算 parity 和传回 train rank 的阶段。
   - `csd_storage` 是把最终 data/parity row 写到 CSD 的阶段。
   - `csd_wait` 是等待 CSD 异步 put 完成的时间。
   - 这些是内部诊断字段，不要和 `adapter_store` 简单相加。

4. `RACER restart/load events`
   - `load_total` 是重启后读取 RACER checkpoint 的总时间。
   - `racer_fetch` 是从 CSD 取回 encoded rows 并恢复所需数据的阶段。
   - `read_wait` 是等待 CSD read op 完成的时间。
   - `materialize` 是把恢复出的 byte payload 变回 Megatron state_dict tensor 的阶段。
   - `tree_decode` 是恢复 tensor tree 结构的阶段。

## 5. 判断 restart 是否真的发生

在 summary 输出里必须看到：

```text
Restart checks: load_observed=True  post_resume_checkpoint_observed=True
```

也可以手动查：

```bash
grep -n "RACER distributed memory checkpoint loaded" results/megatron_csd_restart/gpt2_*/resume_run.log
```

如果没有这行，说明没有实际 load resident checkpoint。

## 6. 读 CSV 的最小规则

`parsed_log_events.csv` 里的 `event` 字段分三类：

- `blocking`: Megatron save_checkpoint 阻塞训练进程的时间。
- `store`: RACER 保存 checkpoint 的内部阶段。
- `load`: 重启后 RACER load checkpoint 的阶段。

最可信的阻塞时间是：

```text
event=blocking 的 save_checkpoint_fn_total_ms
```

最可信的 load 时间是：

```text
event=load 的 total_ms
```

## 7. 1GiB chunk 的含义

`RACER_BUFFER_SIZE=1073741824` 表示每个 rank 的本地 checkpoint byte payload 会按 1GiB 切成多个 logical chunk。每个 logical chunk 进入 RACER 后，会生成属于 train ranks 的 data/parity rows。spare rank 只负责计算，不拥有最终 checkpoint row。

如果某个 rank 在最后一个 logical chunk 没有本地 payload bytes，RACER 会使用固定预分配的 zero-send slot 补齐编码输入，不会给 virtual zero slot 分配真实 checkpoint row。

## 8. 常见失败判断

- CSD 启动失败：看 `csd.log`，确认 CUDA 权限和 pinned pool 大小。
- 没有 restart load：看 `resume_run.log` 是否有 `RACER distributed memory checkpoint loaded`。
- OOM：降低 `RACER_BUFFER_SIZE` 或 `GLOBAL_BATCH_SIZE`，并确认 `CSD_NATIVE_PINNED_TOTAL_BYTES` 足够。
- 性能异常：先看 `save_fn_total`，再看 `adapter_store`，最后看 `csd_profile_summary.csv` 里的 allocation / copy / checksum / sqlite。
