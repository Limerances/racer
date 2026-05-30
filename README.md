# RACER O2

`racer_o2` combines the stronger parts of `/workspace/racer_o1` and `/workspace/racer`:

- Keeps the `racer_o1` architecture for config validation, Cauchy RS matrix generation, elastic layout, routing plans, chunk storage, manifests, pinned/file-mmap backends, and the public `racer.init()`, `racer.store()`, `racer.load()` API.
- Adds `racer`-style `state_dict` flatten/unflatten support, so `store()` can accept either raw `Dict[int, torch.uint8 Tensor]` packets or realistic `Dict[int, Dict[str, Tensor]]` checkpoint states.
- Preserves GF(2^8) Cauchy Reed-Solomon semantics: `k + m == len(train_ranks)`. Spare ranks are CUDA compute/storage accelerators and are not rows in the E matrix.
- Adds lightweight store/load profiling fields on `RacerContext`: `last_store_profile` and `last_load_profile`.
- Adds a GPT2/Megatron-style synthetic checkpoint generator and benchmark with many tensors per rank, including optional Adam optimizer states.
- Adds an optional CUDA extension fast path, `apply_matrix_cuda_table`, that keeps compatibility with the older extension while enabling table-based GF multiplication after rebuild.
- Keeps Jerasure as an external oracle/baseline route only; spare-GPU runtime GF work uses RACER CUDA kernels and does not call Jerasure.

## API

```python
import racer

ctx = racer.init(
    k=3,
    m=1,
    train_ranks=[0, 1, 2, 3],
    spare_ranks=[4],
    backend="cuda",
    storage_backend="in_process_cuda",
)

handle = racer.store(rank_state_dicts, tag="iter_100", context=ctx, async_op=False)
recovered = racer.load(tag="iter_100", failed_train_ranks=[0], context=ctx)
```

`rank_state_dicts` may be either:

- `Dict[int, torch.Tensor]` where every tensor is contiguous `torch.uint8`.
- `Dict[int, Dict[str, torch.Tensor]]` for model/optimizer checkpoint payloads. RACER serializes each rank state into one byte payload, runs EC over those payloads, then reconstructs dtype, shape, device, and `requires_grad` metadata on load.

## GPT2 Checkpoint Benchmark

Default 5-GPU test shape: 4 train GPUs plus 1 spare GPU.

```bash
cd /workspace/racer_o2
python examples/bench_gpt2_checkpoint.py \
  --profile gpt2-124m \
  --tp 4 \
  --dtype bf16 \
  --include-optimizer true \
  --train-ranks 0,1,2,3 \
  --spare-ranks 4 \
  --k 3 --m 1 \
  --iters 3 --warmup 1
```

To print the full GPT2-124M TP4 checkpoint size without allocating tensors:

```bash
cd /workspace/racer_o2
python examples/bench_gpt2_checkpoint.py \
  --profile gpt2-124m \
  --tp 4 \
  --dtype bf16 \
  --include-optimizer true \
  --estimate-only
```

For a quick CUDA smoke test without materializing the full GPT2 checkpoint:

```bash
cd /workspace/racer_o2
python examples/bench_gpt2_checkpoint.py \
  --profile megatron-test-tp4 \
  --max-tensors 24 \
  --iters 1 --warmup 0 \
  --verify
```

The GPT2 generator is intentionally standalone. It only uses Megatron-LM-FT as a shape reference and does not import or integrate Megatron.

## Tests

```bash
cd /workspace/racer_o2
pytest -q
```

CUDA route tests are skipped automatically when CUDA is unavailable. Jerasure comparison tests are skipped when the external Jerasure shared library is unavailable. The project-owned CPU data path is intentionally not supported.
