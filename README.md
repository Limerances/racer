# RACER

RACER is a CUDA-only spare-GPU checkpoint coding and daemon-owned in-memory
checkpoint runtime. The current research path integrates Megatron, per-node
CSD, native pinned memory or topology-aware EGM, strict CUDA IPC, and remote
spare GPU encoding/recovery.

- Preserves GF(2^8) Cauchy Reed-Solomon semantics: `k + m == len(train_ranks)`.
- Treats `spare_ranks` as compute-only CUDA resources outside the E matrix.
- Uses RACER CUDA kernels for the spare-GPU runtime GF work.
- Keeps Jerasure as an external CPU oracle/baseline route for comparison, not as a runtime fallback.
- Supports raw `Dict[int, torch.uint8 Tensor]` packets and realistic `Dict[int, Dict[str, Tensor]]` checkpoint states.
- Exposes lightweight store/load profiling through `RacerContext.last_store_profile` and `RacerContext.last_load_profile`.

## API

```python
import racer
from racer.csd import CheckpointStorageDaemonClient

csd = CheckpointStorageDaemonClient(("127.0.0.1", 7007), authkey="racer-csd")

ctx = racer.init(
    k=3,
    m=1,
    train_ranks=[0, 1, 2, 3],
    spare_ranks=[4],
    storage_backend="csd_egm",
    storage_options={"client": csd},
)

handle = racer.store(rank_state_dicts, tag="iter_100", context=ctx)
recovered = racer.load(tag="iter_100", failed_train_ranks=[0], context=ctx)
```

RACER intentionally has no in-process, CPU-byte, file-mmap, or socket fallback.
The CSD must already be running and its capabilities must match the requested
backend.

`rank_state_dicts` may be either:

- `Dict[int, torch.Tensor]` where every tensor is contiguous `torch.uint8` on CUDA.
- `Dict[int, Dict[str, torch.Tensor]]` for model/optimizer checkpoint payloads.

## GPT2 Checkpoint Benchmark

Default 5-GPU shape: 4 train GPUs plus 1 spare GPU.

```bash
cd /workspace/racer
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

For a quick CUDA smoke test:

```bash
cd /workspace/racer
python examples/bench_gpt2_checkpoint.py \
  --profile megatron-test-tp4 \
  --max-tensors 24 \
  --iters 1 --warmup 0 \
  --verify
```

Jerasure remains available as a comparison route:

```bash
cd /workspace/racer
python examples/bench_jerasure_baseline.py \
  --profile megatron-test-tp4 \
  --max-tensors 24 \
  --iters 1 --warmup 0 \
  --load-ranks failed \
  --fill
```

## Tests

```bash
cd /workspace/racer
pytest -q
```

CUDA route tests are skipped automatically when CUDA is unavailable. Jerasure comparison tests are skipped when the external Jerasure shared library is unavailable. The project-owned CPU data path is intentionally not supported.

## GB200 multi-node path

The maintained three-node EGM/restart procedure, strict success criteria, and
explicit erasure-decode test are documented in
[`docs/gb200_multinode_test_runbook.md`](docs/gb200_multinode_test_runbook.md).
Formal runs must pass `RACER_K`, `RACER_M`, train/spare ranks, GBS, and unique
ports explicitly; do not rely on the shell scripts' generic defaults.
