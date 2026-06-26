import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import racer


def main():
    if torch.cuda.device_count() < 5:
        raise SystemExit("verify_5gpu.py requires GPUs 0,1,2,3 as train ranks and GPU 4 as spare")

    with tempfile.TemporaryDirectory(prefix="racer-csd-") as tmp:
        daemon = racer.start_checkpoint_storage_daemon(
            metadata_dir=Path(tmp) / "metadata",
            backend="native_pinned",
            backend_options={"segment_bytes": 64 * 1024 * 1024, "device": 0},
        )
        try:
            ctx = racer.init(
                k=3,
                m=1,
                train_ranks=[0, 1, 2, 3],
                spare_ranks=[4],
                storage_backend="csd_native_pinned",
                storage_options={"client": daemon.client},
            )
            obj = {
                rank: torch.randint(0, 256, (8 * 1024 * 1024,), dtype=torch.uint8, device=f"cuda:{rank}")
                for rank in [0, 1, 2, 3]
            }
            racer.store(obj, tag="verify", context=ctx)
            recovered = racer.load(tag="verify", failed_train_ranks=[0], context=ctx)
            assert torch.equal(recovered[0].to(obj[0].device), obj[0])
            print("RACER 5-GPU verification passed; failed rank 0 recovered on", recovered[0].device)
        finally:
            daemon.shutdown()


if __name__ == "__main__":
    main()
