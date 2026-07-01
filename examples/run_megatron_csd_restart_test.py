#!/usr/bin/env python3
"""Run Megatron RACER CSD restart tests for GPT2-size models.

The harness:
1. starts a RACER Checkpoint Storage Daemon,
2. launches Megatron training until a configured RACER checkpoint is committed,
3. terminates the training process group,
4. repeats launch/kill cycles against the same CSD and manifest directory,
5. launches a final Megatron process that resumes from the last resident checkpoint,
6. verifies each restart load and the final post-restart checkpoint,
7. writes raw logs plus parsed CSV/JSON/Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any


RACER_ROOT = Path(os.environ.get("RACER_ROOT", Path(__file__).resolve().parents[1])).resolve()
WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", RACER_ROOT.parent)).resolve()
MEGATRON_ROOT = Path(os.environ.get("MEGATRON_ROOT", WORKSPACE_ROOT / "Megatron-LM-FT")).resolve()
DEFAULT_DATA_PATH = Path(os.environ.get("DATA_PATH", WORKSPACE_ROOT / "data/my_shakespeare_text_document")).resolve()
DEFAULT_VOCAB_FILE = Path(
    os.environ.get("GPT2_VOCAB_FILE") or os.environ.get("VOCAB_FILE") or WORKSPACE_ROOT / "gpt2_vocab/vocab.json"
).resolve()
DEFAULT_MERGE_FILE = Path(
    os.environ.get("GPT2_MERGE_FILE") or os.environ.get("MERGE_FILE") or WORKSPACE_ROOT / "gpt2_vocab/merges.txt"
).resolve()


MODEL_CONFIGS = {
    "1.5b": {
        "label": "gpt2_1.5b",
        "num_layers": 48,
        "hidden_size": 1600,
        "ffn_hidden_size": 6400,
        "num_attention_heads": 25,
        "default_save_interval": 5,
        "default_kill_interval_iters": 20,
        "default_kill_count": 3,
        "default_post_kill_train_iters": 20,
        "default_global_batch_size": 16,
        "default_csd_native_pinned_total_bytes": 96 * 1024**3,
        "default_max_local_payload_bytes": 6_314_206_720,
    },
    "5.3b": {
        "label": "gpt2_5.3b",
        "num_layers": 64,
        "hidden_size": 2560,
        "ffn_hidden_size": 10240,
        "num_attention_heads": 40,
        "default_save_interval": 5,
        "default_kill_interval_iters": 20,
        "default_kill_count": 3,
        "default_post_kill_train_iters": 20,
        "default_global_batch_size": 8,
        "default_csd_native_pinned_total_bytes": 256 * 1024**3,
        "default_max_local_payload_bytes": 19_463_132_160,
    },
}


STORE_RE = re.compile(
    r"RACER distributed tensor-tree checkpoint stored: "
    r"tag=(?P<tag>[^,]+), store=(?P<store_ms>[0-9.]+) ms, "
    r"metadata=(?P<metadata_ms>[0-9.]+) ms, "
    r"(?:(?:command=(?P<command_ms>[0-9.]+) ms, )?)"
    r"tensor_view=(?P<tensor_view_ms>[0-9.]+) ms, "
    r"racer_calls=(?P<racer_calls_ms>[0-9.]+) ms, "
    r"(?:(?:racer_inner_total=(?P<racer_inner_total_ms>[0-9.]+) ms, )?)"
    r"(?:(?:chunk_store_max=(?P<chunk_store_max_ms>[0-9.]+) ms, )?)"
    r"(?:(?:setup=(?P<setup_ms>[0-9.]+) ms, )?"
    r"(?:sizing=(?P<sizing_ms>[0-9.]+) ms, )?"
    r"(?:data_rows=(?P<data_rows_ms>[0-9.]+) ms, )?"
    r"(?:parity=(?P<parity_ms>[0-9.]+) ms, )?"
    r"(?:storage=(?P<storage_ms>[0-9.]+) ms, )?"
    r"(?:(?:storage_begin=(?P<storage_begin_ms>[0-9.]+) ms, )?)"
    r"(?:(?:storage_enqueue=(?P<storage_enqueue_ms>[0-9.]+) ms, )?)"
    r"(?:(?:storage_wait=(?P<storage_wait_ms>[0-9.]+) ms, )?)"
    r"(?:(?:storage_commit=(?P<storage_commit_ms>[0-9.]+) ms, )?)"
    r"(?:(?:client_put=(?P<client_put_ms>[0-9.]+) ms, )?)"
    r"(?:(?:client_export=(?P<client_export_ms>[0-9.]+) ms, )?)"
    r"(?:(?:client_rpc=(?P<client_rpc_ms>[0-9.]+) ms, )?)"
    r"(?:(?:daemon_put=(?P<daemon_put_ms>[0-9.]+) ms, )?)"
    r"(?:(?:daemon_dispatch=(?P<daemon_dispatch_ms>[0-9.]+) ms, )?)"
    r"(?:(?:daemon_submit=(?P<daemon_submit_ms>[0-9.]+) ms, )?)"
    r"(?:(?:daemon_backend_write=(?P<daemon_backend_write_ms>[0-9.]+) ms, )?)"
    r"(?:(?:daemon_event_open_us=(?P<daemon_event_open_us>[0-9.]+), )?)"
    r"(?:(?:daemon_set_device=(?P<daemon_set_device_ms>[0-9.]+) ms, )?)"
    r"(?:(?:daemon_event_create=(?P<daemon_event_create_ms>[0-9.]+) ms, )?)"
    r"(?:(?:daemon_enqueue_api=(?P<daemon_enqueue_api_ms>[0-9.]+) ms, )?)"
    r"(?:(?:staging_copy_enqueue_us=(?P<staging_copy_enqueue_us>[0-9.]+), )?)"
    r"(?:(?:wrapper_gap=(?P<wrapper_gap_ms>[0-9.]+) ms, )?)"
    r")?"
    r"bytes=(?P<bytes>[0-9]+), "
    r"local_bytes=(?P<local_bytes>[0-9]+), leaves=(?P<leaves>[0-9]+), chunks=(?P<chunks>[0-9]+)"
)
LOAD_RE = re.compile(
    r"RACER distributed memory checkpoint loaded: "
    r"tag=(?P<tag>[^,]+), total=(?P<total_ms>[0-9.]+) ms, "
    r"racer_fetch=(?P<racer_fetch_ms>[0-9.]+) ms, "
    r"(?:(?:load_read_wait=(?P<load_read_wait_ms>[0-9.]+) ms, )?"
    r"(?:load_route=(?P<load_route_ms>[0-9.]+) ms, )?"
    r"(?:load_barrier=(?P<load_barrier_ms>[0-9.]+) ms, )?)?"
    r"tensor_materialize=(?P<tensor_materialize_ms>[0-9.]+) ms, "
    r"(?:(?:tensor_scatter_sync=(?P<tensor_scatter_sync_ms>[0-9.]+) ms, )?)?"
    r"tree_decode=(?P<tree_decode_ms>[0-9.]+) ms"
    r"(?:(?:, unpack=(?P<unpack_ms>[0-9.]+) ms, "
    r"tensor_rebuild=(?P<tensor_rebuild_ms>[0-9.]+) ms, "
    r"runtime_prewarm_after_load=(?P<runtime_prewarm_after_load_ms>[0-9.]+) ms)?)"
)
BLOCKING_RE = re.compile(
    r"RACER checkpoint blocking profile: iteration=(?P<iteration>[0-9]+), "
    r"pre_state=(?P<pre_state_ms>[0-9.]+) ms, "
    r"optimizer_capture=(?P<optimizer_capture_ms>[0-9.]+) ms, "
    r"state_dict=(?P<state_dict_ms>[0-9.]+) ms, "
    r"racer_adapter_save=(?P<racer_adapter_save_ms>[0-9.]+) ms, "
    r"finalize=(?P<finalize_ms>[0-9.]+) ms, "
    r"save_checkpoint_fn_total=(?P<save_checkpoint_fn_total_ms>[0-9.]+) ms"
)
SAVE_TIMER_RE = re.compile(
    r"RACER save blocking time: iteration=(?P<iteration>[0-9]+), "
    r"save_checkpoint_timer=(?P<save_checkpoint_timer_ms>[0-9.]+) ms"
)
ITER_TIME_RE = re.compile(
    r"iteration\s+(?P<iteration>[0-9]+)/\s*(?P<train_iters>[0-9]+).*?"
    r"elapsed time per iteration \(ms\):\s*(?P<elapsed_ms>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--racer-root", default=str(RACER_ROOT))
    parser.add_argument("--megatron-root", default=str(MEGATRON_ROOT))
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--vocab-file", default=str(DEFAULT_VOCAB_FILE))
    parser.add_argument("--merge-file", default=str(DEFAULT_MERGE_FILE))
    parser.add_argument("--output-root", default=str(RACER_ROOT / "results/megatron_csd_restart"))
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--csd-socket-path", default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument(
        "--kill-interval-iters",
        type=int,
        default=None,
        help="Kill after every N training iterations, once the matching RACER checkpoint is committed.",
    )
    parser.add_argument("--kill-count", type=int, default=None, help="Number of restart kills to inject.")
    parser.add_argument(
        "--post-kill-train-iters",
        type=int,
        default=None,
        help="Iterations to run after the final injected kill.",
    )
    parser.add_argument(
        "--kill-after-iter",
        type=int,
        default=None,
        help="Legacy single-restart alias. Prefer --kill-interval-iters.",
    )
    parser.add_argument("--first-train-iters", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume-train-iters", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--racer-buffer-size", type=int, default=1073741824)
    parser.add_argument(
        "--racer-payload-pool-prewarm-chunks",
        type=int,
        default=None,
        help="Training-process pinned payload buffers to preallocate before first save. "
        "Defaults to ceil(model max local payload bytes / --racer-buffer-size). Use 0 to disable.",
    )
    parser.add_argument("--csd-backend", choices=["native_pinned", "egm"], default="native_pinned")
    parser.add_argument("--csd-native-pinned-total-bytes", type=int, default=None)
    parser.add_argument("--csd-native-pinned-segment-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--csd-native-pinned-device", type=int, default=0)
    parser.add_argument("--csd-ready-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_csd(address: str | tuple[str, int], timeout: float = 30.0) -> None:
    sys.path.insert(0, str(RACER_ROOT))
    from racer.csd import CheckpointStorageDaemonClient

    client = CheckpointStorageDaemonClient(address, authkey="racer-csd")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            client.list_tags()
            return
        except OSError:
            time.sleep(0.2)
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"CSD address {address!r} did not accept authenticated requests within {timeout:.1f}s")


def run_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path(args.output_root) / f"{config['label']}_{stamp}"


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NCCL_DEBUG"] = env.get("NCCL_DEBUG", "WARN")
    env["TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING"] = "false"
    env["TORCHDYNAMO_DISABLE"] = env.get("TORCHDYNAMO_DISABLE", "1")
    env["PYTHONPATH"] = f"{RACER_ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    ptxas = Path("/usr/local/cuda-13.1/bin/ptxas")
    if ptxas.exists():
        env["PATH"] = f"{ptxas.parent}:{env.get('PATH', '')}"
        env["TRITON_PTXAS_PATH"] = str(ptxas)
    return env


def build_train_cmd(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    train_iters: int,
    save_interval: int,
    master_port: int,
    csd_socket_path: Path,
    checkpoint_dir: Path,
    manifest_dir: Path,
    tensorboard_dir: Path,
    profile_dir: Path,
) -> list[str]:
    racer_storage_backend = "csd_native_pinned" if args.csd_backend == "native_pinned" else "csd_egm"
    return [
        "torchrun",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "localhost",
        "--master_port",
        str(master_port),
        "pretrain_gpt.py",
        "--use-mcore-models",
        "--transformer-impl",
        "transformer_engine",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "4",
        "--num-layers",
        str(config["num_layers"]),
        "--hidden-size",
        str(config["hidden_size"]),
        "--ffn-hidden-size",
        str(config["ffn_hidden_size"]),
        "--num-attention-heads",
        str(config["num_attention_heads"]),
        "--seq-length",
        "1024",
        "--max-position-embeddings",
        "1024",
        "--attention-backend",
        "auto",
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--global-batch-size",
        str(args.global_batch_size or config["default_global_batch_size"]),
        "--train-iters",
        str(train_iters),
        "--lr",
        "1.5e-4",
        "--min-lr",
        "1.0e-5",
        "--lr-decay-style",
        "cosine",
        "--lr-warmup-iters",
        "1",
        "--weight-decay",
        "0.1",
        "--clip-grad",
        "1.0",
        "--bf16",
        "--no-bias-dropout-fusion",
        "--use-distributed-optimizer",
        "--ckpt-format",
        "torch",
        "--data-path",
        str(args.data_path),
        "--vocab-file",
        str(args.vocab_file),
        "--merge-file",
        str(args.merge_file),
        "--split",
        "949,50,1",
        "--save",
        str(checkpoint_dir),
        "--load",
        str(checkpoint_dir),
        "--tensorboard-dir",
        str(tensorboard_dir),
        "--log-interval",
        "1",
        "--save-interval",
        str(save_interval),
        "--eval-interval",
        "100000",
        "--eval-iters",
        "2",
        "--racer-checkpoint",
        "--racer-path",
        str(RACER_ROOT),
        "--racer-k",
        "3",
        "--racer-m",
        "1",
        "--racer-train-ranks",
        "0,1,2,3",
        "--racer-spare-ranks",
        "4",
        "--racer-buffer-size",
        str(args.racer_buffer_size),
        "--racer-payload-pool-prewarm-chunks",
        str(args.racer_payload_pool_prewarm_chunks),
        "--racer-retain-checkpoints",
        "1",
        "--racer-distributed-store",
        "--racer-storage-backend",
        racer_storage_backend,
        "--racer-csd-socket-path",
        str(csd_socket_path),
        "--racer-csd-authkey",
        "racer-csd",
        "--racer-manifest-dir",
        str(manifest_dir),
        "--racer-profile-dir",
        str(profile_dir),
    ]


def start_csd(
    csd_socket_path: Path,
    metadata_dir: Path,
    log_path: Path,
    env: dict[str, str],
    *,
    backend: str,
    native_pinned_total_bytes: int,
    native_pinned_segment_bytes: int,
    native_pinned_device: int,
    ready_timeout_seconds: float,
) -> subprocess.Popen:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csd_socket_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "racer.csd",
        "--socket-path",
        str(csd_socket_path),
        "--backend",
        str(backend),
        "--metadata-dir",
        str(metadata_dir),
    ]
    if backend == "native_pinned":
        cmd.extend(
            [
                "--native-pinned-total-bytes",
                str(int(native_pinned_total_bytes)),
                "--native-pinned-segment-bytes",
                str(int(native_pinned_segment_bytes)),
                "--native-pinned-device",
                str(int(native_pinned_device)),
            ]
        )
    proc = subprocess.Popen(
        cmd,
        cwd=str(RACER_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        wait_for_csd(str(csd_socket_path), timeout=float(ready_timeout_seconds))
    except BaseException:
        terminate_group(proc)
        raise
    return proc


def terminate_group(proc: subprocess.Popen, timeout: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_until_checkpoint_then_kill(
    *,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    target_tag: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    start = time.monotonic()
    matched_line = None
    stored_line = None
    target_iteration = iteration_from_checkpoint_tag(target_tag)
    save_done_prefix = None
    if target_iteration is not None:
        save_done_prefix = f"RACER save blocking time: iteration={target_iteration}"
    save_success_fragment = "successfully saved RACER memory checkpoint from iteration"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(MEGATRON_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if target_tag in line and "RACER distributed tensor-tree checkpoint stored" in line:
                    stored_line = line.strip()
                save_finished = False
                if stored_line is not None and save_done_prefix is not None and save_done_prefix in line:
                    save_finished = True
                elif (
                    stored_line is not None
                    and target_iteration is not None
                    and save_success_fragment in line
                    and str(target_iteration) in line
                ):
                    save_finished = True
                if save_finished:
                    matched_line = line.strip()
                    terminate_group(proc)
                    break
                if proc.poll() is not None:
                    break
                if time.monotonic() - start > timeout_seconds:
                    terminate_group(proc)
                    raise TimeoutError(f"timed out waiting for {target_tag}")
            for line in proc.stdout:
                log.write(line)
        finally:
            terminate_group(proc)
            returncode = proc.wait(timeout=30)
    if matched_line is None:
        detail = "store was observed but save completion was not" if stored_line else "store was not observed"
        raise RuntimeError(f"did not observe completed checkpoint tag {target_tag} ({detail}); returncode={returncode}")
    return {
        "returncode": returncode,
        "matched_line": matched_line,
        "stored_line": stored_line,
        "wall_ms_until_kill": (time.monotonic() - start) * 1000.0,
    }


def run_to_completion(
    *,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(MEGATRON_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            terminate_group(proc)
            raise TimeoutError(f"resume run timed out after {timeout_seconds}s")
        except BaseException:
            terminate_group(proc)
            raise
    return {
        "returncode": returncode,
        "wall_ms": (time.monotonic() - start) * 1000.0,
    }


def log_has_addr_in_use(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False
    return "EADDRINUSE" in text or "address already in use" in text


def parse_log(path: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for event, pattern in [
            ("store", STORE_RE),
            ("load", LOAD_RE),
            ("blocking", BLOCKING_RE),
            ("save_timer", SAVE_TIMER_RE),
        ]:
            match = pattern.search(line)
            if not match:
                continue
            row: dict[str, Any] = {"phase": phase, "event": event, "raw": line}
            for key, value in match.groupdict().items():
                if value is None:
                    row[key] = value
                elif key == "tag":
                    row[key] = value
                elif key in {"bytes", "local_bytes", "leaves", "chunks", "iteration"}:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def checkpoint_tag(iteration: int) -> str:
    return f"megatron:iter_{int(iteration):07d}"


def iteration_from_checkpoint_tag(tag: str) -> int | None:
    match = re.search(r"iter_([0-9]+)$", str(tag))
    if not match:
        return None
    return int(match.group(1))


def parse_iteration_times(path: Path, phase: str, save_interval: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ITER_TIME_RE.search(line)
        if not match:
            continue
        iteration = int(match.group("iteration"))
        rows.append(
            {
                "phase": phase,
                "iteration": iteration,
                "train_iters": int(match.group("train_iters")),
                "elapsed_time_per_iteration_ms": float(match.group("elapsed_ms")),
                "checkpoint_iteration": bool(save_interval > 0 and iteration % save_interval == 0),
                "raw": line,
            }
        )
    return rows


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_iteration_times(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        elapsed = row.get("elapsed_time_per_iteration_ms")
        if not isinstance(elapsed, (int, float)):
            continue
        phase = str(row.get("phase", ""))
        groups.setdefault((phase, "all"), []).append(float(elapsed))
        if not row.get("checkpoint_iteration"):
            groups.setdefault((phase, "non_checkpoint"), []).append(float(elapsed))
        groups.setdefault(("all_phases", "all"), []).append(float(elapsed))
        if not row.get("checkpoint_iteration"):
            groups.setdefault(("all_phases", "non_checkpoint"), []).append(float(elapsed))

    summary: list[dict[str, Any]] = []
    for (phase, bucket), values in sorted(groups.items()):
        if not values:
            continue
        ordered = sorted(values)
        summary.append(
            {
                "phase": phase,
                "bucket": bucket,
                "count": len(values),
                "mean_ms": sum(values) / len(values),
                "min_ms": ordered[0],
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "max_ms": ordered[-1],
            }
        )
    return summary


def read_profile_dir(path: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for profile in sorted(path.glob("rank_*.jsonl")):
        for line in profile.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["phase"] = phase
            rows.append(row)
    return rows


def parse_csd_profile_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    prefix = "[CSD_PROFILE] "
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if prefix not in line:
            continue
        payload = line.split(prefix, 1)[1]
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            continue
        profile = record.pop("profile", {}) or {}
        if isinstance(profile, dict):
            for key, value in profile.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    record[key] = value
        rows.append(record)
    return rows


def aggregate_csd_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    numeric_keys = [
        "daemon_allocate_ms",
        "checksum_ms",
        "sqlite_ms",
        "daemon_memcpy_ms_cuda_event",
        "daemon_memcpy_ms_wall",
        "daemon_event_wait_ms",
        "daemon_ipc_open_us",
    ]
    for row in rows:
        tag = str(row.get("tag", ""))
        if not tag:
            continue
        item = grouped.setdefault(tag, {"tag": tag, "op_count": 0})
        item["op_count"] += 1
        source = str(row.get("daemon_allocate_source", ""))
        if source:
            key = f"allocate_source_{source}_count"
            item[key] = int(item.get(key, 0)) + 1
        for key in numeric_keys:
            value = row.get(key)
            if isinstance(value, (int, float)):
                item[f"{key}_sum"] = float(item.get(f"{key}_sum", 0.0)) + float(value)
                item[f"{key}_max"] = max(float(item.get(f"{key}_max", 0.0)), float(value))
    return list(grouped.values())


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fmt_ms(value: Any) -> str:
    numeric = _to_float(value)
    return "-" if numeric is None else f"{numeric:.2f}"


def _fmt_seconds(value_ms: Any) -> str:
    numeric = _to_float(value_ms)
    return "-" if numeric is None else f"{numeric / 1000.0:.2f}"


def _fmt_gib(value: Any) -> str:
    numeric = _to_int(value)
    return "-" if numeric is None else f"{numeric / 1024**3:.2f}"


def _tag_iter(tag: Any) -> str:
    iteration = iteration_from_checkpoint_tag(str(tag or ""))
    return "-" if iteration is None else str(iteration)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_store_events(stores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = [("all", stores)]
    phases = sorted({str(row.get("phase", "")) for row in stores if row.get("phase")})
    groups.extend((phase, [row for row in stores if str(row.get("phase", "")) == phase]) for phase in phases)
    summary: list[dict[str, Any]] = []
    for phase, rows in groups:
        if not rows:
            continue
        store_values = [value for value in (_to_float(row.get("store_ms")) for row in rows) if value is not None]
        wait_values = [value for value in (_to_float(row.get("storage_wait_ms")) for row in rows) if value is not None]
        slowest = max(rows, key=lambda row: _to_float(row.get("store_ms")) or 0.0)
        payload_bytes = _to_int(rows[0].get("local_bytes"))
        chunks = _to_int(rows[0].get("chunks"))
        summary.append(
            {
                "phase": phase,
                "count": len(rows),
                "store_mean_ms": _mean(store_values),
                "store_p50_ms": _percentile(store_values, 0.50),
                "store_max_ms": max(store_values) if store_values else None,
                "wait_p50_ms": _percentile(wait_values, 0.50),
                "wait_max_ms": max(wait_values) if wait_values else None,
                "payload_gib": None if payload_bytes is None else payload_bytes / 1024**3,
                "chunks": chunks,
                "slowest_iter": _tag_iter(slowest.get("tag")),
            }
        )
    return summary


def summarize_csd_profile_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "groups": len(rows),
        "op_count": 0,
        "dynamic_cudaHostAlloc": 0,
        "pool_bump": 0,
        "free_list": 0,
        "allocate_ms_sum": 0.0,
        "checksum_ms_sum": 0.0,
        "sqlite_ms_sum": 0.0,
        "copy_wall_ms_sum": 0.0,
    }
    for row in rows:
        totals["op_count"] += int(row.get("op_count", 0) or 0)
        totals["dynamic_cudaHostAlloc"] += int(row.get("allocate_source_dynamic_cudaHostAlloc_count", 0) or 0)
        totals["pool_bump"] += int(row.get("allocate_source_pool_bump_count", 0) or 0)
        totals["free_list"] += int(row.get("allocate_source_free_list_count", 0) or 0)
        totals["allocate_ms_sum"] += float(row.get("daemon_allocate_ms_sum", 0.0) or 0.0)
        totals["checksum_ms_sum"] += float(row.get("checksum_ms_sum", 0.0) or 0.0)
        totals["sqlite_ms_sum"] += float(row.get("sqlite_ms_sum", 0.0) or 0.0)
        totals["copy_wall_ms_sum"] += float(row.get("daemon_memcpy_ms_wall_sum", 0.0) or 0.0)
    return totals


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def nvidia_smi_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except Exception as exc:
        return f"nvidia-smi failed: {exc}\n"


def _missing_required_paths(paths: dict[str, Path], data_path: Path) -> list[str]:
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    data_bin = Path(str(data_path) + ".bin")
    data_idx = Path(str(data_path) + ".idx")
    if not data_path.exists() and not (data_bin.exists() and data_idx.exists()):
        missing.append(f"data_path={data_path} (expected prefix, or {data_bin.name} + {data_idx.name})")
    return missing


def write_report(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    run_results: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    csd_profile_summary: list[dict[str, Any]],
    iteration_summary: list[dict[str, Any]],
    kill_target_tags: list[str],
    final_tag: str,
) -> None:
    stores = [row for row in parsed_rows if row.get("event") == "store"]
    loads = [row for row in parsed_rows if row.get("event") == "load"]
    restart_load_checks = [
        bool(result.get("expected_load_tag"))
        and any(
            row.get("event") == "load"
            and row.get("phase") == result.get("phase")
            and row.get("tag") == result.get("expected_load_tag")
            for row in loads
        )
        for result in run_results
        if result.get("expected_load_tag")
    ]
    restart_loaded = all(restart_load_checks) if restart_load_checks else True
    final_store = any(row.get("event") == "store" and row.get("tag") == final_tag for row in parsed_rows)
    store_summary = summarize_store_events(stores)
    csd_totals = summarize_csd_profile_totals(csd_profile_summary)
    first_store_by_phase: dict[str, str] = {}
    for row in stores:
        phase = str(row.get("phase", ""))
        first_store_by_phase.setdefault(phase, str(row.get("tag", "")))
    text = [
        f"# Megatron RACER CSD 重启测试: {args.model}",
        "",
        f"- 保存间隔: 每 `{args.resolved_save_interval}` 个 iteration 保存一次",
        f"- 杀进程点: `{', '.join(_tag_iter(tag) for tag in kill_target_tags)}`",
        f"- 最终 checkpoint: iter `{_tag_iter(final_tag)}`",
        f"- 三次恢复 load 是否都观察到: `{restart_loaded}`",
        f"- 最终 checkpoint 是否观察到: `{final_store}`",
        f"- 原始事件数量: `{len(parsed_rows)}`；完整明细见 `parsed_log_events.csv`、`iteration_times.csv`、`csd_profile_summary.csv`",
        "",
        "## 运行分段",
        "",
        "| 阶段 | 动作 | 目标 iter | 期望 load iter | load 观察到 | 返回码 | 墙钟时间 s | 日志 |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for result in run_results:
        expected_load_tag = result.get("expected_load_tag")
        load_observed = (
            any(
                row.get("event") == "load"
                and row.get("phase") == result.get("phase")
                and row.get("tag") == expected_load_tag
                for row in loads
            )
            if expected_load_tag
            else True
        )
        wall_ms = result.get("wall_ms_until_kill", result.get("wall_ms", 0.0))
        text.append(
            f"| {result.get('phase')} | {result.get('action')} | {_tag_iter(result.get('target_tag'))} | "
            f"{_tag_iter(expected_load_tag)} | `{load_observed}` | "
            f"{int(result.get('returncode', 0))} | {_fmt_seconds(wall_ms)} | `{Path(str(result.get('log_path', ''))).name}` |"
        )
    text.extend(
        [
            "",
            "## 每迭代训练耗时摘要",
            "",
            "这里直接来自 Megatron 原始日志的 `elapsed time per iteration (ms)`。`non_checkpoint` 是排除保存点 iteration 后的训练耗时。",
            "",
            "| 阶段 | 范围 | 样本数 | 平均 ms | P50 ms | P95 ms | 最大 ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in iteration_summary:
        text.append(
            f"| {row.get('phase')} | {row.get('bucket')} | {int(row.get('count', 0))} | "
            f"{_fmt_ms(row.get('mean_ms'))} | "
            f"{_fmt_ms(row.get('p50_ms'))} | "
            f"{_fmt_ms(row.get('p95_ms'))} | "
            f"{_fmt_ms(row.get('max_ms'))} |"
        )
    text.extend(
        [
            "",
            "## 保存耗时摘要",
            "",
            "这张表只保留人需要看的字段：保存总耗时、CSD wait、payload 大小、chunks，以及每段最慢保存点。完整横向明细在 `parsed_log_events.csv`。",
            "",
            "| 阶段 | 保存次数 | store 平均 ms | store P50 ms | store 最大 ms | wait P50 ms | wait 最大 ms | payload GiB | chunks | 最慢 iter |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in store_summary:
        text.append(
            f"| {row.get('phase')} | {int(row.get('count', 0))} | "
            f"{_fmt_ms(row.get('store_mean_ms'))} | "
            f"{_fmt_ms(row.get('store_p50_ms'))} | "
            f"{_fmt_ms(row.get('store_max_ms'))} | "
            f"{_fmt_ms(row.get('wait_p50_ms'))} | "
            f"{_fmt_ms(row.get('wait_max_ms'))} | "
            f"{float(row.get('payload_gib') or 0.0):.2f} | "
            f"{row.get('chunks') if row.get('chunks') is not None else '-'} | "
            f"{row.get('slowest_iter')} |"
        )
    text.extend(
        [
            "",
            "## 保存事件短表",
            "",
            "| 阶段 | iter | store ms | CSD wait ms | payload GiB | chunks | 说明 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in stores:
        tag = str(row.get("tag", ""))
        note = ""
        if tag in kill_target_tags:
            note = "杀进程点"
        elif tag == final_tag:
            note = "最终点"
        elif tag == first_store_by_phase.get(str(row.get("phase", ""))) and str(row.get("phase", "")) != "run_00":
            note = "恢复后首次保存"
        text.append(
            f"| {row.get('phase')} | {_tag_iter(tag)} | "
            f"{_fmt_ms(row.get('store_ms'))} | "
            f"{_fmt_ms(row.get('storage_wait_ms'))} | "
            f"{_fmt_gib(row.get('local_bytes'))} | "
            f"{_to_int(row.get('chunks')) if _to_int(row.get('chunks')) is not None else '-'} | "
            f"{note} |"
        )
    text.extend(
        [
            "",
            "## 恢复读取事件",
            "",
            "| 阶段 | load iter | 总耗时 ms | fetch ms | read wait ms | materialize ms | tree decode ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in loads:
        text.append(
            f"| {row.get('phase')} | {_tag_iter(row.get('tag'))} | "
            f"{_fmt_ms(row.get('total_ms'))} | "
            f"{_fmt_ms(row.get('racer_fetch_ms'))} | "
            f"{_fmt_ms(row.get('load_read_wait_ms'))} | "
            f"{_fmt_ms(row.get('tensor_materialize_ms'))} | "
            f"{_fmt_ms(row.get('tree_decode_ms'))} |"
        )
    text.extend(
        [
            "",
            "## CSD 后端摘要",
            "",
            "这里只看内存来源是否稳定。`dynamic_cudaHostAlloc` 越低越好；为 0 表示本轮没有临时 pinned memory 动态分配。",
            "",
            "| profile 组数 | op 数 | dynamic_cudaHostAlloc | pool_bump | free_list | allocate ms | checksum ms | sqlite ms | copy wall ms |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {int(csd_totals.get('groups', 0))} | "
            f"{int(csd_totals.get('op_count', 0))} | "
            f"{int(csd_totals.get('dynamic_cudaHostAlloc', 0))} | "
            f"{int(csd_totals.get('pool_bump', 0))} | "
            f"{int(csd_totals.get('free_list', 0))} | "
            f"{float(csd_totals.get('allocate_ms_sum', 0.0)):.2f} | "
            f"{float(csd_totals.get('checksum_ms_sum', 0.0)):.2f} | "
            f"{float(csd_totals.get('sqlite_ms_sum', 0.0)):.2f} | "
            f"{float(csd_totals.get('copy_wall_ms_sum', 0.0)):.2f} |",
        ]
    )
    text.append("")
    out_dir.joinpath("report.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    global MEGATRON_ROOT, RACER_ROOT
    args = parse_args()
    RACER_ROOT = Path(args.racer_root).expanduser().resolve()
    MEGATRON_ROOT = Path(args.megatron_root).expanduser().resolve()
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    args.vocab_file = str(Path(args.vocab_file).expanduser().resolve())
    args.merge_file = str(Path(args.merge_file).expanduser().resolve())
    args.output_root = str(Path(args.output_root).expanduser().resolve())
    required_paths = {
        "racer_root": RACER_ROOT,
        "megatron_root": MEGATRON_ROOT,
        "megatron_entrypoint": MEGATRON_ROOT / "pretrain_gpt.py",
        "vocab_file": Path(args.vocab_file),
        "merge_file": Path(args.merge_file),
    }
    missing = _missing_required_paths(required_paths, Path(args.data_path))
    if missing:
        raise FileNotFoundError(
            "Missing required Megatron/RACER test paths. Override with --megatron-root, "
            "--racer-root, --data-path, --vocab-file, or --merge-file. Missing: "
            + "; ".join(missing)
        )
    config = MODEL_CONFIGS[args.model]
    if args.racer_payload_pool_prewarm_chunks is None:
        max_local_payload = int(config.get("default_max_local_payload_bytes", 0) or 0)
        args.racer_payload_pool_prewarm_chunks = (
            (max_local_payload + int(args.racer_buffer_size) - 1) // int(args.racer_buffer_size)
            if max_local_payload > 0 and int(args.racer_buffer_size) > 0
            else 0
        )
    save_interval = args.save_interval or int(config["default_save_interval"])
    if args.kill_after_iter is not None and args.kill_interval_iters is None:
        kill_interval_iters = int(args.kill_after_iter)
    else:
        kill_interval_iters = args.kill_interval_iters or int(config["default_kill_interval_iters"])
    if args.resume_train_iters is not None and args.post_kill_train_iters is None:
        post_kill_train_iters = int(args.resume_train_iters) - kill_interval_iters
    else:
        post_kill_train_iters = args.post_kill_train_iters or int(config["default_post_kill_train_iters"])
    kill_count = args.kill_count if args.kill_count is not None else int(config["default_kill_count"])
    if save_interval <= 0:
        raise ValueError("--save-interval must be positive")
    if kill_interval_iters <= 0:
        raise ValueError("--kill-interval-iters must be positive")
    if kill_count < 0:
        raise ValueError("--kill-count must be non-negative")
    if post_kill_train_iters <= 0:
        raise ValueError("--post-kill-train-iters must be positive")
    if kill_interval_iters % save_interval != 0:
        raise ValueError("--kill-interval-iters must be a multiple of --save-interval")
    final_train_iters = kill_interval_iters * kill_count + post_kill_train_iters
    if final_train_iters % save_interval != 0:
        raise ValueError("final train iters must be a multiple of --save-interval")
    kill_target_iters = [kill_interval_iters * (idx + 1) for idx in range(kill_count)]
    kill_target_tags = [checkpoint_tag(iteration) for iteration in kill_target_iters]
    final_tag = checkpoint_tag(final_train_iters)
    args.resolved_save_interval = save_interval
    master_port = args.master_port or (29500 if args.dry_run else free_port())
    out_dir = run_dir(args, config)
    if out_dir.exists() and not args.keep_output:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csd_socket_path = Path(args.csd_socket_path) if args.csd_socket_path else out_dir / "csd.sock"

    checkpoint_dir = out_dir / "checkpoints"
    manifest_dir = out_dir / "racer_manifests"
    tensorboard_dir = out_dir / "tensorboard"
    csd_metadata_dir = out_dir / "csd_metadata"
    csd_native_pinned_total_bytes = int(
        args.csd_native_pinned_total_bytes
        if args.csd_native_pinned_total_bytes is not None
        else config["default_csd_native_pinned_total_bytes"]
    )
    csd_native_pinned_segment_bytes = int(args.csd_native_pinned_segment_bytes)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = base_env(args)
    run_specs: list[dict[str, Any]] = []
    for index in range(kill_count + 1):
        phase = f"run_{index:02d}"
        target_iter = kill_target_iters[index] if index < kill_count else None
        expected_load_iter = kill_target_iters[index - 1] if index > 0 else None
        run_master_port = master_port + index if args.dry_run else free_port()
        profile_dir = out_dir / f"profiles_{phase}"
        cmd = build_train_cmd(
            args=args,
            config=config,
            train_iters=final_train_iters,
            save_interval=save_interval,
            master_port=run_master_port,
            csd_socket_path=csd_socket_path,
            checkpoint_dir=checkpoint_dir,
            manifest_dir=manifest_dir,
            tensorboard_dir=tensorboard_dir,
            profile_dir=profile_dir,
        )
        run_specs.append(
            {
                "phase": phase,
                "action": "kill" if target_iter is not None else "complete",
                "train_iters": final_train_iters,
                "target_iter": target_iter,
                "target_tag": checkpoint_tag(target_iter) if target_iter is not None else final_tag,
                "expected_load_tag": checkpoint_tag(expected_load_iter) if expected_load_iter is not None else None,
                "master_port": run_master_port,
                "log_path": str(out_dir / f"{phase}.log"),
                "profile_dir": str(profile_dir),
                "cmd": cmd,
            }
        )
    metadata = {
        "model": args.model,
        "config": config,
        "save_interval": save_interval,
        "kill_interval_iters": kill_interval_iters,
        "kill_count": kill_count,
        "post_kill_train_iters": post_kill_train_iters,
        "final_train_iters": final_train_iters,
        "kill_target_iters": kill_target_iters,
        "kill_target_tags": kill_target_tags,
        "target_tag": kill_target_tags[0] if kill_target_tags else final_tag,
        "resume_tag": final_tag,
        "final_tag": final_tag,
        "master_port": master_port,
        "csd_socket_path": str(csd_socket_path),
        "racer_root": str(RACER_ROOT),
        "megatron_root": str(MEGATRON_ROOT),
        "data_path": str(args.data_path),
        "vocab_file": str(args.vocab_file),
        "merge_file": str(args.merge_file),
        "csd_backend": args.csd_backend,
        "csd_native_pinned_total_bytes": csd_native_pinned_total_bytes,
        "csd_native_pinned_segment_bytes": csd_native_pinned_segment_bytes,
        "csd_native_pinned_device": int(args.csd_native_pinned_device),
        "csd_ready_timeout_seconds": float(args.csd_ready_timeout_seconds),
        "racer_csd_checksum_type": env.get("RACER_CSD_CHECKSUM_TYPE", ""),
        "racer_csd_manifest_update_mode": env.get("RACER_CSD_MANIFEST_UPDATE_MODE", ""),
        "racer_csd_profile_log": env.get("RACER_CSD_PROFILE_LOG", ""),
        "racer_payload_pool_prewarm_chunks": int(args.racer_payload_pool_prewarm_chunks),
        "run_specs": run_specs,
        "first_cmd": run_specs[0]["cmd"],
        "resume_cmd": run_specs[-1]["cmd"],
        "cuda_visible_devices": args.cuda_visible_devices,
        "nvidia_smi_before": nvidia_smi_snapshot(),
    }
    out_dir.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return

    csd = start_csd(
        csd_socket_path,
        csd_metadata_dir,
        out_dir / "csd.log",
        env,
        backend=args.csd_backend,
        native_pinned_total_bytes=csd_native_pinned_total_bytes,
        native_pinned_segment_bytes=csd_native_pinned_segment_bytes,
        native_pinned_device=int(args.csd_native_pinned_device),
        ready_timeout_seconds=float(args.csd_ready_timeout_seconds),
    )
    run_results: list[dict[str, Any]] = []
    try:
        for index, spec in enumerate(run_specs):
            log_path = Path(spec["log_path"])
            profile_dir = Path(spec["profile_dir"])
            for attempt in range(5):
                if attempt:
                    spec["master_port"] = free_port()
                    spec["cmd"] = build_train_cmd(
                        args=args,
                        config=config,
                        train_iters=final_train_iters,
                        save_interval=save_interval,
                        master_port=int(spec["master_port"]),
                        csd_socket_path=csd_socket_path,
                        checkpoint_dir=checkpoint_dir,
                        manifest_dir=manifest_dir,
                        tensorboard_dir=tensorboard_dir,
                        profile_dir=profile_dir,
                    )
                    metadata["run_specs"][index]["master_port"] = spec["master_port"]
                    metadata["run_specs"][index]["cmd"] = spec["cmd"]
                if spec["action"] == "kill":
                    result = run_until_checkpoint_then_kill(
                        cmd=spec["cmd"],
                        env=env,
                        log_path=log_path,
                        target_tag=str(spec["target_tag"]),
                        timeout_seconds=args.timeout_seconds,
                    )
                else:
                    result = run_to_completion(
                        cmd=spec["cmd"],
                        env=env,
                        log_path=log_path,
                        timeout_seconds=args.timeout_seconds,
                    )
                if result["returncode"] == 0 or not log_has_addr_in_use(log_path):
                    break
            else:
                raise RuntimeError(f"{spec['phase']} failed with EADDRINUSE after 5 master-port attempts")
            result.update(
                {
                    "phase": spec["phase"],
                    "action": spec["action"],
                    "train_iters": spec["train_iters"],
                    "target_tag": spec["target_tag"],
                    "expected_load_tag": spec["expected_load_tag"],
                    "master_port": spec["master_port"],
                    "log_path": Path(spec["log_path"]).name,
                    "profile_dir": Path(spec["profile_dir"]).name,
                }
            )
            run_results.append(result)
            if spec["action"] == "kill":
                time.sleep(5.0)
    finally:
        terminate_group(csd)
    out_dir.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    parsed_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for spec in run_specs:
        phase = str(spec["phase"])
        parsed_rows.extend(parse_log(Path(spec["log_path"]), phase))
        profile_rows.extend(read_profile_dir(Path(spec["profile_dir"]), phase))
        iteration_rows.extend(parse_iteration_times(Path(spec["log_path"]), phase, save_interval))
    iteration_summary = summarize_iteration_times(iteration_rows)
    csd_profile_rows = parse_csd_profile_log(out_dir / "csd.log")
    csd_profile_summary = aggregate_csd_profiles(csd_profile_rows)
    write_csv(out_dir / "parsed_log_events.csv", parsed_rows)
    write_csv(out_dir / "iteration_times.csv", iteration_rows)
    write_csv(out_dir / "iteration_time_summary.csv", iteration_summary)
    write_csv(out_dir / "per_rank_profile_events.csv", profile_rows)
    write_csv(out_dir / "csd_profile_events.csv", csd_profile_rows)
    write_csv(out_dir / "csd_profile_summary.csv", csd_profile_summary)
    restart_load_checks = [
        {
            "phase": result.get("phase"),
            "expected_load_tag": result.get("expected_load_tag"),
            "observed": any(
                row.get("event") == "load"
                and row.get("phase") == result.get("phase")
                and row.get("tag") == result.get("expected_load_tag")
                for row in parsed_rows
            ),
        }
        for result in run_results
        if result.get("expected_load_tag")
    ]
    final_checkpoint_observed = any(row.get("event") == "store" and row.get("tag") == final_tag for row in parsed_rows)
    summary = {
        "metadata": metadata,
        "run_results": run_results,
        "first_run": run_results[0] if run_results else {},
        "resume_run": run_results[-1] if run_results else {},
        "nvidia_smi_after": nvidia_smi_snapshot(),
        "csd_profile_summary": csd_profile_summary,
        "iteration_time_summary": iteration_summary,
        "restart_load_checks": restart_load_checks,
        "restart_load_observed": all(item["observed"] for item in restart_load_checks)
        if restart_load_checks
        else True,
        "post_resume_checkpoint_observed": final_checkpoint_observed,
    }
    out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(
        out_dir=out_dir,
        args=args,
        run_results=run_results,
        parsed_rows=parsed_rows,
        profile_rows=profile_rows,
        csd_profile_summary=csd_profile_summary,
        iteration_summary=iteration_summary,
        kill_target_tags=kill_target_tags,
        final_tag=final_tag,
    )
    print(f"wrote {out_dir}")
    print(json.dumps(summary, indent=2))
    final_returncode = run_results[-1].get("returncode") if run_results else None
    if final_returncode != 0:
        raise RuntimeError(f"final run failed with returncode={final_returncode}")
    if not summary["restart_load_observed"]:
        raise RuntimeError(f"missing expected restart load: {restart_load_checks}")
    if not summary["post_resume_checkpoint_observed"]:
        raise RuntimeError(f"missing final checkpoint store for {final_tag}")


if __name__ == "__main__":
    main()
