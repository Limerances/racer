#!/usr/bin/env python3
"""Run Megatron RACER CSD restart tests for GPT2-size models.

The harness:
1. starts a RACER Checkpoint Storage Daemon,
2. launches Megatron training until a target RACER checkpoint is committed,
3. terminates the training process group,
4. launches Megatron again against the same CSD and manifest directory,
5. verifies that the resumed process loads the resident checkpoint and keeps training,
6. writes raw logs plus parsed CSV/JSON/Markdown summaries.
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
        "default_save_interval": 2,
        "default_kill_after_iter": 2,
        "default_resume_train_iters": 4,
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
        "default_save_interval": 1,
        "default_kill_after_iter": 1,
        "default_resume_train_iters": 2,
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
ITER_RE = re.compile(r" iteration\s+(?P<iteration>[0-9]+)[/ ]")


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
    parser.add_argument("--kill-after-iter", type=int, default=None)
    parser.add_argument("--first-train-iters", type=int, default=None)
    parser.add_argument("--resume-train-iters", type=int, default=None)
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
        raise RuntimeError(f"did not observe committed checkpoint tag {target_tag}; returncode={returncode}")
    return {
        "returncode": returncode,
        "matched_line": matched_line,
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
    first: dict[str, Any],
    resume: dict[str, Any],
    parsed_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    csd_profile_summary: list[dict[str, Any]],
    target_tag: str,
    resume_tag: str,
) -> None:
    stores = [row for row in parsed_rows if row.get("event") == "store"]
    loads = [row for row in parsed_rows if row.get("event") == "load"]
    resume_loaded = any(row.get("event") == "load" and row.get("tag") == target_tag for row in parsed_rows)
    post_resume_store = any(row.get("event") == "store" and row.get("tag") == resume_tag for row in parsed_rows)
    text = [
        f"# Megatron RACER CSD restart test: {args.model}",
        "",
        f"- target resident checkpoint: `{target_tag}`",
        f"- post-resume checkpoint expected: `{resume_tag}`",
        f"- first run returncode after kill: `{first['returncode']}`",
        f"- first run wall until kill: `{first['wall_ms_until_kill']:.2f} ms`",
        f"- resume run returncode: `{resume['returncode']}`",
        f"- resume wall time: `{resume['wall_ms']:.2f} ms`",
        f"- restart load observed: `{resume_loaded}`",
        f"- post-resume training checkpoint observed: `{post_resume_store}`",
        f"- parsed log events: `{len(parsed_rows)}`",
        f"- per-rank profile events: `{len(profile_rows)}`",
        f"- CSD profile groups: `{len(csd_profile_summary)}`",
        "",
        "## Store events",
        "",
        "| phase | tag | store ms | command ms | racer calls ms | chunk max ms | storage ms | enqueue ms | wait ms | client put ms | client rpc ms | daemon put ms | backend write ms | commit ms | bytes | chunks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stores:
        text.append(
            f"| {row.get('phase')} | `{row.get('tag')}` | "
            f"{float(row.get('store_ms', 0.0)):.2f} | "
            f"{float(row.get('command_ms') or 0.0):.2f} | "
            f"{float(row.get('racer_calls_ms') or 0.0):.2f} | "
            f"{float(row.get('chunk_store_max_ms') or 0.0):.2f} | "
            f"{float(row.get('storage_ms') or 0.0):.2f} | "
            f"{float(row.get('storage_enqueue_ms') or 0.0):.2f} | "
            f"{float(row.get('storage_wait_ms') or 0.0):.2f} | "
            f"{float(row.get('client_put_ms') or 0.0):.2f} | "
            f"{float(row.get('client_rpc_ms') or 0.0):.2f} | "
            f"{float(row.get('daemon_put_ms') or 0.0):.2f} | "
            f"{float(row.get('daemon_backend_write_ms') or 0.0):.2f} | "
            f"{float(row.get('storage_commit_ms') or 0.0):.2f} | "
            f"{int(row.get('bytes', 0))} | {int(row.get('chunks', 0))} |"
        )
    text.extend(
        [
            "",
            "## Load events",
            "",
            "| phase | tag | total ms | racer fetch ms | materialize ms | tree decode ms | runtime prewarm ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in loads:
        text.append(
            f"| {row.get('phase')} | `{row.get('tag')}` | "
            f"{float(row.get('total_ms', 0.0)):.2f} | "
            f"{float(row.get('racer_fetch_ms', 0.0)):.2f} | "
            f"{float(row.get('tensor_materialize_ms', 0.0)):.2f} | "
            f"{float(row.get('tree_decode_ms', 0.0)):.2f} | "
            f"{float(row.get('runtime_prewarm_after_load_ms') or 0.0):.2f} |"
        )
    text.extend(
        [
            "",
            "## CSD profile summary",
            "",
            "| tag | ops | alloc ms sum | checksum ms sum | sqlite ms sum | copy event ms sum | dynamic alloc ops | pool bump ops | free-list ops |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in csd_profile_summary:
        text.append(
            f"| `{row.get('tag')}` | "
            f"{int(row.get('op_count', 0))} | "
            f"{float(row.get('daemon_allocate_ms_sum', 0.0)):.2f} | "
            f"{float(row.get('checksum_ms_sum', 0.0)):.2f} | "
            f"{float(row.get('sqlite_ms_sum', 0.0)):.2f} | "
            f"{float(row.get('daemon_memcpy_ms_cuda_event_sum', 0.0)):.2f} | "
            f"{int(row.get('allocate_source_dynamic_cudaHostAlloc_count', 0))} | "
            f"{int(row.get('allocate_source_pool_bump_count', 0))} | "
            f"{int(row.get('allocate_source_free_list_count', 0))} |"
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
    kill_after_iter = args.kill_after_iter or int(config["default_kill_after_iter"])
    resume_train_iters = args.resume_train_iters or int(config["default_resume_train_iters"])
    first_train_iters = args.first_train_iters or resume_train_iters
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
    first_profile_dir = out_dir / "profiles_first"
    resume_profile_dir = out_dir / "profiles_resume"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = base_env(args)
    target_tag = f"megatron:iter_{kill_after_iter:07d}"
    resume_tag = f"megatron:iter_{resume_train_iters:07d}"
    first_cmd = build_train_cmd(
        args=args,
        config=config,
        train_iters=first_train_iters,
        save_interval=save_interval,
        master_port=master_port,
        csd_socket_path=csd_socket_path,
        checkpoint_dir=checkpoint_dir,
        manifest_dir=manifest_dir,
        tensorboard_dir=tensorboard_dir,
        profile_dir=first_profile_dir,
    )
    resume_master_port = master_port + 1 if args.dry_run else free_port()
    resume_cmd = build_train_cmd(
        args=args,
        config=config,
        train_iters=resume_train_iters,
        save_interval=save_interval,
        master_port=resume_master_port,
        csd_socket_path=csd_socket_path,
        checkpoint_dir=checkpoint_dir,
        manifest_dir=manifest_dir,
        tensorboard_dir=tensorboard_dir,
        profile_dir=resume_profile_dir,
    )
    metadata = {
        "model": args.model,
        "config": config,
        "target_tag": target_tag,
        "resume_tag": resume_tag,
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
        "first_cmd": first_cmd,
        "resume_cmd": resume_cmd,
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
    try:
        first = run_until_checkpoint_then_kill(
            cmd=first_cmd,
            env=env,
            log_path=out_dir / "first_run.log",
            target_tag=target_tag,
            timeout_seconds=args.timeout_seconds,
        )
        time.sleep(5.0)
        resume_log = out_dir / "resume_run.log"
        for attempt in range(5):
            resume = run_to_completion(
                cmd=resume_cmd,
                env=env,
                log_path=resume_log,
                timeout_seconds=args.timeout_seconds,
            )
            if resume["returncode"] == 0 or not log_has_addr_in_use(resume_log):
                break
            resume_master_port = free_port()
            resume_cmd = build_train_cmd(
                args=args,
                config=config,
                train_iters=resume_train_iters,
                save_interval=save_interval,
                master_port=resume_master_port,
                csd_socket_path=csd_socket_path,
                checkpoint_dir=checkpoint_dir,
                manifest_dir=manifest_dir,
                tensorboard_dir=tensorboard_dir,
                profile_dir=resume_profile_dir,
            )
            metadata["resume_cmd"] = resume_cmd
        else:
            raise RuntimeError("resume run failed with EADDRINUSE after 5 master-port attempts")
    finally:
        terminate_group(csd)

    parsed_rows = parse_log(out_dir / "first_run.log", "first") + parse_log(out_dir / "resume_run.log", "resume")
    profile_rows = read_profile_dir(first_profile_dir, "first") + read_profile_dir(resume_profile_dir, "resume")
    csd_profile_rows = parse_csd_profile_log(out_dir / "csd.log")
    csd_profile_summary = aggregate_csd_profiles(csd_profile_rows)
    write_csv(out_dir / "parsed_log_events.csv", parsed_rows)
    write_csv(out_dir / "per_rank_profile_events.csv", profile_rows)
    write_csv(out_dir / "csd_profile_events.csv", csd_profile_rows)
    write_csv(out_dir / "csd_profile_summary.csv", csd_profile_summary)
    summary = {
        "metadata": metadata,
        "first_run": first,
        "resume_run": resume,
        "nvidia_smi_after": nvidia_smi_snapshot(),
        "csd_profile_summary": csd_profile_summary,
        "restart_load_observed": any(
            row.get("event") == "load" and row.get("tag") == target_tag for row in parsed_rows
        ),
        "post_resume_checkpoint_observed": any(
            row.get("event") == "store" and row.get("tag") == resume_tag for row in parsed_rows
        ),
    }
    out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(
        out_dir=out_dir,
        args=args,
        first=first,
        resume=resume,
        parsed_rows=parsed_rows,
        profile_rows=profile_rows,
        csd_profile_summary=csd_profile_summary,
        target_tag=target_tag,
        resume_tag=resume_tag,
    )
    print(f"wrote {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
