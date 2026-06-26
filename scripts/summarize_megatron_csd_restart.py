#!/usr/bin/env python3
"""Summarize a Megatron RACER CSD restart test result directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ITER_TIME_RE = re.compile(
    r"iteration\s+(?P<iteration>[0-9]+).*?elapsed time per iteration \(ms\):\s*(?P<ms>[0-9.]+)"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f} ms"


def _fmt_gib(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1024**3:.2f} GiB"


def _iter_times(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    rows: list[tuple[int, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ITER_TIME_RE.search(line)
        if match:
            rows.append((int(match.group("iteration")), float(match.group("ms"))))
    return rows


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("(none)")
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, item in enumerate(row):
            widths[idx] = max(widths[idx], len(item))
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(item.ljust(widths[idx]) for idx, item in enumerate(row)))


def summarize(result_dir: Path) -> None:
    result_dir = result_dir.resolve()
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    events = _read_csv(result_dir / "parsed_log_events.csv")
    csd = _read_csv(result_dir / "csd_profile_summary.csv")

    print(f"Result directory: {result_dir}")
    if summary:
        metadata = summary.get("metadata", {}) if isinstance(summary.get("metadata"), dict) else {}
        print(
            f"Model: {metadata.get('model', summary.get('model'))}  "
            f"target_tag: {metadata.get('target_tag', summary.get('target_tag'))}  "
            f"resume_tag: {metadata.get('resume_tag', summary.get('resume_tag'))}"
        )
        print(
            "Restart checks: "
            f"load_observed={summary.get('restart_load_observed')}  "
            f"post_resume_checkpoint_observed={summary.get('post_resume_checkpoint_observed')}"
        )
        print(
            "Run wall time: "
            f"first_until_kill={_fmt_ms(_float(summary.get('first_run', {}), 'wall_ms_until_kill'))}  "
            f"resume={_fmt_ms(_float(summary.get('resume_run', {}), 'wall_ms'))}"
        )
    print()

    print("Normal training iteration time from raw Megatron logs")
    iter_rows: list[list[str]] = []
    for phase, name in (("first", "first_run.log"), ("resume", "resume_run.log")):
        for iteration, ms in _iter_times(result_dir / name):
            iter_rows.append([phase, str(iteration), f"{ms:.2f} ms"])
    _print_table(["phase", "iteration", "elapsed_per_iter"], iter_rows)
    print()

    print("Megatron checkpoint blocking events")
    blocking_rows = []
    for row in events:
        if row.get("event") != "blocking":
            continue
        blocking_rows.append(
            [
                row.get("phase", ""),
                str(row.get("iteration", "")),
                _fmt_ms(_float(row, "save_checkpoint_fn_total_ms")),
                _fmt_ms(_float(row, "racer_adapter_save_ms")),
                _fmt_ms(_float(row, "state_dict_ms")),
                _fmt_ms(_float(row, "optimizer_capture_ms")),
            ]
        )
    _print_table(
        ["phase", "iter", "save_fn_total", "racer_adapter_save", "state_dict", "optimizer_capture"],
        blocking_rows,
    )
    print()

    print("RACER store events")
    store_rows = []
    for row in events:
        if row.get("event") != "store":
            continue
        store_rows.append(
            [
                row.get("phase", ""),
                row.get("tag", ""),
                _fmt_ms(_float(row, "store_ms")),
                _fmt_ms(_float(row, "racer_calls_ms")),
                _fmt_ms(_float(row, "chunk_store_max_ms")),
                _fmt_ms(_float(row, "data_rows_ms")),
                _fmt_ms(_float(row, "parity_ms")),
                _fmt_ms(_float(row, "storage_ms")),
                _fmt_ms(_float(row, "storage_wait_ms")),
                _fmt_gib(_int(row, "local_bytes")),
                str(row.get("chunks", "")),
            ]
        )
    _print_table(
        [
            "phase",
            "tag",
            "adapter_store",
            "racer_calls",
            "max_chunk_store",
            "data_rows",
            "parity",
            "csd_storage",
            "csd_wait",
            "local_payload",
            "chunks",
        ],
        store_rows,
    )
    print()

    print("RACER restart/load events")
    load_rows = []
    for row in events:
        if row.get("event") != "load":
            continue
        load_rows.append(
            [
                row.get("phase", ""),
                row.get("tag", ""),
                _fmt_ms(_float(row, "total_ms")),
                _fmt_ms(_float(row, "racer_fetch_ms")),
                _fmt_ms(_float(row, "load_read_wait_ms")),
                _fmt_ms(_float(row, "tensor_materialize_ms")),
                _fmt_ms(_float(row, "tree_decode_ms")),
                _fmt_ms(_float(row, "runtime_prewarm_after_load_ms")),
            ]
        )
    _print_table(
        ["phase", "tag", "load_total", "racer_fetch", "read_wait", "materialize", "tree_decode", "prewarm"],
        load_rows,
    )
    print()

    print("CSD backend summary")
    csd_rows = []
    for row in csd:
        csd_rows.append(
            [
                row.get("tag", ""),
                str(row.get("op_count", "")),
                _fmt_ms(_float(row, "daemon_allocate_ms_sum")),
                _fmt_ms(_float(row, "daemon_memcpy_ms_cuda_event_sum")),
                _fmt_ms(_float(row, "checksum_ms_sum")),
                _fmt_ms(_float(row, "sqlite_ms_sum")),
                str(row.get("allocate_source_dynamic_cudaHostAlloc_count", "0")),
                str(row.get("allocate_source_pool_bump_count", "0")),
                str(row.get("allocate_source_free_list_count", "0")),
            ]
        )
    _print_table(
        ["tag", "ops", "alloc_sum", "copy_event_sum", "checksum_sum", "sqlite_sum", "dynamic_alloc", "pool_bump", "free_list"],
        csd_rows,
    )
    print()

    print("Notes")
    print("- save_fn_total is the Megatron training-process blocking time for save_checkpoint().")
    print("- adapter_store is inside save_fn_total; do not add them together.")
    print("- data_rows/parity/csd_storage are RACER internal phases inside adapter_store; they are for diagnosis.")
    print("- load_total is measured in the restarted training process before resumed training continues.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    summarize(args.result_dir)


if __name__ == "__main__":
    main()
