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


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _iteration_from_tag(tag: str) -> str:
    match = re.search(r"iter_([0-9]+)$", str(tag or ""))
    return str(int(match.group(1))) if match else "-"


def _summarize_store_events(stores: list[dict[str, str]]) -> list[list[str]]:
    groups: list[tuple[str, list[dict[str, str]]]] = [("all", stores)]
    phases = sorted({str(row.get("phase", "")) for row in stores if row.get("phase")})
    groups.extend((phase, [row for row in stores if str(row.get("phase", "")) == phase]) for phase in phases)
    rows: list[list[str]] = []
    for phase, items in groups:
        if not items:
            continue
        store_values = [value for value in (_float(row, "store_ms") for row in items) if value is not None]
        wait_values = [value for value in (_float(row, "storage_wait_ms") for row in items) if value is not None]
        slowest = max(items, key=lambda row: _float(row, "store_ms") or 0.0)
        rows.append(
            [
                phase,
                str(len(items)),
                _fmt_ms(_mean(store_values)),
                _fmt_ms(_percentile(store_values, 0.50)),
                _fmt_ms(max(store_values) if store_values else None),
                _fmt_ms(_percentile(wait_values, 0.50)),
                _fmt_ms(max(wait_values) if wait_values else None),
                _fmt_gib(_int(items[0], "local_bytes")),
                str(items[0].get("chunks", "")),
                _iteration_from_tag(str(slowest.get("tag", ""))),
            ]
        )
    return rows


def _summarize_csd_totals(csd: list[dict[str, str]]) -> list[list[str]]:
    totals = {
        "groups": len(csd),
        "ops": 0,
        "dynamic": 0,
        "pool": 0,
        "free": 0,
        "alloc_ms": 0.0,
        "checksum_ms": 0.0,
        "sqlite_ms": 0.0,
        "copy_wall_ms": 0.0,
    }
    for row in csd:
        totals["ops"] += _int(row, "op_count") or 0
        totals["dynamic"] += _int(row, "allocate_source_dynamic_cudaHostAlloc_count") or 0
        totals["pool"] += _int(row, "allocate_source_pool_bump_count") or 0
        totals["free"] += _int(row, "allocate_source_free_list_count") or 0
        totals["alloc_ms"] += _float(row, "daemon_allocate_ms_sum") or 0.0
        totals["checksum_ms"] += _float(row, "checksum_ms_sum") or 0.0
        totals["sqlite_ms"] += _float(row, "sqlite_ms_sum") or 0.0
        totals["copy_wall_ms"] += _float(row, "daemon_memcpy_ms_wall_sum") or 0.0
    return [
        [
            str(totals["groups"]),
            str(totals["ops"]),
            str(totals["dynamic"]),
            str(totals["pool"]),
            str(totals["free"]),
            _fmt_ms(float(totals["alloc_ms"])),
            _fmt_ms(float(totals["checksum_ms"])),
            _fmt_ms(float(totals["sqlite_ms"])),
            _fmt_ms(float(totals["copy_wall_ms"])),
        ]
    ]


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


def summarize(result_dir: Path, *, show_iteration_details: bool = False) -> None:
    result_dir = result_dir.resolve()
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    events = _read_csv(result_dir / "parsed_log_events.csv")
    csd = _read_csv(result_dir / "csd_profile_summary.csv")

    print(f"结果目录: {result_dir}")
    if summary:
        metadata = summary.get("metadata", {}) if isinstance(summary.get("metadata"), dict) else {}
        print(
            f"模型: {metadata.get('model', summary.get('model'))}  "
            f"杀进程 iter: {[ _iteration_from_tag(tag) for tag in metadata.get('kill_target_tags', [metadata.get('target_tag', summary.get('target_tag'))]) ]}  "
            f"最终 iter: {_iteration_from_tag(str(metadata.get('final_tag', metadata.get('resume_tag', summary.get('resume_tag')))))}"
        )
        print(
            "正确性检查: "
            f"三次恢复 load={summary.get('restart_load_observed')}  "
            f"最终 checkpoint={summary.get('post_resume_checkpoint_observed')}"
        )
        if metadata:
            print(
                "测试节奏: "
                f"保存间隔={metadata.get('save_interval')}  "
                f"杀进程间隔={metadata.get('kill_interval_iters')}  "
                f"杀进程次数={metadata.get('kill_count')}  "
                f"最终迭代={metadata.get('final_train_iters')}"
            )
        run_results = summary.get("run_results")
        if isinstance(run_results, list):
            print("运行分段")
            run_rows = []
            for item in run_results:
                if not isinstance(item, dict):
                    continue
                wall = _float(item, "wall_ms_until_kill")
                if wall is None:
                    wall = _float(item, "wall_ms")
                run_rows.append(
                    [
                        str(item.get("phase", "")),
                        str(item.get("action", "")),
                        _iteration_from_tag(str(item.get("target_tag", ""))),
                        _iteration_from_tag(str(item.get("expected_load_tag", ""))),
                        str(item.get("returncode", "")),
                        _fmt_ms(wall),
                    ]
                )
            _print_table(["阶段", "动作", "目标iter", "期望load", "返回码", "墙钟"], run_rows)
        else:
            print(
                "运行墙钟时间: "
                f"first_until_kill={_fmt_ms(_float(summary.get('first_run', {}), 'wall_ms_until_kill'))}  "
                f"resume={_fmt_ms(_float(summary.get('resume_run', {}), 'wall_ms'))}"
            )
    print()

    iteration_summary = _read_csv(result_dir / "iteration_time_summary.csv")
    if iteration_summary:
        print("每迭代训练耗时摘要")
        _print_table(
            ["阶段", "范围", "样本数", "平均", "P50", "P95", "最大"],
            [
                [
                    row.get("phase", ""),
                    row.get("bucket", ""),
                    row.get("count", ""),
                    _fmt_ms(_float(row, "mean_ms")),
                    _fmt_ms(_float(row, "p50_ms")),
                    _fmt_ms(_float(row, "p95_ms")),
                    _fmt_ms(_float(row, "max_ms")),
                ]
                for row in iteration_summary
            ],
        )
        print()

    iter_csv = _read_csv(result_dir / "iteration_times.csv")
    if iter_csv and not show_iteration_details:
        print(f"每迭代训练耗时明细: 已写入 {result_dir / 'iteration_times.csv'}")
        print("需要在命令行展开时加 `--show-iteration-details`。")
    else:
        print("每迭代训练耗时明细")
        iter_rows: list[list[str]] = []
        if iter_csv:
            for row in iter_csv:
                iter_rows.append(
                    [
                        row.get("phase", ""),
                        row.get("iteration", ""),
                        _fmt_ms(_float(row, "elapsed_time_per_iteration_ms")),
                        row.get("checkpoint_iteration", ""),
                    ]
                )
            _print_table(["阶段", "iter", "每迭代耗时", "是否保存点"], iter_rows)
        else:
            for phase, name in (("first", "first_run.log"), ("resume", "resume_run.log")):
                for iteration, ms in _iter_times(result_dir / name):
                    iter_rows.append([phase, str(iteration), f"{ms:.2f} ms"])
            _print_table(["阶段", "iter", "每迭代耗时"], iter_rows)
    print()

    print("Megatron 保存阻塞耗时")
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
        ["阶段", "iter", "save总耗时", "RACER保存", "state_dict", "optimizer"],
        blocking_rows,
    )
    print()

    stores = [row for row in events if row.get("event") == "store"]
    print("RACER 保存耗时摘要")
    _print_table(
        [
            "阶段",
            "次数",
            "store平均",
            "storeP50",
            "store最大",
            "waitP50",
            "wait最大",
            "payload",
            "chunks",
            "最慢iter",
        ],
        _summarize_store_events(stores),
    )
    print()

    print("RACER 保存事件短表")
    store_rows = []
    for row in stores:
        store_rows.append(
            [
                row.get("phase", ""),
                _iteration_from_tag(row.get("tag", "")),
                _fmt_ms(_float(row, "store_ms")),
                _fmt_ms(_float(row, "storage_wait_ms")),
                _fmt_gib(_int(row, "local_bytes")),
                str(row.get("chunks", "")),
            ]
        )
    _print_table(
        ["阶段", "iter", "store", "CSD wait", "payload", "chunks"],
        store_rows,
    )
    print()

    print("RACER 恢复读取事件")
    load_rows = []
    for row in events:
        if row.get("event") != "load":
            continue
        load_rows.append(
            [
                row.get("phase", ""),
                _iteration_from_tag(row.get("tag", "")),
                _fmt_ms(_float(row, "total_ms")),
                _fmt_ms(_float(row, "racer_fetch_ms")),
                _fmt_ms(_float(row, "load_read_wait_ms")),
                _fmt_ms(_float(row, "tensor_materialize_ms")),
                _fmt_ms(_float(row, "tree_decode_ms")),
                _fmt_ms(_float(row, "runtime_prewarm_after_load_ms")),
            ]
        )
    _print_table(
        ["阶段", "load iter", "总耗时", "fetch", "read wait", "materialize", "tree decode", "prewarm"],
        load_rows,
    )
    print()

    print("CSD 后端摘要")
    _print_table(
        ["profile组", "ops", "dynamic_alloc", "pool_bump", "free_list", "alloc", "checksum", "sqlite", "copy wall"],
        _summarize_csd_totals(csd),
    )
    print()

    print("说明")
    print("- save总耗时是训练进程里 save_checkpoint() 的阻塞时间。")
    print("- store 是 RACER adapter 的保存耗时，包含在 save总耗时里，不要相加。")
    print("- 保存短表只保留人读的关键字段；完整字段在 parsed_log_events.csv。")
    print("- dynamic_alloc 为 0 时，表示本轮没有临时 cudaHostAlloc pinned memory 分配。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--show-iteration-details", action="store_true", help="Print every iteration timing row.")
    args = parser.parse_args()
    summarize(args.result_dir, show_iteration_details=args.show_iteration_details)


if __name__ == "__main__":
    main()
