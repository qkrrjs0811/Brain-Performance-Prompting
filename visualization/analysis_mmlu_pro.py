#!/usr/bin/env python3
"""
MMLU-Pro 로그에서 카테고리별 정확도(성능)를 grouped bar chart로 시각화한다.

기본 입력은 mmlu_pro_2100 run01(gpt-4o-2024-08-06)이며,
standard / spp / bpp 각 1개 결과 JSONL을 자동 탐색한다.

사용 예:
  python visualization/analysis_mmlu_pro.py
  python visualization/analysis_mmlu_pro.py --output visualization/out.png
  python visualization/analysis_mmlu_pro.py \\
    --standard path/to/...__method-standard_....jsonl \\
    --spp path/to/...__method-spp_....jsonl \\
    --bpp path/to/...__method-bpp_....jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np

_VIZ_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _VIZ_DIR.parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))
import method_colors

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib 가 필요합니다: pip install matplotlib") from e


MMLU_PRO_CATEGORIES: List[str] = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
]

METHOD_ORDER: List[str] = ["standard", "spp", "bpp"]
METHOD_LABELS: Dict[str, str] = {
    "standard": "Standard",
    "spp": "SPP",
    "bpp": "BPP",
}

METHOD_COLORS: Dict[str, str] = {
    "standard": method_colors.METHOD_COLORS_SLUG["standard"],
    "spp": method_colors.METHOD_COLORS_SLUG["spp"],
    "bpp": method_colors.METHOD_COLORS_SLUG["bpp"],
}

DEFAULT_TASK_STEM = "mmlu_pro_2100"
DEFAULT_MODEL_SUBSTRING = "gpt-4o-2024-08-06"
DEFAULT_RUN_MARKER = "_run01__"


def default_batch_dir(repo: Path) -> Path:
    return (
        repo
        / "logs"
        / "hf_multiple_choice"
        / "gpt-4o-2024-08-06_wo_sys_mes"
        / "mmlu_pro"
        / "batch_api"
    )


def _is_result_jsonl(name: str) -> bool:
    if not name.endswith(".jsonl"):
        return False
    if "batch_input" in name or "batch_meta" in name or "batch_phase" in name:
        return False
    return True


def discover_mmlu_pro_logs(
    batch_dir: Path,
    *,
    task_stem: str = DEFAULT_TASK_STEM,
    model_substring: str = DEFAULT_MODEL_SUBSTRING,
    run_marker: str = DEFAULT_RUN_MARKER,
) -> Dict[str, Path]:
    candidates: DefaultDict[str, List[Tuple[float, Path]]] = defaultdict(list)
    if not batch_dir.is_dir():
        return {}

    for p in batch_dir.rglob("*.jsonl"):
        if not p.is_file() or not _is_result_jsonl(p.name):
            continue
        if task_stem not in p.name:
            continue
        if model_substring not in p.name:
            continue
        if run_marker not in p.name:
            continue
        mo = re.search(r"__method-([^_]+)_model-", p.name)
        if not mo:
            continue
        method = mo.group(1)
        if method not in METHOD_ORDER:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        candidates[method].append((mtime, p))

    out: Dict[str, Path] = {}
    for m in METHOD_ORDER:
        lst = candidates.get(m)
        if not lst:
            continue
        lst.sort(key=lambda x: x[0], reverse=True)
        out[m] = lst[0][1]
    return out


def category_from_record(rec: Dict[str, Any]) -> Optional[str]:
    td = rec.get("task_data") or {}
    meta = td.get("metadata") or {}
    c = meta.get("category")
    return str(c).strip().lower() if isinstance(c, str) else None


def correct_from_record(rec: Dict[str, Any]) -> Optional[bool]:
    toi = rec.get("test_output_infos")
    if not toi or not isinstance(toi[0], dict):
        return None
    if "correct" not in toi[0]:
        return None
    return bool(toi[0]["correct"])


def aggregate_log(path: Path) -> Tuple[Dict[str, float], int]:
    sums_ok: DefaultDict[str, List[int]] = defaultdict(list)
    skipped = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cat = category_from_record(rec)
            cor = correct_from_record(rec)
            if cat is None or cor is None:
                skipped += 1
                continue
            sums_ok[cat].append(1 if cor else 0)

    acc: Dict[str, float] = {}
    for cat in MMLU_PRO_CATEGORIES:
        ok = sums_ok.get(cat, [])
        acc[cat] = 100.0 * sum(ok) / len(ok) if ok else float("nan")

    return acc, skipped


def plot_mmlu_pro_analysis(
    by_method_acc: Dict[str, Dict[str, float]],
    *,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    fig.suptitle(title, fontsize=14, fontweight="600", y=0.98)

    x = np.arange(len(MMLU_PRO_CATEGORIES), dtype=float)
    width = 0.24
    offsets = {
        "standard": -width,
        "spp": 0.0,
        "bpp": width,
    }
    # 카테고리별 최고 성능(동률 허용) 표시를 위해 method x category 행렬 구성
    vals_by_method: Dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        vals_by_method[method] = np.array(
            [by_method_acc.get(method, {}).get(c, float("nan")) for c in MMLU_PRO_CATEGORIES],
            dtype=float,
        )

    winner_mask: Dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        winner_mask[method] = np.zeros(len(MMLU_PRO_CATEGORIES), dtype=bool)
    for i in range(len(MMLU_PRO_CATEGORIES)):
        col_vals = np.array([vals_by_method[m][i] for m in METHOD_ORDER], dtype=float)
        finite = np.isfinite(col_vals)
        if not np.any(finite):
            continue
        max_v = np.max(col_vals[finite])
        for mi, method in enumerate(METHOD_ORDER):
            if finite[mi] and abs(col_vals[mi] - max_v) <= 1e-9:
                winner_mask[method][i] = True

    max_val = 0.0
    for method in METHOD_ORDER:
        vals = vals_by_method[method]
        vals_plot = np.nan_to_num(vals, nan=0.0)
        bars = ax.bar(
            x + offsets[method],
            vals_plot,
            width=width * 0.95,
            color=METHOD_COLORS[method],
            edgecolor="#5B6B7D",
            linewidth=0.7,
            label=METHOD_LABELS[method],
            alpha=0.95 if method == "bpp" else 0.9,
        )
        for i, b in enumerate(bars):
            v = vals[i]
            if math.isnan(v):
                continue
            max_val = max(max_val, float(v))
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.35,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#2F3B4C",
                fontweight="bold" if winner_mask[method][i] else "normal",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MMLU_PRO_CATEGORIES, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("MMLU-Pro category-wise accuracy", fontsize=12, fontweight="600")
    ax.grid(True, axis="y", linestyle=":", alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", ncol=3, fontsize=10, framealpha=0.95)
    ax.set_ylim(0, max(100.0, math.ceil((max_val + 3.0) / 5.0) * 5.0))

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMLU-Pro category-wise accuracy bar chart.")
    p.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="batch_api 폴더 (기본: logs/.../mmlu_pro/batch_api)",
    )
    p.add_argument("--standard", type=Path, default=None)
    p.add_argument("--spp", type=Path, default=None)
    p.add_argument("--bpp", type=Path, default=None)
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="PNG 경로 (기본: visualization/analysis_mmlu_pro_2100.png)",
    )
    p.add_argument(
        "--task-stem",
        type=str,
        default=DEFAULT_TASK_STEM,
        help=f"파일명에 포함되는 task stem (기본: {DEFAULT_TASK_STEM})",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_SUBSTRING,
        help=f"파일명에 포함되는 모델 식별 문자열 (기본: {DEFAULT_MODEL_SUBSTRING})",
    )
    p.add_argument(
        "--run-marker",
        type=str,
        default=DEFAULT_RUN_MARKER,
        help=f"파일명에 포함되는 run 마커 (기본: {DEFAULT_RUN_MARKER!r})",
    )
    p.add_argument(
        "--title",
        type=str,
        default="MMLU-Pro (2100) category-wise accuracy — gpt-4o-2024-08-06, run01",
        help="그림 제목",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    repo = _REPO_ROOT

    paths: Dict[str, Path] = {}
    for key, cli_path in [
        ("standard", args.standard),
        ("spp", args.spp),
        ("bpp", args.bpp),
    ]:
        if cli_path is not None:
            if not cli_path.is_file():
                raise SystemExit(f"파일이 없습니다: {cli_path}")
            paths[key] = cli_path.resolve()

    if len(paths) < 3:
        batch_dir = (args.batch_dir or default_batch_dir(repo)).resolve()
        discovered = discover_mmlu_pro_logs(
            batch_dir,
            task_stem=args.task_stem,
            model_substring=args.model,
            run_marker=args.run_marker,
        )
        for m in METHOD_ORDER:
            if m not in paths and m in discovered:
                paths[m] = discovered[m]
        missing = [m for m in METHOD_ORDER if m not in paths]
        if missing:
            raise SystemExit(
                "다음 방법 로그를 찾지 못했습니다: "
                + ", ".join(missing)
                + f"\n  batch_dir={batch_dir}"
                + f"\n  필터: task_stem={args.task_stem!r}, model={args.model!r}, run_marker={args.run_marker!r}\n"
                "  --standard / --spp / --bpp 로 경로를 직접 지정하거나 "
                "--task-stem / --model / --run-marker 를 조정하세요."
            )

    by_acc: Dict[str, Dict[str, float]] = {}

    for method in METHOD_ORDER:
        p = paths[method]
        acc_d, skip = aggregate_log(p)
        by_acc[method] = acc_d
        if skip:
            print(f"[{method}] skipped lines: {skip}", file=sys.stderr)

    out = (args.output or (repo / "visualization" / "analysis_mmlu_pro_2100.png")).resolve()
    plot_mmlu_pro_analysis(by_acc, title=args.title, output_path=out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
