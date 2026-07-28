#!/usr/bin/env python3
"""
MMLU-Pro 결과를 방법론별 가로 막대 그래프로 시각화한다.

- 항목(방법론): Standard / SPP / BPP
- 지표: 카테고리별 정확도의 거시 평균(Macro Accuracy, %)
- 정렬: 점수 높은 순 (내림차순)

기본은 mmlu_pro_2100, gpt-4o-2024-08-06, run01 로그를 자동 탐색한다.
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


def aggregate_accuracy(path: Path) -> Tuple[Dict[str, float], int]:
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


def macro_accuracy(acc_by_cat: Dict[str, float]) -> float:
    vals = [
        acc_by_cat[c]
        for c in MMLU_PRO_CATEGORIES
        if not math.isnan(acc_by_cat.get(c, float("nan")))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def plot_method_ranking_bar(
    macro_scores: Dict[str, float],
    *,
    title: str,
    output_path: Path,
) -> None:
    rows = [(m, v) for m, v in macro_scores.items() if not math.isnan(v)]
    if not rows:
        raise SystemExit("표시할 점수가 없습니다.")

    row_map = {m: v for m, v in rows}
    methods = [m for m in METHOD_ORDER if m in row_map]
    vals = [row_map[m] for m in methods]

    labels = [METHOD_LABELS[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

    # 기존 시각화들과 톤을 맞춘 공통 스타일
    plt.rcParams["font.family"] = ["DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    background_color = "#FFFFFF"
    axis_color = "#5B6B7D"
    grid_color = "#C9CED6"
    text_color = "#2F3B4C"

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    y = np.arange(len(methods))
    bars = ax.barh(
        y,
        vals,
        color=colors,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.93,
        height=0.62,
    )
    ax.invert_yaxis()  # 높은 점수가 위에 오도록

    for i, b in enumerate(bars):
        v = vals[i]
        ax.text(
            v + 0.18,
            b.get_y() + b.get_height() / 2,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=text_color,
        )

    # 증분 시각화: Standard→SPP, SPP→BPP
    # - 증가율(%): (new-old)/old*100
    idx = {m: i for i, m in enumerate(methods)}
    arrow_x_offset = 0.66
    if "standard" in idx and "spp" in idx:
        i0, i1 = idx["standard"], idx["spp"]
        v0, v1 = vals[i0], vals[i1]
        y_mid = (y[i0] + y[i1]) / 2.0
        pct_inc = ((v1 - v0) / v0 * 100.0) if v0 != 0 else float("nan")
        ax.annotate(
            "",
            xy=(v1 + arrow_x_offset, y[i1]),
            xytext=(v0 + arrow_x_offset, y[i0]),
            arrowprops=dict(
                arrowstyle="Simple,tail_width=0.9,head_width=9,head_length=11",
                color="#D67D6A",
                lw=0.0,
                connectionstyle="arc3,rad=-0.25",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=2.0,
                alpha=0.78,
            ),
            zorder=4,
        )
        ax.text(
            (v0 + v1) / 2.0 + arrow_x_offset,
            y_mid - 0.05,
            f"+{pct_inc:.2f}%",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#7D3A2A",
            bbox=dict(facecolor="#FDE9E5", edgecolor="#E3A69A", boxstyle="round,pad=0.20"),
            zorder=5,
        )

    if "spp" in idx and "bpp" in idx:
        i0, i1 = idx["spp"], idx["bpp"]
        v0, v1 = vals[i0], vals[i1]
        y_mid = (y[i0] + y[i1]) / 2.0
        pct_inc = ((v1 - v0) / v0 * 100.0) if v0 != 0 else float("nan")
        ax.annotate(
            "",
            xy=(v1 + arrow_x_offset, y[i1]),
            xytext=(v0 + arrow_x_offset, y[i0]),
            arrowprops=dict(
                arrowstyle="Simple,tail_width=0.9,head_width=9,head_length=11",
                color="#D67D6A",
                lw=0.0,
                connectionstyle="arc3,rad=-0.25",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=2.0,
                alpha=0.78,
            ),
            zorder=4,
        )
        ax.text(
            (v0 + v1) / 2.0 + arrow_x_offset,
            y_mid + 0.05,
            f"+{pct_inc:.2f}%",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#7D3A2A",
            bbox=dict(facecolor="#FDE9E5", edgecolor="#E3A69A", boxstyle="round,pad=0.20"),
            zorder=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=axis_color)
    ax.set_xlabel("Accuracy (%)", fontsize=11, color=text_color, fontweight="bold")
    ax.tick_params(axis="x", labelsize=10, colors=axis_color)
    ax.grid(True, axis="x", color=grid_color, linewidth=1.1, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(axis_color)
    ax.spines["bottom"].set_color(axis_color)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.set_xlim(65.0, 72.0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=background_color)
    plt.close(fig)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMLU-Pro 방법론별 가로 막대(순위) 시각화.")
    p.add_argument("--batch-dir", type=Path, default=None)
    p.add_argument("--standard", type=Path, default=None)
    p.add_argument("--spp", type=Path, default=None)
    p.add_argument("--bpp", type=Path, default=None)
    p.add_argument("--task-stem", type=str, default=DEFAULT_TASK_STEM)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_SUBSTRING)
    p.add_argument("--run-marker", type=str, default=DEFAULT_RUN_MARKER)
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="PNG 경로 (기본: visualization/analysis_mmlu_pro_variant1_{task-stem}.png)",
    )
    p.add_argument(
        "--title",
        type=str,
        default="MMLU-Pro method ranking (macro accuracy)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    repo = _REPO_ROOT

    paths: Dict[str, Path] = {}
    for key, cli_path in [("standard", args.standard), ("spp", args.spp), ("bpp", args.bpp)]:
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

    macro_scores: Dict[str, float] = {}
    for method in METHOD_ORDER:
        p = paths[method]
        acc_d, skip = aggregate_accuracy(p)
        macro_scores[method] = macro_accuracy(acc_d)
        if skip:
            print(f"[{method}] skipped lines: {skip}", file=sys.stderr)

    stem_safe = re.sub(r"[^\w\-.]+", "_", args.task_stem).strip("_") or "mmlu_pro"
    out = (
        args.output
        or (repo / "visualization" / f"analysis_mmlu_pro_variant1_{stem_safe}.png")
    ).resolve()
    plot_method_ranking_bar(macro_scores, title=args.title, output_path=out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

