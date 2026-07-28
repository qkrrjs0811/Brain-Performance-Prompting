#!/usr/bin/env python3
"""
logs/ 아래 gpt-5.4 reasoning effort 실험 폴더(`gpt-5.4_re-{none,low,medium,high}_wo_sys_mes`)의
결과 JSONL을 읽어 시각화한다.

**Standard**: API reasoning effort(None→High)에 따른 정확도 곡선(run 평균의 mean ± SEM).

**BPP**: 주장 비교를 위해 **reasoning = none** 조건에서의 성능만 표시한다.
(BPP를 low/medium/high effort로 돌린 다른 로그는 그리지 않음 — 추가 reasoning budget 가설과의 대비용.)

- X축: Reasoning effort (Standard의 설정 축)
- Y축: Accuracy (%)

사용 예:
  python visualization/gpt54_bpp_vs_standard_reasoning_effort.py
  python visualization/gpt54_bpp_vs_standard_reasoning_effort.py \\
    --logs-root logs --output visualization/gpt54_effort_vs_accuracy.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))
import method_colors

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib 가 필요합니다: pip install matplotlib") from e


EFFORTS: Tuple[str, ...] = ("none", "low", "medium", "high")
EFFORT_XLABELS = ("None", "Low", "Medium", "High")

PLOT_METHODS: Tuple[str, ...] = ("standard", "bpp")

SKIP_SUBSTR = ("batch_input", "batch_meta", "batch_phase", "batch_anthropic")
RE_LOG_FOLDER = re.compile(r"gpt-5\.4_re-(none|low|medium|high)_wo_sys_mes")

C_STANDARD = method_colors.METHOD_COLORS_SLUG["standard"]
C_BPP = method_colors.METHOD_COLORS_SLUG["bpp"]
C_BPP_EDGE = method_colors.BPP_MARKER_EDGE
C_GRID = "#D8DEE6"
C_TEXT = "#2B3542"
C_MUTED = "#8A95A3"
BG_PANEL = "#FAFBFC"
FIG_BG = "#F4F6F9"


def _method_from_filename(name: str) -> Optional[str]:
    if "__method-" not in name:
        return None
    return name.split("__method-")[1].split("_model")[0]


def _dataset_stem_from_filename(name: str) -> str:
    if "__method-" not in name:
        return name
    return name.split("__method-")[0]


def _effort_from_path(path: Path) -> Optional[str]:
    m = RE_LOG_FOLDER.search(str(path))
    return m.group(1) if m else None


def _task_from_path(path: Path, logs_root: Path) -> Optional[str]:
    try:
        rel = path.relative_to(logs_root)
    except ValueError:
        return None
    parts = rel.parts
    return parts[0] if parts else None


def _skip_file(name: str) -> bool:
    return any(s in name for s in SKIP_SUBSTR)


def _mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    xs = [float(x) for x in values]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(xs))
    if n < 2:
        return mean, 0.0
    sem = float(np.std(xs, ddof=1) / math.sqrt(n))
    return mean, sem


def _accuracy_trivia(path: Path) -> float:
    accs: List[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for info in d.get("test_output_infos", []):
                qc = info.get("question_count") or 0
                if qc > 0:
                    accs.append(info.get("correct_count", 0) / qc)
    if not accs:
        raise ValueError(f"no trivia accuracies: {path}")
    return float(np.mean(accs))


def _accuracy_codenames(path: Path) -> float:
    accs: List[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for info in d["test_output_infos"]:
                tc = info.get("target_count") or 0
                if tc > 0:
                    accs.append(info.get("matched_count", 0) / tc)
    if not accs:
        raise ValueError(f"no codenames accuracies: {path}")
    return float(np.mean(accs))


def _accuracy_logic_grid(path: Path) -> float:
    accs: List[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for info in d["test_output_infos"]:
                accs.append(1.0 if info.get("correct") else 0.0)
    if not accs:
        raise ValueError(f"no logic accuracies: {path}")
    return float(np.mean(accs))


ACC_HANDLERS: Dict[str, Callable[[Path], float]] = {
    "trivia_creative_writing": _accuracy_trivia,
    "codenames_collaborative": _accuracy_codenames,
    "logic_grid_puzzle": _accuracy_logic_grid,
}


def iter_metric_files(logs_root: Path, methods: Tuple[str, ...]) -> Iterable[Path]:
    for p in sorted(logs_root.rglob("*.jsonl")):
        if _skip_file(p.name):
            continue
        if not RE_LOG_FOLDER.search(str(p)):
            continue
        m = _method_from_filename(p.name)
        if m is None or m not in methods:
            continue
        task = _task_from_path(p, logs_root)
        if task is None or task not in ACC_HANDLERS:
            continue
        yield p


def collect_accuracy(
    logs_root: Path,
    methods: Tuple[str, ...] = PLOT_METHODS,
) -> Dict[Tuple[str, str, str, str], List[float]]:
    acc_map: DefaultDict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for path in iter_metric_files(logs_root, methods):
        task = _task_from_path(path, logs_root)
        assert task is not None
        effort = _effort_from_path(path)
        assert effort is not None
        method = _method_from_filename(path.name)
        assert method is not None
        ds = _dataset_stem_from_filename(path.name)
        acc_map[(task, ds, method, effort)].append(ACC_HANDLERS[task](path))
    return dict(acc_map)


def _panel_title(task: str, dataset_stem: str) -> str:
    if task == "codenames_collaborative":
        return "Codenames.C"
    if task == "logic_grid_puzzle":
        return "Logic.G.Puzzle"
    if task == "trivia_creative_writing":
        if "_n_5" in dataset_stem:
            return "Trivia C.W (N=5)"
        if "_n_10" in dataset_stem:
            return "Trivia C.W (N=10)"
        return "Trivia C.W"
    return dataset_stem[:22]


def _ordered_panels(
    acc_map: Dict[Tuple[str, str, str, str], List[float]],
) -> List[Tuple[str, str]]:
    seen: set[Tuple[str, str]] = set()
    order: List[Tuple[str, str]] = []
    preferred = [
        ("trivia_creative_writing", "trivia_creative_writing_100_n_5.jsonl"),
        ("trivia_creative_writing", "trivia_creative_writing_100_n_10.jsonl"),
        ("codenames_collaborative", "codenames_50.jsonl"),
        ("logic_grid_puzzle", "logic_grid_puzzle_200.jsonl"),
    ]
    for t, d in preferred:
        if any(k[0] == t and k[1] == d for k in acc_map):
            order.append((t, d))
            seen.add((t, d))
    for (t, d, _, _) in sorted(acc_map.keys()):
        if (t, d) not in seen:
            order.append((t, d))
            seen.add((t, d))
    return order


def _apply_panel_style(ax: plt.Axes) -> None:
    ax.set_facecolor(BG_PANEL)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.85, linestyle="-", alpha=0.95)
    ax.tick_params(axis="both", colors=C_TEXT, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_MUTED)
    ax.spines["bottom"].set_color(C_MUTED)


def plot_effort_vs_performance(
    acc_map: Dict[Tuple[str, str, str, str], List[float]],
    out_path: Path,
) -> None:
    panels = _ordered_panels(acc_map)
    n = len(panels)
    if n == 0:
        raise ValueError("no panels")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.unicode_minus": False,
            "figure.facecolor": FIG_BG,
            "axes.edgecolor": C_MUTED,
        }
    )

    fig_w = min(3.75 * n, 16.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.35), dpi=165, squeeze=False, facecolor=FIG_BG)
    axes_flat = np.atleast_1d(axes).ravel()

    x = np.arange(len(EFFORTS), dtype=float)

    for j, (task, ds) in enumerate(panels):
        ax = axes_flat[j]

        # Standard: 모든 reasoning effort
        std_means: List[float] = []
        std_sems: List[float] = []
        for e in EFFORTS:
            m, s = _mean_sem(acc_map.get((task, ds, "standard", e), []))
            std_means.append(m * 100.0 if not math.isnan(m) else float("nan"))
            std_sems.append(s * 100.0 if not math.isnan(s) else 0.0)
        std_arr = np.array(std_means, dtype=float)
        valid_std = np.isfinite(std_arr)
        if np.any(valid_std):
            ax.plot(
                x[valid_std],
                std_arr[valid_std],
                color=C_STANDARD,
                marker="o",
                markersize=6.2,
                markerfacecolor="white",
                markeredgewidth=1.85,
                markeredgecolor=C_STANDARD,
                linewidth=2.0,
                label="_nolegend_",
                zorder=3,
                alpha=0.98,
            )
            ax.errorbar(
                x,
                std_arr,
                yerr=np.array(std_sems),
                fmt="none",
                ecolor=C_STANDARD,
                elinewidth=1.45,
                capsize=3.2,
                capthick=1.45,
                zorder=2,
                alpha=0.9,
            )

        # BPP: reasoning = none 만 (다른 effort의 BPP는 플롯하지 않음)
        bpp_m, bpp_s = _mean_sem(acc_map.get((task, ds, "bpp", "none"), []))
        if not math.isnan(bpp_m):
            bpp_pct = bpp_m * 100.0
            bpp_sem = bpp_s * 100.0
            x_left = float(x[0]) - 0.22
            x_right = float(x[-1]) + 0.22
            ax.hlines(
                bpp_pct,
                x_left,
                x_right,
                colors=C_BPP,
                linestyles=(0, (6, 4)),
                linewidth=2.5,
                label="_nolegend_",
                zorder=4,
                alpha=0.95,
            )
            ax.fill_between(
                [x_left, x_right],
                bpp_pct - bpp_sem,
                bpp_pct + bpp_sem,
                color=C_BPP,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            ax.errorbar(
                [0.0],
                [bpp_pct],
                yerr=[bpp_sem],
                fmt="none",
                ecolor=C_BPP,
                elinewidth=1.45,
                capsize=3.2,
                capthick=1.45,
                zorder=5,
                alpha=0.95,
            )
            ax.plot(
                [0.0],
                [bpp_pct],
                color=C_BPP,
                marker="D",
                markersize=7.5,
                markerfacecolor=C_BPP,
                markeredgewidth=1.85,
                markeredgecolor=C_BPP_EDGE,
                linestyle="None",
                label="_nolegend_",
                zorder=6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(EFFORT_XLABELS, fontsize=8.8, color=C_TEXT)
        ax.set_xlabel("Reasoning effort", fontsize=9.5, color=C_MUTED)
        if j == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=10.5, color=C_TEXT)
        ax.set_title(_panel_title(task, ds), fontsize=11.2, fontweight="600", color=C_TEXT, pad=8)
        _apply_panel_style(ax)

        ypool: List[float] = []
        for e in EFFORTS:
            m, _ = _mean_sem(acc_map.get((task, ds, "standard", e), []))
            if not math.isnan(m):
                ypool.append(m * 100.0)
        bm, _ = _mean_sem(acc_map.get((task, ds, "bpp", "none"), []))
        if not math.isnan(bm):
            ypool.append(bm * 100.0)
        if ypool:
            lo, hi = min(ypool), max(ypool)
            pad = max(1.2, (hi - lo) * 0.14)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.25, float(x[-1]) + 0.25)

    handles = [
        Line2D(
            [0],
            [0],
            color=C_STANDARD,
            marker="o",
            linestyle="-",
            linewidth=2.35,
            markersize=9,
            markerfacecolor="white",
            markeredgecolor=C_STANDARD,
            markeredgewidth=1.65,
            label="Standard",
        ),
        Line2D(
            [-0.22, 0.0, 0.22],
            [0.0, 0.0, 0.0],
            color=C_BPP,
            linestyle=(0, (6, 4)),
            linewidth=2.75,
            marker="D",
            markersize=9,
            markerfacecolor=C_BPP,
            markeredgecolor=C_BPP_EDGE,
            markeredgewidth=1.55,
            markevery=[1],
            label="BPP",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        edgecolor="#D0D5DD",
        fontsize=10.8,
        borderpad=0.65,
        labelspacing=0.55,
        handlelength=2.2,
        handleheight=0.9,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=(0, 0.06, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs-root", type=Path, default=_REPO_ROOT / "logs")
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "visualization" / "gpt54_bpp_vs_standard_reasoning_effort.png",
        help="저장할 PNG 경로",
    )
    args = p.parse_args()

    logs_root = args.logs_root.resolve()
    if not logs_root.is_dir():
        raise SystemExit(f"logs 폴더가 없습니다: {logs_root}")

    acc_map = collect_accuracy(logs_root, PLOT_METHODS)
    if not acc_map:
        raise SystemExit(
            f"{logs_root} 에서 gpt-5.4_re-* 로그(standard/bpp)를 찾지 못했습니다."
        )

    out = args.output.resolve()
    plot_effort_vs_performance(acc_map, out)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
