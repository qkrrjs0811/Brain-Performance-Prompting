from pathlib import Path
import argparse
from collections import Counter
import json
import re

import matplotlib.pyplot as plt
import numpy as np


# 공통 시각 스타일
plt.rcParams["font.family"] = ["DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

BACKGROUND_COLOR = "#FFFFFF"
AXIS_COLOR = "#5B6B7D"
GRID_COLOR = "#C9CED6"
TEXT_COLOR = "#2F3B4C"

# 페르소나별 막대 색상
PERSONA_COLORS = {
    "Spymaster": "#8DB7D8",  # brain_region_frequency.py와 동일
    "Guesser": "#A8D5BA",    # brain_region_frequency.py와 동일
}

# x축에서 표시하지 않을 참가자
EXCLUDED_X_AXIS_REGIONS = {"AI Assistant (you)"}

# x축 정렬 우선순위: 넓은 영역 -> 세분화 영역
REGION_ORDER_BROAD_TO_SPECIFIC = [
    "Frontal Lobe",
    "Temporal Lobe",
    "Occipital Lobe",
    "Limbic System",
    "Hippocampus",
    "Amygdala",
    "Olfactory Bulb",
    "Primary Visual Cortex",
    "Superior Parietal Lobule",
    "Dorsolateral Prefrontal Cortex",
]

BROAD_REGIONS = {
    "Frontal Lobe",
    "Temporal Lobe",
    "Occipital Lobe",
    "Limbic System",
    "Hippocampus",
}

DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "logs/codenames_collaborative/gpt-4o-2024-08-06_wo_sys_mes/"
    "codenames_50.jsonl__method-bpp_model-gpt-4o-2024-08-06_temp-_temp-0.0_topp-_topp-1.0_start0-end50_run01__wo_sys.jsonl"
)

PARTICIPANTS_PATTERN = re.compile(r"Participants:\s*(.*)")


def _extract_participants(content):
    match = PARTICIPANTS_PATTERN.search(content)
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(";") if item.strip()]


def _count_from_role(raw_role_entries):
    counter = Counter()
    for entry in raw_role_entries:
        try:
            content = entry["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            continue
        for participant in _extract_participants(content):
            counter[participant] += 1
    return counter


def load_brain_region_frequencies(log_path):
    spymaster_counter = Counter()
    guesser_counter = Counter()

    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            spymaster_counter.update(_count_from_role(record.get("raw_response_spymaster", [])))
            guesser_counter.update(_count_from_role(record.get("raw_response_guesser", [])))

    if not spymaster_counter or not guesser_counter:
        raise ValueError("로그에서 Spymaster/Guesser 참가자 정보를 찾지 못했습니다.")

    return spymaster_counter, guesser_counter


def _build_grouped_series(spymaster_counter, guesser_counter):
    all_regions = (
        set(spymaster_counter.keys()) | set(guesser_counter.keys())
    ) - EXCLUDED_X_AXIS_REGIONS

    ordered_regions = [r for r in REGION_ORDER_BROAD_TO_SPECIFIC if r in all_regions]
    extras = sorted(
        [r for r in all_regions if r not in REGION_ORDER_BROAD_TO_SPECIFIC]
    )
    all_regions = ordered_regions + extras

    spymaster_values = [spymaster_counter.get(region, 0) for region in all_regions]
    guesser_values = [guesser_counter.get(region, 0) for region in all_regions]
    return all_regions, spymaster_values, guesser_values


def _wrap_label_two_lines(label, max_first_line=16):
    """긴 라벨을 2줄로 줄바꿈."""
    if len(label) <= max_first_line or " " not in label:
        return label

    words = label.split()
    line1 = []
    line2 = []
    length = 0
    for word in words:
        candidate = length + (1 if line1 else 0) + len(word)
        if candidate <= max_first_line:
            line1.append(word)
            length = candidate
        else:
            line2.append(word)

    if not line2:
        return label
    return f"{' '.join(line1)}\n{' '.join(line2)}"


def create_figure(log_path):
    spymaster_counter, guesser_counter = load_brain_region_frequencies(log_path)
    regions, spymaster_values, guesser_values = _build_grouped_series(spymaster_counter, guesser_counter)

    x = np.arange(len(regions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(18, 7), dpi=120)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    bars_spymaster = ax.bar(
        x - width / 2,
        spymaster_values,
        width=width,
        color=PERSONA_COLORS["Spymaster"],
        label="Spymaster",
        zorder=3,
    )
    bars_guesser = ax.bar(
        x + width / 2,
        guesser_values,
        width=width,
        color=PERSONA_COLORS["Guesser"],
        label="Guesser",
        zorder=3,
    )

    # 넓은 영역 vs 세부 영역 시각적 구분 (배경 밴드 + 경계선 + 라벨)
    broad_last_idx = -1
    for idx, region in enumerate(regions):
        if region in BROAD_REGIONS:
            broad_last_idx = idx

    if broad_last_idx >= 0 and broad_last_idx < len(regions) - 1:
        ax.axvspan(-0.5, broad_last_idx + 0.5, color="#E9F0F6", alpha=0.44, zorder=0)
        ax.axvspan(broad_last_idx + 0.5, len(regions) - 0.5, color="#EAF3E2", alpha=0.42, zorder=0)
        ax.axvline(broad_last_idx + 0.5, color="#A9B5C7", linestyle="--", linewidth=1.3, zorder=1)

    for bars in (bars_spymaster, bars_guesser):
        for bar in bars:
            val = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.45,
                str(val),
                ha="center",
                va="bottom",
                fontsize=11.5,
                color=TEXT_COLOR,
                fontweight="bold",
                zorder=4,
            )

    ax.set_ylabel("Frequency", fontsize=16, color=TEXT_COLOR, fontweight="bold")
    ax.set_xticks(x)
    wrapped_regions = [_wrap_label_two_lines(region) for region in regions]
    ax.set_xticklabels(wrapped_regions, rotation=0, ha="center", fontsize=12, color=AXIS_COLOR)
    ax.tick_params(axis="y", labelsize=12, colors=AXIS_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1.1, alpha=0.9)
    ax.set_axisbelow(True)

    max_val = max(max(spymaster_values), max(guesser_values))
    ax.set_ylim(0, max_val + 8)

    if broad_last_idx >= 0 and broad_last_idx < len(regions) - 1:
        y_top = max_val + 6.4
        ax.text(
            broad_last_idx / 2,
            y_top,
            "Broad Regions",
            ha="center",
            va="center",
            fontsize=12.5,
            color="#3F5F7A",
            fontweight="bold",
            bbox=dict(facecolor="#DEEAF3", edgecolor="#B9CCDD", boxstyle="round,pad=0.22"),
        )
        ax.text(
            (broad_last_idx + 1 + len(regions) - 1) / 2,
            y_top,
            "Specialized Regions",
            ha="center",
            va="center",
            fontsize=12.5,
            color="#4C6E3A",
            fontweight="bold",
            bbox=dict(facecolor="#E3EFD3", edgecolor="#C5D8AA", boxstyle="round,pad=0.22"),
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS_COLOR)
        ax.spines[spine].set_linewidth(1.4)

    ax.legend(
        fontsize=13,
        loc="upper right",
        frameon=True,
        facecolor="whitesmoke",
        edgecolor=AXIS_COLOR,
    )

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="codenames BPP 결과 jsonl 경로",
    )
    args = parser.parse_args()

    if not args.log_path.exists():
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {args.log_path}")

    fig = create_figure(args.log_path)
    output_path = Path(__file__).with_suffix(".png")
    fig.savefig(output_path, dpi=300, facecolor=BACKGROUND_COLOR, bbox_inches="tight")
    print(f"Loaded log: {args.log_path}")
    print(f"Figure saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
