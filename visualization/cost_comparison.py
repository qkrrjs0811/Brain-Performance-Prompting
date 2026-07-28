from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))
import method_colors

# -------------------------
# 1) 데이터 설정
# -------------------------
methods = list(method_colors.METHOD_ORDER_DISPLAY)
costs = np.array([0.4592, 0.5993, 1.93, 1.1965, 1.3627])  # 비용($)
accuracies = np.array([77.4, 70.4, 80.4, 79.8, 80.4])  # 정확도(%)
colors = list(method_colors.METHOD_COLORS_HEX)
scaled_sizes = [1200, 1200, 1200, 1200, 1200]  # 마커 면적(points^2)

# -------------------------
# 0) 폰트 크기 변수 선언
# -------------------------
title_font_size = 24
legend_font_size = 12
axis_title_font_size = 14
axis_tick_font_size = 12
bubble_text_size = 11

# -------------------------
# 0-1) 공통 스타일
# -------------------------
plt.rcParams['font.family'] = ['DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

background_color = '#FFFFFF'
axis_color = '#5B6B7D'
grid_color = '#C9CED6'
text_color = '#2F3B4C'

# -------------------------
# 2) 버블 차트 그리기
# -------------------------
fig, ax = plt.subplots(figsize=(11, 8), dpi=100)
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

# 텍스트 라벨 오프셋(겹침 완화)
offsets = {
    "Standard": (-0.045, 0.35),
    "CoT": (0.045, -0.6),
    "Self Refine": (0.0, 0.35),
    "SPP": (-0.055, 0.2),
    "BPP": (0.06, -0.35),
}

for i, method in enumerate(methods):
    point_color = colors[i]
    point_alpha = 0.82
    edge_color = 'white'
    label_color = text_color
    point_size = scaled_sizes[i]
    line_width = 1.8

    # CoT는 Pareto 논의의 핵심이 아니므로 시선 분산을 줄이기 위해 디밍 처리
    if method == "CoT":
        point_color = method_colors.COT_SCATTER_DIMMED
        point_alpha = 0.55
        edge_color = method_colors.COT_SCATTER_DIMMED
        label_color = '#4A5771'
    elif method == "BPP":
        point_alpha = 0.98
        edge_color = method_colors.BPP_MARKER_EDGE
        point_size = scaled_sizes[i] * 1.22
        line_width = 2.6

    ax.scatter(
        costs[i],
        accuracies[i],
        s=point_size,
        c=point_color,
        alpha=point_alpha,
        edgecolors=edge_color,
        linewidths=line_width,
        zorder=3
    )

    dx, dy = offsets[method]
    txt = ax.text(
        costs[i] + dx,
        accuracies[i] + dy,
        method,
        fontsize=bubble_text_size,
        color=label_color,
        fontweight=('heavy' if method == "BPP" else 'bold'),
        ha='center',
        va='center',
        zorder=4
    )
    txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground='white')])

# Pareto Frontier: Standard -> SPP -> BPP -> Self Refine
method_to_idx = {method: i for i, method in enumerate(methods)}
pareto_order = ["Standard", "SPP", "BPP", "Self Refine"]
pareto_x = [costs[method_to_idx[m]] for m in pareto_order]
pareto_y = [accuracies[method_to_idx[m]] for m in pareto_order]
# 의미 있는 투자 구간: Standard -> SPP -> BPP (진한 실선)
ax.plot(
    pareto_x[:3],
    pareto_y[:3],
    color='#A8C8FF',
    linewidth=3.0,
    linestyle='-',
    marker='o',
    markersize=5.5,
    markerfacecolor='#A8C8FF',
    markeredgecolor='white',
    zorder=2.7
)

# 한계효용 구간: BPP -> Self Refine (옅은 점선)
ax.plot(
    pareto_x[2:],
    pareto_y[2:],
    color='#FF9BA7',
    linewidth=2.6,
    linestyle='--',
    marker='o',
    markersize=5.5,
    markerfacecolor='#FF9BA7',
    markeredgecolor='white',
    zorder=2.6
)

# BPP 이후 한계효용 감소(곡선 평탄화) 강조 주석
ax.annotate(
    'Sweet Spot (Optimal trade-off)',
    xy=(costs[method_to_idx["BPP"]], accuracies[method_to_idx["BPP"]]),
    xytext=(costs[method_to_idx["BPP"]] - 0.12, accuracies[method_to_idx["BPP"]] + 1.05),
    fontsize=10.5,
    color='#2E5A24',
    ha='left',
    va='bottom',
    arrowprops=dict(arrowstyle='->', lw=1.4, color='#73A95C'),
    bbox=dict(facecolor='#DDF4C8', edgecolor='#8FBE74', boxstyle='round,pad=0.28'),
    zorder=5
)

# 수평선 오해 방지: BPP 이후의 아주 작은 성능 증가 대비 높은 비용 명시
mid_x = (costs[method_to_idx["BPP"]] + costs[method_to_idx["Self Refine"]]) / 2
mid_y = (accuracies[method_to_idx["BPP"]] + accuracies[method_to_idx["Self Refine"]]) / 2
ax.annotate(
    'Cost jumps sharply,\nbut same Accuracy',
    xy=(mid_x, mid_y),
    xytext=(mid_x - 0.08, mid_y - 0.65),
    fontsize=10.5,
    color='#7A1F2B',
    ha='left',
    va='top',
    arrowprops=dict(arrowstyle='->', lw=1.3, color='#D98995'),
    bbox=dict(facecolor='#FDE4E7', edgecolor='#D98995', boxstyle='round,pad=0.22'),
    zorder=5
)

# -------------------------
# 3) 레이아웃 설정
# -------------------------
ax.set_xlabel('Cost ($)', fontsize=axis_title_font_size, color=text_color, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=axis_title_font_size, color=text_color, fontweight='bold')
ax.tick_params(axis='both', labelsize=axis_tick_font_size, colors=axis_color)
ax.grid(True, color=grid_color, linewidth=1.15, alpha=0.9)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color(axis_color)
    ax.spines[spine].set_linewidth(1.5)

ax.set_xlim(costs.min() - 0.18, costs.max() + 0.12)
ax.set_ylim(accuracies.min() - 2.0, accuracies.max() + 1.8)

legend_handles = [
    Line2D(
        [0], [0],
        marker='o',
        linestyle='',
        markersize=10,
        markerfacecolor=(
            method_colors.COT_LEGEND_DIMMED if methods[i] == 'CoT' else colors[i]
        ),
        markeredgecolor=(
            '#D7DBE2' if methods[i] == 'CoT' else 'white'
        ),
        markeredgewidth=1.2,
        label=methods[i]
    )
    for i in range(len(methods))
]
legend = ax.legend(
    handles=legend_handles,
    fontsize=legend_font_size,
    loc='upper left',
    frameon=True,
    facecolor='whitesmoke',
    edgecolor=axis_color
)
for text in legend.get_texts():
    text.set_color(text_color)

plt.tight_layout()

# PNG 파일 저장 (스크립트 파일명과 동일)
script_path = Path(__file__).resolve()
output_filename = script_path.with_suffix('.png')
fig.savefig(output_filename, dpi=300, facecolor=background_color, bbox_inches='tight')

print(f"Figure가 '{output_filename}' 파일로 저장되었습니다.")

plt.show()
