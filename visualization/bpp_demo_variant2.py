from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))
import method_colors

# ===== 데이터 설정 (bpp_demo.py와 동일) =====
tasks = ["Trivia C.W (N=5)", "Trivia C.W (N=10)", "Codenames.C", "Logic.G.Puzzle"]
bpp_values = [80.4, 82.4, 83.0, 72.0]
bpp_w_r_demo = [78.2, 77.2, 78.3, 69.0]
bpp_w_k_demo = [79.4, 80.9, 79.7, 64.0]
bpp_two_r_demo = [77.6, 74.3, 76.5, 67.0]
bpp_two_k_demo = [77.8, 82.0, 80.0, 69.0]

methods = ["BPP", "BPP-W-R-Demo", "BPP-W-K-Demo", "BPP-Two-R-Demo", "BPP-Two-K-Demo"]
scores_by_method = {
    "BPP-W-R-Demo": bpp_w_r_demo,
    "BPP-W-K-Demo": bpp_w_k_demo,
    "BPP-Two-R-Demo": bpp_two_r_demo,
    "BPP-Two-K-Demo": bpp_two_k_demo,
    "BPP": bpp_values,
}
method_labels_short = {
    "BPP": "BPP",
    "BPP-W-R-Demo": "W-R",
    "BPP-W-K-Demo": "W-K",
    "BPP-Two-R-Demo": "Two-R",
    "BPP-Two-K-Demo": "Two-K",
}
method_markers = {
    "BPP-W-R-Demo": "o",
    "BPP-W-K-Demo": "o",
    "BPP-Two-R-Demo": "^",
    "BPP-Two-K-Demo": "^",
    "BPP": "D",
}

demo_methods = [m for m in methods if m != "BPP"]
plot_order_front = demo_methods + ["BPP"]

method_color_map = {
    "BPP-W-R-Demo": "#3D72AD",
    "BPP-W-K-Demo": "#B85D8F",
    "BPP-Two-R-Demo": "#4A9B5E",
    "BPP-Two-K-Demo": "#7E4FA0",
    "BPP": method_colors.METHOD_COLORS_DISPLAY["BPP"],
}

plt.rcParams["font.family"] = ["DejaVu Serif"]
plt.rcParams["axes.unicode_minus"] = False

background_color = "#FFFFFF"
panel_bg_color = "#FBFCFE"
axis_color = "#5B6B7D"
grid_color = "#C9CED6"
text_color = "#2F3B4C"
axis_title_size = 11.85
tick_size = 11
legend_size = 10.4
line_width = 2.05


def _panel_style(ax, *, vertical_grid=False, horizontal_grid=False):
    ax.set_facecolor(panel_bg_color)
    ax.tick_params(colors=axis_color, labelsize=tick_size)
    ax.set_axisbelow(True)
    if vertical_grid:
        ax.grid(axis="x", color=grid_color, linestyle="--", linewidth=0.95, alpha=0.92)
    if horizontal_grid:
        ax.grid(axis="y", color=grid_color, linestyle="--", linewidth=0.95, alpha=0.92)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(axis_color)
        ax.spines[s].set_linewidth(1.15)


fig, ax_s = plt.subplots(1, 1, dpi=120, figsize=(9.0, 5.55), facecolor=background_color)

_panel_style(ax_s, vertical_grid=True, horizontal_grid=True)

x = np.arange(len(tasks))
all_scores_flat = np.array([scores_by_method[m][t] for m in plot_order_front for t in range(len(tasks))])

for method in plot_order_front:
    y = scores_by_method[method]
    c = method_color_map[method]
    is_bpp = method == "BPP"
    lw = line_width
    ms = 8.4 if is_bpp else 6.45
    a = 1.0 if is_bpp else 0.9
    z = 6 if is_bpp else 3
    ax_s.plot(
        x,
        y,
        linestyle="-" if is_bpp else (0, (5, 3)),
        linewidth=lw,
        color=c,
        alpha=a,
        marker=method_markers[method],
        markersize=ms,
        markerfacecolor=c if is_bpp else "#F5F8FC",
        markeredgewidth=1.25 if not is_bpp else 1.2,
        markeredgecolor=method_colors.BPP_MARKER_EDGE if is_bpp else c,
        label=method_labels_short[method],
        zorder=z,
    )

ymax = float(np.ceil(all_scores_flat.max() + 0.85))
ymin = float(np.floor(all_scores_flat.min() - 0.95))
ymax = max(ymax, ymin + 6.5)
y0 = (int(np.floor(ymin)) // 5) * 5
y1 = ((int(np.ceil(ymax)) + 4) // 5) * 5 + 5
yticks_score = np.arange(y0, y1 + 1e-9, step=5, dtype=float)
ax_s.set_xticks(x)
ax_s.set_xticklabels(tasks, rotation=0, ha="center")
ax_s.set_ylim(ymin, ymax)
ax_s.set_yticks(yticks_score)
ax_s.set_xlabel("")
ax_s.set_ylabel("Score (%)", fontsize=axis_title_size, color=text_color, fontweight="bold")

_handles, _lbls = ax_s.get_legend_handles_labels()
ax_s.legend(
    _handles,
    _lbls,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5,
    frameon=True,
    facecolor="white",
    edgecolor=axis_color,
    fontsize=legend_size,
)

plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.86])

script_path = Path(__file__).resolve()
output_filename = script_path.with_suffix(".png")
fig.savefig(output_filename, dpi=300, facecolor=background_color, bbox_inches="tight")

print(f"Figure가 '{output_filename}' 파일로 저장되었습니다.")

plt.show()
