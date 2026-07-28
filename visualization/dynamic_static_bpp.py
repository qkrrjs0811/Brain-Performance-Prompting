from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))
import method_colors

# --- 데이터 설정 (태스크별로 재구성) ---
plot_data = [
    {
        'task': 'Trivia C.W (N=5)',
        'scores': {'Standard': 77.4, 'Macro-BPP': 81.0, 'Meso-BPP': 76.0, 'Micro-BPP': 78.4, 'BPP': 80.4}
    },
    {
        'task': 'Trivia C.W (N=10)',
        'scores': {'Standard': 80.0, 'Macro-BPP': 81.3, 'Meso-BPP': 80.9, 'Micro-BPP': 78.5, 'BPP': 82.4}
    },
    {
        'task': 'Codenames.C',
        'scores': {'Standard': 77.5, 'Macro-BPP': 80.7, 'Meso-BPP': 80.3, 'Micro-BPP': 74.0, 'BPP': 83.0}
    },
    {
        'task': 'Logic.G.Puzzle',
        'scores': {'Standard': 68.0, 'Macro-BPP': 66.5, 'Meso-BPP': 70.5, 'Micro-BPP': 68.0, 'BPP': 72.0}
    }
]


# --- 폰트/스타일 설정 ---
plt.rcParams['font.family'] = ['DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

bpp_types = ['Macro-BPP', 'Meso-BPP', 'Micro-BPP', 'BPP']
colors = {k: method_colors.SCALED_BPP_VARIANT_BAR_COLORS[k] for k in bpp_types}
BAR_EDGE = "#FFFFFF"
BAR_EDGEWIDTH = 1.25

background_color = '#FFFFFF'
axis_color = '#5B6B7D'
grid_color = '#C9CED6'
text_color = '#2F3B4C'

legend_font_size = 12
bar_text_size = 12
axis_title_font_size = 13
axis_tick_font_size = 11
subplot_title_font_size = 16


def build_figure(nrows: int, ncols: int, figsize: tuple[float, float]) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=100)
    fig.patch.set_facecolor(background_color)
    axes = np.atleast_1d(axes).flatten()

    x_idx = np.arange(len(bpp_types))
    bar_width = 0.62

    for i, data in enumerate(plot_data):
        ax = axes[i]
        ax.set_facecolor(background_color)
        scores = data['scores']

        y_vals = [scores[key] for key in bpp_types]
        bar_colors = [colors[key] for key in bpp_types]
        bars = ax.bar(
            x_idx, y_vals, width=bar_width, color=bar_colors,
            edgecolor=BAR_EDGE, linewidth=BAR_EDGEWIDTH,
        )

        for rect in bars:
            h = rect.get_height()
            value_text = ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.6,
                f'{h:.1f}',
                ha='center',
                va='bottom',
                fontsize=bar_text_size,
                color=text_color,
                fontweight='bold'
            )
            value_text.set_path_effects([pe.withStroke(linewidth=3.0, foreground='white')])

        standard_score = scores['Standard']
        std_line_c = "#000000"
        ax.axhline(standard_score, color=std_line_c, linewidth=2.2, linestyle=':')
        std_text = ax.text(
            -0.45,
            standard_score + 0.4,
            f'{standard_score:.1f}',
            fontsize=11,
            color=std_line_c,
            fontweight='bold',
            ha='left',
            va='bottom'
        )
        std_text.set_path_effects([pe.withStroke(linewidth=2.8, foreground='white')])

        ax.set_title(data['task'], fontsize=subplot_title_font_size, color=text_color, fontweight='bold', pad=10)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(bpp_types, fontsize=axis_tick_font_size, color=axis_color, rotation=0, ha="center")
        ax.tick_params(axis='y', labelsize=axis_tick_font_size, colors=axis_color)
        ax.set_ylim(60, 90)
        ax.set_yticks(np.arange(60, 91, 5))
        ax.grid(axis='y', linestyle='-', linewidth=1.1, color=grid_color, alpha=0.9)
        ax.set_axisbelow(True)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(axis_color)
            ax.spines[spine].set_linewidth(1.5)

        if i % ncols == 0:
            ax.set_ylabel('Score (%)', fontsize=axis_title_font_size, color=text_color, fontweight='bold')

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=colors[name],
            edgecolor="#5B6B7D",
            linewidth=0.9,
        )
        for name in bpp_types
    ]
    standard_handle = plt.Line2D(
        [0], [0],
        color="#000000",
        linewidth=2.2,
        linestyle=':',
    )
    legend_labels = bpp_types + ['Standard']
    legend_handles.append(standard_handle)

    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.975),
        ncol=5,
        frameon=False,
        fontsize=legend_font_size
    )

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    return fig, axes


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, facecolor=background_color, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    base = script_path.with_suffix("")

    fig_2x2, _ = build_figure(2, 2, (9.75, 12))
    out_2x2 = Path(f"{base}.png")
    save_figure(fig_2x2, out_2x2)
    print(f"Figure가 '{out_2x2}' 파일로 저장되었습니다.")

    fig_1x4, _ = build_figure(1, 4, (20.0, 5.75))
    out_1x4 = Path(f"{base}_1x4.png")
    save_figure(fig_1x4, out_1x4)
    print(f"Figure가 '{out_1x4}' 파일로 저장되었습니다.")
