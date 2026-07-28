from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

_VIZ_DIR = Path(__file__).resolve().parent
if str(_VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(_VIZ_DIR))
import method_colors

# --- 데이터 설정 ---
# 오픈 모델 행(temp=0): result_all.py, logs ..._temp-0_wo_sys_mes 평균 정확도(%)
tasks = ['Trivia.C.W<br>(N=5)', 'Trivia.C.W<br>(N=10)', 'Codenames.C', 'Logic.G.Puzzle']
models = [
    'GPT-4o',
    'GPT-4o-mini',
    'o1-mini',
    'Qwen2.5-7B-Instruct',
    'Qwen2.5-14B-Instruct',
    'Llama-3.1-8B-Instruct',
]

standard = {
    'Trivia.C.W<br>(N=5)': [77.4, 52.4, 65.4, 38.2, 52.4, 63.6],
    'Trivia.C.W<br>(N=10)': [80.0, 53.6, 68.9, 45.8, 57.9, 68.2],
    'Codenames.C': [77.5, 75.8, 81.0, 61.3, 70.2, 24.2],
    'Logic.G.Puzzle': [68.0, 66.0, 92.5, 51.0, 51.0, 27.5],
}

spp = {
    'Trivia.C.W<br>(N=5)': [79.8, 61.8, 55.2, 36.6, 51.8, 54.2],
    'Trivia.C.W<br>(N=10)': [79.1, 63.7, 59.7, 38.0, 55.6, 48.9],
    'Codenames.C': [78.3, 70.0, 33.2, 63.7, 57.0, 44.5],
    'Logic.G.Puzzle': [69.5, 62.5, 74.0, 18.5, 48.0, 23.0],
}

bpp = {
    'Trivia.C.W<br>(N=5)': [80.4, 62.8, 55.0, 32.8, 51.2, 46.0],
    'Trivia.C.W<br>(N=10)': [82.4, 66.6, 59.8, 34.9, 54.1, 48.4],
    'Codenames.C': [83.0, 70.5, 57.3, 57.5, 40.7, 61.8],
    'Logic.G.Puzzle': [72.0, 61.0, 78.5, 17.0, 56.0, 24.0],
}

# 방법론 색상 (cost_comparison / method_colors 기준)
colors = {
    'Standard': method_colors.METHOD_COLORS_DISPLAY['Standard'],
    'SPP': method_colors.METHOD_COLORS_DISPLAY['SPP'],
    'BPP': method_colors.METHOD_COLORS_DISPLAY['BPP'],
}

# --- 폰트/스타일 설정 ---
# 시스템 기본 가용 폰트만 사용 (대부분 환경에서 기본 제공)
plt.rcParams['font.family'] = ['DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

legend_font_size = 14
bar_text_size = 11
axis_title_font_size = 13
axis_tick_font_size = 11
subplot_title_font_size = 16

background_color = '#FFFFFF'
axis_color = '#5B6B7D'
grid_color = '#C9CED6'
text_color = '#2F3B4C'

# x축 라벨은 HTML 태그 제거 버전 사용
tasks_clean = ['Trivia C.W\n(N=5)', 'Trivia C.W\n(N=10)', 'Codenames.C', 'Logic.G.Puzzle']
tasks_raw = list(standard.keys())

# --- 서브플롯 생성 (2행 3열) ---
fig, axes = plt.subplots(2, 3, figsize=(20.7, 15), dpi=100)
fig.patch.set_facecolor(background_color)
axes = axes.flatten()

bar_offset = 0.24
bar_width = 0.22
x_idx = np.arange(len(tasks_clean))

for idx, model in enumerate(models):
    ax = axes[idx]
    ax.set_facecolor(background_color)

    y_standard = [standard[task][idx] for task in tasks_raw]
    y_spp = [spp[task][idx] for task in tasks_raw]
    y_bpp = [bpp[task][idx] for task in tasks_raw]

    b1 = ax.bar(x_idx - bar_offset, y_standard, width=bar_width, color=colors['Standard'], label='Standard')
    b2 = ax.bar(x_idx, y_spp, width=bar_width, color=colors['SPP'], label='SPP')
    b3 = ax.bar(x_idx + bar_offset, y_bpp, width=bar_width, color=colors['BPP'], label='BPP')

    for bars in (b1, b2, b3):
        for rect in bars:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + 1.2,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=bar_text_size,
                color=text_color,
                fontweight='bold'
            )

    ax.set_title(model, fontsize=subplot_title_font_size, fontweight='bold', color=text_color, pad=10)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(tasks_clean, fontsize=axis_tick_font_size, color=axis_color)
    ax.tick_params(axis='y', labelsize=axis_tick_font_size, colors=axis_color)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', linestyle='-', linewidth=1.2, color=grid_color, alpha=0.9)
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(axis_color)
        ax.spines[spine].set_linewidth(1.6)

    if idx % 3 == 0:
        ax.set_ylabel('Score (%)', fontsize=axis_title_font_size, color=text_color, fontweight='bold')

# --- 공통 범례 설정 ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.985),
    ncol=3,
    frameon=False,
    fontsize=legend_font_size
)

plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.94])

# --- PNG 파일 저장 (스크립트 파일명과 동일) ---
script_path = Path(__file__).resolve()
output_filename = script_path.with_suffix('.png')
fig.savefig(output_filename, dpi=300, facecolor=background_color, bbox_inches='tight')

print(f"Figure가 '{output_filename}' 파일로 저장되었습니다.")
print("크기: 2070px (너비) x 1500px (높이), 저장 DPI: 300")

plt.show()