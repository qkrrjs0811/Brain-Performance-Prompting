from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ===== 데이터 설정 =====
tasks = ["Trivia C.W (N=5)", "Trivia C.W (N=10)", "Codenames.C", "Logic.G.Puzzle"]
bpp_values = [80.4, 82.4, 83.0, 72.0]
bpp_w_r_demo = [78.2, 77.2, 78.3, 69.0]
bpp_w_k_demo = [79.4, 80.9, 79.7, 64.0]
bpp_two_r_demo = [77.6, 74.3, 76.5, 67.0]
bpp_two_k_demo = [77.8, 82.0, 80.0, 69.0]

# 평균 계산
one_demo_avg = [(r + k) / 2 for r, k in zip(bpp_w_r_demo, bpp_w_k_demo)]
two_demo_avg = [(r + k) / 2 for r, k in zip(bpp_two_r_demo, bpp_two_k_demo)]

# x 좌표 설정
base_x = np.arange(len(tasks))
scale = 1.5
x_vals = base_x * scale
width = 0.25
one_demo_x = x_vals - width
two_demo_x = x_vals
bpp_x = x_vals + width

# 색상 설정 (파스텔톤)
colors = {
    "BPP": "#D98989",      # pastel coral red
    "OneDemo": "#8DB7D8",  # pastel blue
    "TwoDemo": "#F2BC8E",  # pastel orange
    "R": "#9ED6A7",        # pastel green
    "K": "#E8A5C9"         # pastel pink
}

# ===== 스타일 설정 =====
plt.rcParams['font.family'] = ['DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

background_color = '#FFFFFF'
axis_color = '#5B6B7D'
grid_color = '#C9CED6'
text_color = '#2F3B4C'

legend_font_size = 12
bar_text_size = 12
axis_title_font_size = 14
axis_tick_font_size = 12

# ===== Figure 구성 =====
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
fig.patch.set_facecolor(background_color)
ax.set_facecolor(background_color)

b1 = ax.bar(one_demo_x, one_demo_avg, width=width * 0.92, color=colors["OneDemo"], label="BPP-One-Demo")
b2 = ax.bar(two_demo_x, two_demo_avg, width=width * 0.92, color=colors["TwoDemo"], label="BPP-Two-Demo")
b3 = ax.bar(bpp_x, bpp_values, width=width * 0.92, color=colors["BPP"], label="BPP")

# 막대 상단 점수 텍스트 (기존 위치 유지 + 겹침 대비 가독성 강화)
for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        x_text = rect.get_x() + rect.get_width() / 2
        y_text = h + 0.05
        if round(h, 1) in (78.8, 79.0):
            x_text -= 0.16
        if round(h, 1) == 77.7:
            y_text += 0.20
        if round(h, 1) == 68.0:
            y_text -= 0.20
        txt = ax.text(
            x_text,
            y_text,
            f'{h:.1f}',
            ha='center',
            va='bottom',
            fontsize=bar_text_size,
            color=text_color,
            fontweight='bold',
            zorder=6,
            clip_on=False,
            bbox=dict(facecolor=background_color, edgecolor='none', pad=0.15)
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground='white')])

# 산점도: One-Demo R/K
s1 = ax.scatter(one_demo_x, bpp_w_r_demo, color=colors["R"], s=70, marker='o', label="BPP-W-R-Demo", zorder=3)
s2 = ax.scatter(one_demo_x, bpp_w_k_demo, color=colors["K"], s=70, marker='o', label="BPP-W-K-Demo", zorder=3)

# 산점도: Two-Demo R/K
s3 = ax.scatter(two_demo_x, bpp_two_r_demo, color=colors["R"], s=75, marker='^', label="BPP-Two-R-Demo", zorder=3)
s4 = ax.scatter(two_demo_x, bpp_two_k_demo, color=colors["K"], s=75, marker='^', label="BPP-Two-K-Demo", zorder=3)

# 수직 점선 추가
for x, y1, y2 in zip(one_demo_x, bpp_w_r_demo, bpp_w_k_demo):
    ax.vlines(x, ymin=min(y1, y2), ymax=max(y1, y2), colors='#2D2D2D', linestyles='--', linewidth=1.4, zorder=2)
for x, y1, y2 in zip(two_demo_x, bpp_two_r_demo, bpp_two_k_demo):
    ax.vlines(x, ymin=min(y1, y2), ymax=max(y1, y2), colors='#2D2D2D', linestyles='--', linewidth=1.4, zorder=2)

# ===== 레이아웃 설정 =====
ax.set_xticks(x_vals)
ax.set_xticklabels(tasks, fontsize=axis_tick_font_size, color=axis_color)
ax.set_xlim(-1, x_vals[-1] + 1)
ax.set_ylim(60, 85)
ax.set_ylabel("Score (%)", fontsize=axis_title_font_size, color=text_color, fontweight='bold')
ax.tick_params(axis='y', labelsize=axis_tick_font_size, colors=axis_color)
ax.grid(axis='y', color=grid_color, linewidth=1.2, alpha=0.9)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color(axis_color)
    ax.spines[spine].set_linewidth(1.5)

handles = [b1, b2, b3, s1, s2, s3, s4]
labels = ["BPP-One-Demo", "BPP-Two-Demo", "BPP", "BPP-W-R-Demo", "BPP-W-K-Demo", "BPP-Two-R-Demo", "BPP-Two-K-Demo"]
ax.legend(
    handles,
    labels,
    fontsize=legend_font_size,
    loc='upper right',
    frameon=True,
    facecolor='whitesmoke',
    edgecolor=axis_color
)

# PNG 파일 저장 (스크립트 파일명과 동일)
plt.tight_layout()
script_path = Path(__file__).resolve()
output_filename = script_path.with_suffix('.png')
fig.savefig(output_filename, dpi=300, facecolor=background_color, bbox_inches='tight')

print(f"Figure가 '{output_filename}' 파일로 저장되었습니다.")

plt.show()