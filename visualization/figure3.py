import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Kaleido 패키지가 필요할 수 있습니다. 설치: pip install kaleido
# pio.kaleido.scope.mathjax = None # MathJax 관련 오류 발생 시 주석 해제

# --- 데이터 설정 ---
tasks = ['Trivia.C.W<br>(N=5)', 'Trivia.C.W<br>(N=10)', 'Codenames.C', 'Logic.G.Puzzle']
models = ['GPT-4o', 'GPT-4o-mini', 'o1-mini', 'GPT-3.5-turbo', 'Qwen2.5-7B-Instruct', 'Llama-3.1-8B-Instruct']

standard = {
    'Trivia.C.W<br>(N=5)': [77.4, 52.4, 65.4, 50.6, 38.6, 62.8],
    'Trivia.C.W<br>(N=10)': [80.0, 53.6, 68.9, 49.3, 45.2, 67.8],
    'Codenames.C': [77.5, 75.8, 81.0, 71.5, 39.5, 28.8],
    'Logic.G.Puzzle': [68.0, 66.0, 92.5, 45.0, 10.0, 11.5]
}

spp = {
    'Trivia.C.W<br>(N=5)': [79.8, 61.8, 55.2, 51.8, 36.4, 53.4],
    'Trivia.C.W<br>(N=10)': [79.1, 63.7, 59.7, 29.0, 35.5, 49.5],
    'Codenames.C': [78.3, 70.0, 33.2, 67.8, 33.5, 54.8],
    'Logic.G.Puzzle': [69.5, 62.5, 74.0, 47.0, 23.0, 19.5]
}

bpp = {
    'Trivia.C.W<br>(N=5)': [80.4, 62.8, 55.0, 24.0, 33.6, 47.2],
    'Trivia.C.W<br>(N=10)': [82.4, 66.6, 59.8, 13.5, 36.7, 46.9],
    'Codenames.C': [83.0, 70.5, 57.3, 59.5, 51.2, 64.2],
    'Logic.G.Puzzle': [72.0, 61.0, 78.5, 43.0, 23.5, 25.0]
}

colors = {'Standard': 'lightskyblue', 'SPP': 'orange', 'BPP': 'darkred'}

# --- 폰트 크기 변수 선언 ---
legend_font_size = 16
bar_text_size = 14
axis_title_font_size = 23
axis_tick_font_size = 19
subplot_title_font_size = 25

# --- 서브플롯 생성 (2행 3열) ---
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f"<b>{model}</b>" for model in models],
    vertical_spacing=0.15,
    horizontal_spacing=0.05
)

# --- 각 모델에 대해 subplot에 그리기 ---
for idx, model in enumerate(models):
    row = idx // 3 + 1
    col = idx % 3 + 1
    
    x = list(standard.keys())
    y_standard = [standard[task][idx] for task in x]
    y_spp = [spp[task][idx] for task in x]
    y_bpp = [bpp[task][idx] for task in x]

    fig.add_trace(go.Bar(x=x, y=y_standard, name="Standard", marker_color=colors['Standard'], text=[f"<b>{val:.1f}</b>" if val is not None else "" for val in y_standard], textposition='outside', textfont=dict(size=bar_text_size), showlegend=(idx == 0)), row=row, col=col)
    fig.add_trace(go.Bar(x=x, y=y_spp, name="SPP", marker_color=colors['SPP'], text=[f"<b>{val:.1f}</b>" if val is not None else "" for val in y_spp], textposition='outside', textfont=dict(size=bar_text_size), showlegend=(idx == 0)), row=row, col=col)
    fig.add_trace(go.Bar(x=x, y=y_bpp, name="BPP", marker_color=colors['BPP'], text=[f"<b>{val:.1f}</b>" if val is not None else "" for val in y_bpp], textposition='outside', textfont=dict(size=bar_text_size), showlegend=(idx == 0)), row=row, col=col)

    # --- 축 스타일 업데이트 ---
    fig.update_xaxes(tickfont=dict(size=axis_tick_font_size), showline=True, linewidth=1, linecolor='black', row=row, col=col)
    fig.update_yaxes(tickfont=dict(size=axis_tick_font_size), showline=True, linewidth=1, linecolor='black', showgrid=True, gridwidth=1, gridcolor='lightgray', dtick=20, row=row, col=col)

# --- 전체 레이아웃 설정 ---
ACL_WIDTH_DOUBLE_COL = 2070
ACL_HEIGHT_DOUBLE_COL_2x3 = 1500

fig.update_layout(
    width=ACL_WIDTH_DOUBLE_COL,
    height=ACL_HEIGHT_DOUBLE_COL_2x3,
    barmode='group',
    bargap=0.2,
    bargroupgap=0.05,
    plot_bgcolor='white',
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=1.08,  # <<< 범례 y 위치를 위로 조정
        font=dict(size=legend_font_size)
    ),
    margin=dict(l=80, r=40, t=120, b=80) # <<< 위쪽 여백(t)을 늘려 공간 확보
)

# --- 서브플롯 제목 크기 조정 ---
for annotation in fig.layout.annotations:
    annotation.font.size = subplot_title_font_size

# Y축 제목 및 범위 설정
fig.update_yaxes(title_text="Score (%)", range=[0, 118], title_font=dict(size=axis_title_font_size))

# --- 고해상도 PNG 파일로 저장 ---
output_filename = "BPP_performance_ACL_2x3_final.png"
fig.write_image(output_filename)

print(f"범례 위치가 수정된 Figure가 '{output_filename}' 파일로 저장되었습니다.")
print(f"크기: {ACL_WIDTH_DOUBLE_COL}px (너비) x {ACL_HEIGHT_DOUBLE_COL_2x3}px (높이)")

fig.show()