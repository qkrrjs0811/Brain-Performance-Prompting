import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Kaleido 패키지가 필요할 수 있습니다. 설치: pip install kaleido
# pio.kaleido.scope.mathjax = None

# --- 데이터 설정 (태스크별로 재구성) ---
plot_data = [
    {
        'task': 'Trivia.C.W(N=5)', 'row': 1, 'col': 1,
        'scores': {'Standard': 77.4, 'Macro-BPP': 81.0, 'Meso-BPP': 76.0, 'Micro-BPP': 78.4, 'BPP': 80.4}
    },
    {
        'task': 'Trivia.C.W(N=10)', 'row': 1, 'col': 2,
        'scores': {'Standard': 80.0, 'Macro-BPP': 81.3, 'Meso-BPP': 80.9, 'Micro-BPP': 78.5, 'BPP': 82.4}
    },
    {
        'task': 'Codenames.C', 'row': 2, 'col': 1,
        'scores': {'Standard': 77.5, 'Macro-BPP': 80.7, 'Meso-BPP': 80.3, 'Micro-BPP': 74.0, 'BPP': 83.0}
    },
    {
        'task': 'Logic.G.Puzzle', 'row': 2, 'col': 2,
        'scores': {'Standard': 68.0, 'Macro-BPP': 66.5, 'Meso-BPP': 70.5, 'Micro-BPP': 68.0, 'BPP': 72.0}
    }
]


# --- 2x2 서브플롯 생성 ---
fig = make_subplots(
    rows=2, cols=2,
    vertical_spacing=0.05,
    horizontal_spacing=0.1
)

# --- 각 서브플롯에 그래프 그리기 ---
bpp_types = ['Macro-BPP', 'Meso-BPP', 'Micro-BPP', 'BPP']
colors = ['lightblue', 'sandybrown', 'lightgreen', 'darkred']

for i, data in enumerate(plot_data):
    task_name = [data['task']]  # X축 레이블로 사용할 태스크 이름
    scores = data['scores']
    row = data['row']
    col = data['col']
    show_legend = (i == 0) # 첫 번째 서브플롯에만 범례 표시

    # BPP 막대 그래프 추가
    for bpp_type, color in zip(bpp_types, colors):
        score_val = [scores[bpp_type]]
        fig.add_trace(go.Bar(
            name=bpp_type,
            x=task_name,
            y=score_val,
            text=[f'<b>{s:.1f}</b>' for s in score_val],
            textposition='outside',
            marker_color=color,
            textfont=dict(size=18),
            showlegend=show_legend
        ), row=row, col=col)

    # Standard 기준선 및 점수 텍스트 추가
    standard_score = scores['Standard']
    fig.add_shape(type='line', x0=-0.4, x1=0.4, y0=standard_score, y1=standard_score,
                  line=dict(color='black', width=3, dash='dot'), row=row, col=col)
    fig.add_annotation(x=0, y=standard_score, text=f"<b>{standard_score:.1f}</b>",
                       showarrow=False, font=dict(size=16, color='black'),
                       yshift=10, xshift=-140, row=row, col=col)


# --- Standard 범례용 더미 trace 추가 ---
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                         line=dict(color='black', width=3, dash='dot'), name='Standard'))

# --- 레이아웃 설정 ---
ACL_WIDTH_SINGLE_COL = 975
ACL_HEIGHT_SINGLE_COL = 1200

fig.update_layout(
    barmode='group', bargap=0.2, bargroupgap=0.2,
    height=ACL_HEIGHT_SINGLE_COL, width=ACL_WIDTH_SINGLE_COL,
    plot_bgcolor='white',
    legend=dict(font_size=18, orientation='h', x=0.5, xanchor='center', y=1.05),
    margin=dict(l=80, r=40, t=100, b=80)
)

# --- 모든 서브플롯의 축 설정 ---
# X축 레이블(태스크 이름)을 다시 표시하고 폰트 크기 설정
fig.update_xaxes(tickfont=dict(size=20), showline=True, linewidth=1, linecolor='black')
fig.update_yaxes(
    title_text="Score (%)", range=[60, 90], # Y축 범위를 약간 늘려 텍스트가 잘리지 않게 함
    title_font=dict(size=18), tickfont=dict(size=18),
    showline=True, linewidth=1, linecolor='black',
    showgrid=True, gridwidth=1, gridcolor='lightgray'
)

# 최종 이미지 생성
fig.show()