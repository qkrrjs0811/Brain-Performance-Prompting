import plotly.graph_objects as go
import numpy as np

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

# 색상 설정
colors = {
    "BPP": "darkred",
    "OneDemo": "#1f77b4",
    "TwoDemo": "orange",
    "R": "lime",
    "K": "deeppink"
}

# ===== Figure 구성 =====
fig = go.Figure()

# --- 함수: 수직 점선 추가용 ---
def make_vertical_lines(x_coords, y1_list, y2_list):
    return [
        dict(type='line', x0=x, x1=x, y0=y1, y1=y2,
             xref='x', yref='y',
             line=dict(color='black', dash='dash', width=1.5))
        for x, y1, y2 in zip(x_coords, y1_list, y2_list)
    ]

# --- 막대: One-Demo 평균
fig.add_trace(go.Bar(
    x=one_demo_x, y=one_demo_avg,
    name="BPP-One-Demo",
    marker_color=colors["OneDemo"],
    text=[f"<b>{v:.1f}</b>" for v in one_demo_avg],
    textposition='none',
    textfont=dict(size=20)
))

# 수동 텍스트 추가 (One-Demo 평균)
for x, y in zip(one_demo_x, one_demo_avg):
    fig.add_annotation(
        x= x - 0.3, y= y + 0.3,
        text=f"<b>{y:.1f}</b>", showarrow=False,
        font=dict(size=20), xanchor="center", xshift=8
    )

# --- 막대: Two-Demo 평균
fig.add_trace(go.Bar(
    x=two_demo_x, y=two_demo_avg,
    name="BPP-Two-Demo",
    marker_color=colors["TwoDemo"],
    text=[f"<b>{v:.1f}</b>" for v in two_demo_avg],
    textposition='none',
    textfont=dict(size=20)
))

# --- Two-Demo 평균 bar 위에 텍스트 수동 추가 ---
for x, y in zip(two_demo_x, two_demo_avg):
    fig.add_annotation(
        x=x, y=y + 0.5,
        text=f"<b>{y:.1f}</b>", showarrow=False,
        font=dict(size=20), xanchor="center"
    )

# --- 막대: BPP 기준선
fig.add_trace(go.Bar(
    x=bpp_x, y=bpp_values,
    name="BPP",
    marker_color=colors["BPP"],
    text=[f"<b>{v:.1f}</b>" for v in bpp_values],
    textposition='none',
    textfont=dict(size=20)
))

# --- BPP bar 위에 텍스트 수동 추가 ---
for x, y in zip(bpp_x, bpp_values):
    fig.add_annotation(
        x=x, y=y + 0.5,
        text=f"<b>{y:.1f}</b>", showarrow=False,
        font=dict(size=20), xanchor="center"
    )

# --- 산점도: One-Demo R/K
fig.add_trace(go.Scatter(
    x=one_demo_x, y=bpp_w_r_demo,
    mode='markers', name="BPP-W-R-Demo",
    marker=dict(color=colors["R"], size=11, symbol='circle'),
    textfont=dict(size=20)
))
fig.add_trace(go.Scatter(
    x=one_demo_x, y=bpp_w_k_demo,
    mode='markers', name="BPP-W-K-Demo",
    marker=dict(color=colors["K"], size=11, symbol='circle'),
    textfont=dict(size=20)
))

# --- 산점도: Two-Demo R/K
fig.add_trace(go.Scatter(
    x=two_demo_x, y=bpp_two_r_demo,
    mode='markers', name="BPP-Two-R-Demo",
    marker=dict(color=colors["R"], size=11, symbol='triangle-up'),
    textfont=dict(size=20)
))
fig.add_trace(go.Scatter(
    x=two_demo_x, y=bpp_two_k_demo,
    mode='markers', name="BPP-Two-K-Demo",
    marker=dict(color=colors["K"], size=11, symbol='triangle-up'),
    textfont=dict(size=20)
))

# --- 수직 점선 추가 ---
fig.update_layout(shapes=
    make_vertical_lines(one_demo_x, bpp_w_r_demo, bpp_w_k_demo) +
    make_vertical_lines(two_demo_x, bpp_two_r_demo, bpp_two_k_demo)
)

# ===== 레이아웃 설정 =====
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=x_vals,
        ticktext=tasks,
        tickfont=dict(size=20),
        range=[-1, x_vals[-1] + 1],
        title_font=dict(size=24)
    ),
    yaxis=dict(
        title="Score (%)",
        range=[60, 85],
        tickfont=dict(size=20),
        title_font=dict(size=24),
        gridcolor='lightgray',
        gridwidth=1.2
    ),
    barmode='group',
    bargap=0.15,         # 막대 그룹 간 간격
    bargroupgap=0.05,    # 그룹 내(bar끼리) 간격
    width=1200,
    height=600,
    legend=dict(
        orientation='v',
        font=dict(size=20),
        x=0.99,
        xanchor='right',
        y=0.99,
        yanchor='top',
        itemsizing='constant',
        itemwidth=60,
        borderwidth=1,
        bgcolor="whitesmoke",
        bordercolor="black"
    ),
    margin=dict(l=60, r=30, t=60, b=20),
    template='plotly_white'
)


# 이미지 저장
fig.show()