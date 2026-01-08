import plotly.graph_objects as go

# 데이터 설정
methods = ["BPP-W-K-Demo", "BPP-Two-K-Demo"]
recall_values = [0.944, 0.920]
accuracy_values = [0.844, 0.846]

# x 좌표 설정
x = [0, 1]

# bar 색상 설정
bar_colors = ['steelblue', 'orange']

# 그래프 생성
fig = go.Figure()

# --- Recall Bar ---
fig.add_trace(go.Bar(
    x=methods,
    y=recall_values,
    name='Recall',
    marker_color=bar_colors,
    text=[f"<b>{v:.3f}</b>" for v in recall_values],
    textposition='outside',
    textfont=dict(color='black', size=18)
))

# --- Dialogue Accuracy (점 + 점선 연결) ---
fig.add_trace(go.Scatter(
    x=methods,
    y=accuracy_values,
    mode='markers+lines+text',
    name='Dialogue Accuracy',
    marker=dict(color='red', size=10),
    line=dict(dash='dot', color='red'),
    text=[f"<b>{v:.3f}</b>" for v in accuracy_values],
    textposition='bottom center',
    textfont=dict(color='red', size=18)
))

# 레이아웃 설정
fig.update_layout(
    xaxis=dict(
        title="Method",
        title_font=dict(size=22),
        tickfont=dict(size=18)
    ),
    yaxis=dict(
        title="Recall",
        range=[0.8, 1.0],
        title_font=dict(size=22),
        tickfont=dict(size=18)
    ),
    legend=dict(
        orientation='h',
        font=dict(size=20),
        bordercolor='black',
        borderwidth=1,
        x=0.5,
        xanchor='center',
        y=0.98,
        yanchor='top'
    ),
    width=1000,
    height=900,
    template='plotly_white',
    margin=dict(l=60, r=40, t=50, b=60)
)

# 결과 출력 또는 저장
fig.show()