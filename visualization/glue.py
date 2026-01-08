import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def load_glue_data():
    """GLUE subset별 데이터를 로드하고 방법별 평균 정확도를 계산합니다."""
    subsets = ['cola', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2']
    methods = ['bpp', 'spp', 'standard']
    
    data = {}
    
    for subset in subsets:
        file_path = f'logs/glue_{subset}/gpt-4o-2024-08-06_wo_sys_mes/glue_{subset}_accuracy_results_gpt-4o_wo_sys_mes.xlsx'
        
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            
            # 방법별 평균 정확도 계산
            subset_data = {}
            for method in methods:
                method_data = df[df['Method'] == method]
                if not method_data.empty:
                    subset_data[method] = method_data['Accuracy'].mean()
                else:
                    subset_data[method] = 0.0
            
            data[subset.upper()] = subset_data
        else:
            print(f"경고: {file_path} 파일을 찾을 수 없습니다.")
    
    return data

def create_glue_visualization():
    """GLUE subset별 성능을 시각화합니다."""
    data = load_glue_data()
    
    if not data:
        print("데이터를 로드할 수 없습니다.")
        return
    
    # 색상 정의 (이미지 참고)
    colors = {
        'standard': '#1f77b4',  # 밝은 파란색
        'spp': '#ff7f0e',       # 주황색  
        'bpp': '#d62728'        # 짙은 빨간색
    }
    
    methods = ['standard', 'spp', 'bpp']
    method_labels = ['Standard', 'SPP', 'BPP']
    
    # 데이터 준비
    subsets = list(data.keys())
    x_positions = []
    y_values = []
    colors_list = []
    text_values = []
    
    # 각 subset에 대해 데이터 준비
    for i, subset in enumerate(subsets):
        subset_data = data[subset]
        for j, method in enumerate(methods):
            x_positions.append(f"{subset}")
            y_values.append(subset_data.get(method, 0) * 100)  # 백분율로 변환
            colors_list.append(colors[method])
            text_values.append(f'{subset_data.get(method, 0)*100:.1f}')
    
    # 단일 막대 그래프 생성
    fig = go.Figure()
    
    # 각 방법별로 그룹화된 막대 그래프 생성
    for i, method in enumerate(methods):
        method_x = []
        method_y = []
        method_text = []
        
        for j, subset in enumerate(subsets):
            method_x.append(subset)
            method_y.append(data[subset].get(method, 0) * 100)
            method_text.append(f'{data[subset].get(method, 0)*100:.1f}')
        
        fig.add_trace(go.Bar(
            x=method_x,
            y=method_y,
            name=method_labels[i],
            marker_color=colors[method],
            text=method_text,
            textposition='auto',
            hovertemplate=f'<b>{method_labels[i]}</b><br>' +
                         'Task: %{x}<br>' +
                         'Score: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))
    
    # 레이아웃 업데이트
    fig.update_layout(
        title={
            'text': 'GLUE Task별 Method 성능 비교',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Task",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        height=600,
        width=1000,
        barmode='group',  # 그룹화된 막대 그래프
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(240,240,240,0.5)',  # 연한 회색 배경
        font=dict(size=12)
    )
    
    # x축 레이블 회전
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_matplotlib_visualization():
    """matplotlib을 사용하여 GLUE subset별 성능을 시각화합니다."""
    data = load_glue_data()
    
    if not data:
        print("데이터를 로드할 수 없습니다.")
        return None
    
    colors = {
        'standard': '#1f77b4',
        'spp': '#ff7f0e',
        'bpp': '#d62728'
    }
    
    methods = ['standard', 'spp', 'bpp']
    method_labels = ['Standard', 'SPP', 'BPP']
    subsets = list(data.keys())
    
    # ⭐ 변경 사항 1: figsize을 살짝 조정하여 상단 여유 공간 확보
    fig, ax = plt.subplots(figsize=(12, 8.5)) 
    
    # ⭐ 변경 사항 2: ax.set_title 대신 fig.suptitle() 사용하여 전체 그림의 최상단에 제목 배치
    fig.suptitle('GLUE Task Performance Comparison by Method', fontsize=18, fontweight='bold')

    x = np.arange(len(subsets))
    width = 0.25
    
    for i, method in enumerate(methods):
        values = [data[subset].get(method, 0) * 100 for subset in subsets]
        bars = ax.bar(x + (i - 1) * width, values, width, 
                      label=method_labels[i], 
                      color=colors[method],
                      alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Task', fontsize=14)
    ax.set_ylabel('Score (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, rotation=45, ha='right')
    ax.set_ylim(0, 105) 
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # ⭐ 변경 사항 3: 범례 위치를 타이틀 아래, 그래프 위로 재조정
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), # y 좌표를 1.0으로 변경
              ncol=3, fancybox=True, shadow=True, fontsize=12)
    
    ax.set_facecolor('#f5f5f5')
    
    # ⭐ 변경 사항 4: tight_layout()에서 rect 인자 제거하여 자동 조정
    plt.tight_layout()

    # suptitle과 legend 공간을 고려하여 레이아웃 추가 조정
    plt.subplots_adjust(top=0.92)
    
    return fig

def main():
    """메인 함수"""
    print("GLUE 데이터 로딩 중...")
    
    fig = create_matplotlib_visualization()
    
    if fig:
        print("시각화 생성 완료!")
        
        output_path = 'glue_performance_comparison.png'
        fig.savefig(output_path, dpi=300) # bbox_inches='tight' 제거, plt.tight_layout()으로 대체
        print(f"결과가 {output_path}에 저장되었습니다.")
        
        plt.show()
    else:
        print("시각화 생성에 실패했습니다.")

if __name__ == "__main__":
    main()
