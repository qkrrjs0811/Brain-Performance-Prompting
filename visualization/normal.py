import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_task_data():
    """각 태스크별 데이터를 로드하고 방법별 평균 정확도를 계산합니다."""
    tasks = {
        'trivia_creative_writing': {
            'n_5': 'Trivia.C.W (N=5)',
            'n_10': 'Trivia.C.W (N=10)'
        },
        'codenames_collaborative': {
            'default': 'Codenames.C'
        },
        'logic_grid_puzzle': {
            'default': 'Logic.G.Puzzle'
        }
    }
    
    models = ['gpt-4.1', 'gpt-4.1-mini']
    methods = ['standard', 'spp', 'bpp']  # 이미지에서 사용된 3가지 방법만 사용
    
    data = {}
    
    for model in models:
        data[model] = {}
        
        for task_name, task_variants in tasks.items():
            file_path = f'logs/{task_name}/{model}_wo_sys_mes/accuracy_results_{model}_wo_sys_mes.xlsx'
            
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                
                for variant_key, display_name in task_variants.items():
                    if variant_key == 'default':
                        # codenames, logic의 경우 전체 데이터 사용
                        task_data = df
                    else:
                        # trivia의 경우 N=5, N=10 구분
                        if variant_key == 'n_5':
                            task_data = df[df['File'].str.contains('_n_5.jsonl')]
                        elif variant_key == 'n_10':
                            task_data = df[df['File'].str.contains('_n_10.jsonl')]
                    
                    # 방법별 평균 정확도 계산
                    variant_data = {}
                    for method in methods:
                        method_data = task_data[task_data['Method'] == method]
                        if not method_data.empty:
                            variant_data[method] = method_data['Accuracy'].mean()
                        else:
                            variant_data[method] = 0.0
                    
                    data[model][display_name] = variant_data
            else:
                print(f"경고: {file_path} 파일을 찾을 수 없습니다.")
    
    return data

def create_visualization():
    """첨부 이미지와 같은 스타일로 시각화를 생성합니다."""
    data = load_task_data()
    
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
    models = ['gpt-4.1', 'gpt-4.1-mini']
    task_order = ['Trivia.C.W (N=5)', 'Trivia.C.W (N=10)', 'Codenames.C', 'Logic.G.Puzzle']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for model_idx, model in enumerate(models):
        ax = axes[model_idx]
        x = np.arange(len(task_order))
        width = 0.25
        
        for i, method in enumerate(methods):
            values = []
            for task in task_order:
                if task in data[model]:
                    values.append(data[model][task].get(method, 0) * 100)
                else:
                    values.append(0.0)
            
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
        if model_idx == 0:  # Y축 레이블은 왼쪽에만 표시
            ax.set_ylabel('Score (%)', fontsize=14)
            
        ax.set_title(f'{model.upper()}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x) # 막대 그룹 중앙에 틱 설정
        ax.set_xticklabels(task_order, rotation=0, ha='center')
        ax.set_ylim(0, 105) # 텍스트 공간 확보
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f5f5f5')

    # ⭐ 수정 1: 전체 제목 위치를 y=0.98로 명확하게 지정
    fig.suptitle('BPP Performance Comparison Across Several Models', fontsize=18, fontweight='bold', y=0.98)
    
    # ⭐ 수정 2: 범례 위치를 y=0.88로 내려 서브플롯 타이틀과 겹치지 않게 함
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=3, fontsize=12, frameon=True)
    
    # ⭐ 수정 3: tight_layout() 대신 subplots_adjust를 사용하여 레이아웃 직접 제어
    # top: 서브플롯 상단 위치를 내려 제목/범례 공간 확보, wspace: 서브플롯 간 좌우 간격 조정
    plt.subplots_adjust(top=0.78, wspace=0.15)
    
    return fig

def main():
    """메인 함수"""
    print("Normal task 데이터 로딩 중...")
    
    # 시각화 생성
    fig = create_visualization()
    
    if fig:
        print("시각화 생성 완료!")
        
        # PNG 파일로 저장
        output_path = 'normal_performance_comparison.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"결과가 {output_path}에 저장되었습니다.")
        
        # 화면에 표시
        plt.show()
    else:
        print("시각화 생성에 실패했습니다.")

if __name__ == "__main__":
    main()
