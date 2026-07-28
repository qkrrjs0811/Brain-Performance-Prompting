import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse

# ==================== 전역 설정 (여기서 한 번만 수정하면 전체에 적용됨) ====================
MODELS = ['gpt-5.4-nano', 'gpt-5.2']  # 사용할 모델 리스트

METHODS = ['standard', 'spp', 'bpp']  # 사용할 방법 리스트 (원하는 만큼 추가 가능)

# 방법별 표시 레이블 (METHODS와 동일한 순서로 매핑)
METHOD_LABELS = {
    'standard': 'Standard',
    'spp': 'SPP',
    'bpp': 'BPP'
    # 새로운 method 추가 시 여기에 레이블 추가
}

# 방법별 색상 (METHODS에 포함된 모든 method에 대한 색상 정의 필요)
METHOD_COLORS = {
    'standard': '#1f77b4',
    'spp': '#ff7f0e',
    'bpp': '#d62728'
    # 새로운 method 추가 시 여기에 색상 추가 (예: '#2ca02c')
}
# ==================================================================================

def extract_run_number(file_name):
    """파일명에서 run 번호를 추출합니다. 없으면 None을 반환합니다."""
    if not isinstance(file_name, str):
        return None
    match = re.search(r"(?:^|[_-])run[_-]?(\d+)(?:[_-]|\.|$)", file_name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def filter_by_run(df, run):
    """run 인자에 따라 DataFrame을 필터링합니다."""
    if 'File' not in df.columns:
        return df

    run_series = df['File'].apply(extract_run_number)
    if run == 0:
        # 기본 형태(= run 정보가 없는 파일명)만 사용
        return df[run_series.isna()]
    return df[run_series == run]


def load_task_data(run=0):
    """각 태스크별 데이터를 로드하고 방법별 평균 정확도를 계산합니다."""
    tasks = {
        'trivia_creative_writing': {
            'n_5': 'Trivia C.W (N=5)',
            'n_10': 'Trivia C.W (N=10)'
        },
        'codenames_collaborative': {
            'default': 'Codenames.C'
        },
        'logic_grid_puzzle': {
            'default': 'Logic.G.Puzzle'
        }
    }
    
    data = {}
    
    for model in MODELS:
        data[model] = {}
        
        for task_name, task_variants in tasks.items():
            file_path = f'logs/{task_name}/{model}_wo_sys_mes/accuracy_results_{model}_wo_sys_mes.xlsx'
            
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                df = filter_by_run(df, run)
                if df.empty:
                    print(f"경고: {file_path}에서 run={run} 조건에 맞는 데이터가 없습니다.")
                    continue
                
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
                    for method in METHODS:
                        method_data = task_data[task_data['Method'] == method]
                        if not method_data.empty:
                            variant_data[method] = method_data['Accuracy'].mean()
                        else:
                            variant_data[method] = 0.0
                    
                    data[model][display_name] = variant_data
            else:
                print(f"경고: {file_path} 파일을 찾을 수 없습니다.")
    
    return data

def create_visualization(run=0):
    """첨부 이미지와 같은 스타일로 시각화를 생성합니다."""
    data = load_task_data(run=run)
    
    if not data:
        print("데이터를 로드할 수 없습니다.")
        return None
    
    task_order = ['Trivia C.W (N=5)', 'Trivia C.W (N=10)', 'Codenames.C', 'Logic.G.Puzzle']
    
    # METHODS에 대한 레이블 리스트 생성
    method_labels = [METHOD_LABELS.get(method, method.capitalize()) for method in METHODS]
    
    # squeeze=False를 사용하여 axes를 항상 배열로 반환 (1개일 때도 배열로 유지)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 6), squeeze=False)
    axes = axes.flatten()  # 2D 배열을 1D로 변환
    
    for model_idx, model in enumerate(MODELS):
        ax = axes[model_idx]
        x = np.arange(len(task_order))
        # 막대 너비를 method 개수에 맞게 동적으로 계산 (전체 너비 0.8을 method 개수로 나눔)
        width = 0.8 / len(METHODS)
        
        for i, method in enumerate(METHODS):
            values = []
            for task in task_order:
                if task in data[model]:
                    values.append(data[model][task].get(method, 0) * 100)
                else:
                    values.append(0.0)
            
            # 막대 위치 계산: 중앙 정렬을 위해 (i - (len(METHODS)-1)/2) 사용
            offset = (i - (len(METHODS) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width, 
                          label=method_labels[i], 
                          color=METHOD_COLORS.get(method, '#808080'),  # 색상이 없으면 회색
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
    # fig.suptitle('BPP Performance Comparison Across Several Models', fontsize=18, fontweight='bold', y=0.98)
    
    # ⭐ 수정 2: 범례 위치를 y=0.88로 내려 서브플롯 타이틀과 겹치지 않게 함
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), 
               ncol=len(METHODS), fontsize=12, frameon=True)  # method 개수에 맞게 동적으로 설정
    
    # ⭐ 수정 3: tight_layout() 대신 subplots_adjust를 사용하여 레이아웃 직접 제어
    # top: 서브플롯 상단 위치를 내려 제목/범례 공간 확보, wspace: 서브플롯 간 좌우 간격 조정
    plt.subplots_adjust(top=0.78, wspace=0.15)
    
    return fig

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, default=0, help='0: 기본 파일명(run 없음), 1 이상: run 번호')
    cli_args = parser.parse_args()
    run = cli_args.run

    print(f"Normal task 데이터 로딩 중... (run={run})")
    
    # 시각화 생성
    fig = create_visualization(run=run)
    
    if fig:
        print("시각화 생성 완료!")
        
        # PNG 파일로 저장
        output_path = f'normal_performance_comparison_{run}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"결과가 {output_path}에 저장되었습니다.")
        
        # 화면에 표시
        plt.show()
    else:
        print("시각화 생성에 실패했습니다.")

if __name__ == "__main__":
    main()
