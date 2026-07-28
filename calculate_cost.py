#!/usr/bin/env python3
import json
import sys
import os
from collections import defaultdict
from pathlib import Path

# 모델별 가격 정보 (100만 토큰당)
# Output: 출력 토큰 가격, Input: 입력 토큰 가격
MODEL_PRICES = {
    "gpt-4o-2024-08-06": {"output": 10.00, "input": 2.50},
    "gpt-3.5-turbo": {"output": 1.500, "input": 0.500},
    "gpt-4o-mini": {"output": 0.600, "input": 0.150},
    "o1-mini": {"output": 4.40, "input": 1.10},
    "gpt-4.1": {"output": 8.00, "input": 2.00},
    "gpt-4.1-mini": {"output": 1.60, "input": 0.40},
    "gpt-5.2": {"output": 14.00, "input": 1.75},
    "gpt-5.4": {"output": 15.00, "input": 2.50},
    "gpt-5.4-nano": {"output": 1.25, "input": 0.20},
    # Open models은 비용이 0
    "llama3.1-8b-inst": {"output": 0.0, "input": 0.0},
    "meta-llama-Llama-3.1-8B-Instruct": {"output": 0.0, "input": 0.0},
    "qwen2.5-7b-instruct": {"output": 0.0, "input": 0.0},
    "Qwen-Qwen2.5-7B-Instruct": {"output": 0.0, "input": 0.0},
}

def get_model_price(model_name):
    """모델 이름으로 가격 정보를 반환합니다."""
    # 여러 형태의 모델 이름 지원
    if model_name in MODEL_PRICES:
        return MODEL_PRICES[model_name]
    
    # 부분 매칭 시도
    for key in MODEL_PRICES:
        if key.lower() in model_name.lower() or model_name.lower() in key.lower():
            return MODEL_PRICES[key]
    
    # 기본값 (알 수 없는 모델은 0)
    return {"output": 0.0, "input": 0.0}

def calculate_cost_from_jsonl(file_path):
    """JSONL 파일에서 토큰 사용량과 비용을 계산합니다."""
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # raw_response에서 usage 정보 추출
                    if 'raw_response' in data and isinstance(data['raw_response'], list):
                        for response in data['raw_response']:
                            if 'usage' in response:
                                usage = response['usage']
                                total_prompt_tokens += usage.get('prompt_tokens', 0)
                                total_completion_tokens += usage.get('completion_tokens', 0)
                            
                except json.JSONDecodeError as e:
                    print(f"Warning: {file_path} Line {line_num}에서 JSON 파싱 오류: {e}", file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다: {file_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: {file_path} 처리 중 오류: {e}", file=sys.stderr)
        return None
    
    return {
        'prompt_tokens': total_prompt_tokens,
        'completion_tokens': total_completion_tokens,
        'total_tokens': total_prompt_tokens + total_completion_tokens
    }

def extract_info_from_filepath(file_path):
    """파일 경로와 파일명에서 모델, 메소드, task 정보를 추출합니다."""
    file_path = str(file_path)
    file_name = os.path.basename(file_path)
    
    # task 추출 (logs/task_name/... 형식)
    parts = file_path.split(os.sep)
    task = "unknown"
    if len(parts) >= 2 and "logs" in parts:
        logs_idx = parts.index("logs")
        if logs_idx + 1 < len(parts):
            task = parts[logs_idx + 1]
    
    # n 값 추출 (파일명에서 _n_5 또는 _n_10 등의 패턴)
    n_value = None
    import re
    n_match = re.search(r'_n_(\d+)', file_name)
    if n_match:
        n_value = n_match.group(1)
        # trivia_creative_writing의 경우 n 값을 task에 추가
        if task == "trivia_creative_writing":
            task = f"{task} (n={n_value})"
    
    # 모델 추출 (파일명에서 _model-{model}_ 부분)
    model = "unknown"
    if "_model-" in file_name:
        try:
            model = file_name.split("_model-")[1].split("_")[0]
        except:
            pass
    
    # 메소드 추출 (파일명에서 __method-{method}_ 부분)
    method = "unknown"
    if "__method-" in file_name:
        try:
            method = file_name.split("__method-")[1].split("_model")[0]
        except:
            pass
    
    return task, model, method

def process_all_logs(logs_dir="logs"):
    """logs 폴더의 모든 JSONL 파일을 처리하고 비용을 집계합니다."""
    
    # 집계용 딕셔너리: (task, model, method) -> 토큰 정보
    stats = defaultdict(lambda: {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'cost': 0.0,
        'file_count': 0
    })
    
    # 모든 JSONL 파일 찾기
    jsonl_files = list(Path(logs_dir).rglob("*.jsonl"))
    
    if not jsonl_files:
        print(f"Error: {logs_dir} 폴더에서 JSONL 파일을 찾을 수 없습니다.")
        return None
    
    print(f"총 {len(jsonl_files)}개의 JSONL 파일을 찾았습니다.\n")
    print("파일 처리 중...")
    
    for file_path in jsonl_files:
        task, model, method = extract_info_from_filepath(file_path)
        
        # 토큰 사용량 계산
        token_info = calculate_cost_from_jsonl(file_path)
        if token_info is None:
            continue
        
        # 가격 정보 가져오기
        prices = get_model_price(model)
        
        # 비용 계산
        cost = (token_info['completion_tokens'] / 1000000 * prices['output'] + 
                token_info['prompt_tokens'] / 1000000 * prices['input'])
        
        # 집계
        key = (task, model, method)
        stats[key]['prompt_tokens'] += token_info['prompt_tokens']
        stats[key]['completion_tokens'] += token_info['completion_tokens']
        stats[key]['total_tokens'] += token_info['total_tokens']
        stats[key]['cost'] += cost
        stats[key]['file_count'] += 1
    
    return stats

def print_summary(stats):
    """집계된 통계를 표 형식으로 출력합니다."""
    
    if not stats:
        print("집계할 데이터가 없습니다.")
        return
    
    # 데이터를 리스트로 변환
    data = []
    for (task, model, method), info in stats.items():
        data.append({
            'task': task,
            'model': model,
            'method': method,
            'prompt_tokens': info['prompt_tokens'],
            'completion_tokens': info['completion_tokens'],
            'total_tokens': info['total_tokens'],
            'cost': info['cost'],
            'file_count': info['file_count']
        })
    
    # task, model, method 순으로 정렬
    data.sort(key=lambda x: (x['task'], x['model'], x['method']))
    
    # 표 출력
    print("\n" + "=" * 120)
    print("모델별/방법론별/Task별 API 비용 집계")
    print("=" * 120)
    print(f"{'Task':<30} {'Model':<25} {'Method':<20} {'Input Tokens':>15} {'Output Tokens':>15} {'Total Tokens':>15} {'Cost ($)':>12} {'Files':>8}")
    print("-" * 120)
    
    total_cost = 0.0
    for row in data:
        print(f"{row['task']:<30} {row['model']:<25} {row['method']:<20} "
              f"{row['prompt_tokens']:>15,} {row['completion_tokens']:>15,} "
              f"{row['total_tokens']:>15,} ${row['cost']:>10.4f} {row['file_count']:>8}")
        total_cost += row['cost']
    
    print("-" * 120)
    print(f"{'TOTAL':<76} ${total_cost:>10.4f}")
    print("=" * 120)
    
    # Task별 집계
    task_summary = defaultdict(lambda: {'cost': 0.0, 'files': 0})
    for row in data:
        task_summary[row['task']]['cost'] += row['cost']
        task_summary[row['task']]['files'] += row['file_count']
    
    print("\n" + "=" * 60)
    print("Task별 총 비용")
    print("=" * 60)
    print(f"{'Task':<30} {'Cost ($)':>15} {'Files':>8}")
    print("-" * 60)
    for task, info in sorted(task_summary.items()):
        print(f"{task:<30} ${info['cost']:>14.4f} {info['files']:>8}")
    print("=" * 60)
    
    # Model별 집계
    model_summary = defaultdict(lambda: {'cost': 0.0, 'files': 0})
    for row in data:
        model_summary[row['model']]['cost'] += row['cost']
        model_summary[row['model']]['files'] += row['file_count']
    
    print("\n" + "=" * 60)
    print("Model별 총 비용")
    print("=" * 60)
    print(f"{'Model':<30} {'Cost ($)':>15} {'Files':>8}")
    print("-" * 60)
    for model, info in sorted(model_summary.items()):
        print(f"{model:<30} ${info['cost']:>14.4f} {info['files']:>8}")
    print("=" * 60)
    
    # Method별 집계
    method_summary = defaultdict(lambda: {'cost': 0.0, 'files': 0})
    for row in data:
        method_summary[row['method']]['cost'] += row['cost']
        method_summary[row['method']]['files'] += row['file_count']
    
    print("\n" + "=" * 60)
    print("Method별 총 비용")
    print("=" * 60)
    print(f"{'Method':<30} {'Cost ($)':>15} {'Files':>8}")
    print("-" * 60)
    for method, info in sorted(method_summary.items()):
        print(f"{method:<30} ${info['cost']:>14.4f} {info['files']:>8}")
    print("=" * 60)

def calculate_cost_from_jsonl_single(file_path):
    """단일 JSONL 파일에서 토큰 사용량과 비용을 계산합니다 (기존 기능 유지)."""
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    
    # 파일명에서 모델 추출
    file_name = os.path.basename(file_path)
    model = "gpt-4o-2024-08-06"  # 기본값
    if "_model-" in file_name:
        try:
            model = file_name.split("_model-")[1].split("_")[0]
        except:
            pass
    
    prices = get_model_price(model)
    output_price_per_1M = prices['output']
    input_price_per_1M = prices['input']
    
    token_info = calculate_cost_from_jsonl(file_path)
    if token_info is None:
        return
    
    total_prompt_tokens = token_info['prompt_tokens']
    total_completion_tokens = token_info['completion_tokens']
    total_tokens = token_info['total_tokens']
    
    # 비용 계산
    cost = (total_completion_tokens / 1000000 * output_price_per_1M + 
            total_prompt_tokens / 1000000 * input_price_per_1M)
    
    # 결과 출력
    print("=" * 60)
    print(f"API 사용량 및 비용 계산 결과 ({model})")
    print("=" * 60)
    print(f"\n토큰 사용량:")
    print(f"  Input tokens (prompt_tokens):  {total_prompt_tokens:>12,}")
    print(f"  Output tokens (completion_tokens): {total_completion_tokens:>12,}")
    print(f"  Total tokens:                  {total_tokens:>12,}")
    print(f"\n비용 계산 ({model} 가격):")
    print(f"  Input cost:  ${total_prompt_tokens / 1000000 * input_price_per_1M:>10.4f}")
    print(f"  Output cost: ${total_completion_tokens / 1000000 * output_price_per_1M:>10.4f}")
    print(f"  Total cost:  ${cost:>10.4f}")
    print("=" * 60)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # 인자가 없으면 logs 폴더의 모든 파일 처리
        stats = process_all_logs()
        if stats:
            print_summary(stats)
    elif len(sys.argv) == 2:
        file_path = sys.argv[1]
        if os.path.isdir(file_path):
            # 디렉토리인 경우
            stats = process_all_logs(file_path)
            if stats:
                print_summary(stats)
        elif os.path.isfile(file_path):
            # 단일 파일인 경우
            calculate_cost_from_jsonl_single(file_path)
        else:
            print(f"Error: 경로를 찾을 수 없습니다: {file_path}")
            sys.exit(1)
    else:
        print("사용법:")
        print("  python calculate_cost.py              # logs 폴더의 모든 파일 처리")
        print("  python calculate_cost.py <file_path>  # 단일 파일 또는 디렉토리 처리")
        sys.exit(1)
