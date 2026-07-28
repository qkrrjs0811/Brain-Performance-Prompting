import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from configs import claude_configs, open_model_configs
from log_path_utils import (
    insert_sampling_before_sys_mes_folder,
    model_log_folder_stem,
    open_sampling_params_for_accuracy,
)

SUPPORTED_MODELS = [
    "gpt-4o",
    "gpt35-turbo",
    "gpt-4o-mini",
    "o1-mini",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.4-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "llama3.1-8b-inst",
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

print()
print('==================================')

def calculate_accuracy(file_path):
    result = []
    with open(file_path, 'r', encoding='utf8') as data_file:
        for line in data_file:
            result.append(json.loads(line))
    
    accuracies = []
    for i in range(len(result)):
        for data in result[i]['test_output_infos']:
            if data['correct'] == False:
                accuracy = 0
            else:
                accuracy = 1
            accuracies.append(accuracy)
    
    return accuracies


def calculate_accuracy_self_refine(file_path):
    accuracies = []
    try:
        with open(file_path, 'r', encoding='utf-8') as data_file:
            line_number = 0
            for line in data_file:
                line_number += 1
                try:
                    data = json.loads(line.strip())
                    if 'answer_1' in data:
                        value = data['answer_1']
                        if 'test_output_infos' in value:
                            for test_info in value['test_output_infos']:
                                correct = test_info.get('correct', 0)
                                
                                if correct == False:
                                    accuracy = 0
                                else:
                                    accuracy = 1
                                accuracies.append(accuracy)
                except json.JSONDecodeError:
                    print(f"JSONDecodeError at line {line_number}: Line could not be parsed. Skipping this line.")
                except Exception as e:
                    print(f"Error at line {line_number}: {e}")
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")

    return accuracies


def process_all_files(root_dir, output_file_name):
    data = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jsonl'):
                # Batch 실행 관련 입력/메타 파일은 request payload라 test_output_infos가 없어 정확도 계산이 0으로 찍힘.
                if (
                    "batch_input" in file
                    or "batch_meta" in file
                    or "batch_phase" in file
                    or "batch_anthropic_requests" in file
                ):
                    continue
                file_path = os.path.join(root, file)
                
                try:
                    model = file.split("_model-")[1].split("_")[0]
                    method = file.split("__method-")[1].split("_model")[0]
                    
                    print(f"\nProcessing file: {file}")
                    print(f"Model: {model}")
                    print(f"Method: {method}")

                    if method == 'self_refine':
                        accuracies = calculate_accuracy_self_refine(file_path)
                    else:
                        accuracies = calculate_accuracy(file_path)
                    
                    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

                    for idx, acc in enumerate(accuracies):
                        data.append({
                            "File": file,
                            "Model": model,
                            "Method": method,
                            "Result Index": idx + 1,
                            "Accuracy": acc
                        })
                        print(f"Accuracy for result {idx + 1}: {acc:.2f}")

                    data.append({
                        "File": file,
                        "Model": model,
                        "Method": method,
                        "Result Index": "Mean",
                        "Accuracy": mean_accuracy
                    })
                    print(f"Mean Accuracy: {mean_accuracy:.2f}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    df = pd.DataFrame(data)

    try:
        output_file = os.path.join(root_dir, output_file_name)
        df.to_excel(output_file, index=False)
        print(f"\nExcel file saved to {output_file}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")




def get_root_dir_by_model(model_name):
    # Define the base directory for logs (modify this path according to your environment)
    base_dir = 'logs/logic_grid_puzzle'
    
    if model_name == 'gpt-4o':
        return {
            'with_sys_mes': os.path.join(base_dir, 'gpt-4o-2024-08-06_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'gpt-4o-2024-08-06_wo_sys_mes')
        }
    elif model_name == 'gpt35-turbo':
        return {
            'with_sys_mes': os.path.join(base_dir, 'gpt-3.5-turbo_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'gpt-3.5-turbo_wo_sys_mes')
        }
    elif model_name == 'gpt-4o-mini':
        return {
            'with_sys_mes': os.path.join(base_dir, 'gpt-4o-mini_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'gpt-4o-mini_wo_sys_mes')
        }
    elif model_name == 'o1-mini':
        return {
            'with_sys_mes': os.path.join(base_dir, 'o1-mini_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'o1-mini_wo_sys_mes')
        }
    elif model_name == 'gpt-5.2':
        return {
            'with_sys_mes': os.path.join(base_dir, 'gpt-5.2_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'gpt-5.2_wo_sys_mes')
        }
    elif model_name == 'gpt-5.4':
        return {
            'with_sys_mes': os.path.join(base_dir, 'gpt-5.4_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'gpt-5.4_wo_sys_mes')
        }
    elif model_name == 'gpt-5.4-nano':
        return {
            'with_sys_mes': os.path.join(base_dir, 'gpt-5.4-nano_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'gpt-5.4-nano_wo_sys_mes')
        }
    elif model_name == 'gpt-4.1':
        return {
            'with_sys_mes': os.path.join(base_dir, 'gpt-4.1_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'gpt-4.1_wo_sys_mes')
        }
    elif model_name == 'gpt-4.1-mini':
        return {
            'with_sys_mes': os.path.join(base_dir, 'gpt-4.1-mini_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'gpt-4.1-mini_wo_sys_mes')
        }
    elif model_name == 'llama3.1-8b-inst':
        return {
            'with_sys_mes': os.path.join(base_dir, 'meta-llama-Llama-3.1-8B-Instruct_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'meta-llama-Llama-3.1-8B-Instruct_wo_sys_mes')
        }
    elif model_name == 'qwen2.5-7b-instruct':
        return {
            'with_sys_mes': os.path.join(base_dir, 'Qwen-Qwen2.5-7B-Instruct_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'Qwen-Qwen2.5-7B-Instruct_wo_sys_mes')
        }
    elif model_name == 'qwen2.5-14b-instruct':
        return {
            'with_sys_mes': os.path.join(base_dir, 'Qwen-Qwen2.5-14B-Instruct_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'Qwen-Qwen2.5-14B-Instruct_wo_sys_mes')
        }
    elif model_name in claude_configs:
        stem = model_log_folder_stem(claude_configs[model_name])
        return {
            "with_sys_mes": os.path.join(base_dir, f"{stem}_w_sys_mes"),
            "wo_sys_mes": os.path.join(base_dir, f"{stem}_wo_sys_mes"),
        }
    else:
        raise ValueError(f"Model {model_name} is not supported.")

def _apply_reasoning_suffix_if_needed(model_name, folder_name, reasoning_effort):
    if model_name.startswith("gpt-5") and reasoning_effort:
        suffix = "_w_sys_mes" if folder_name.endswith("_w_sys_mes") else "_wo_sys_mes"
        base = folder_name[: -len(suffix)]
        return f"{base}_re-{reasoning_effort}{suffix}"
    return folder_name


def _resolve_existing_dir(path_with_effort, fallback_path):
    if os.path.exists(path_with_effort):
        return path_with_effort
    return fallback_path


def parse_args():
    p = argparse.ArgumentParser(description="Logic grid puzzle 로그에서 정확도 집계")
    p.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        default="gpt-5.4",
        help="get_root_dir_by_model()에 대응하는 모델 키",
    )
    p.add_argument(
        "--reasoning_effort",
        type=str,
        default="none",
        help="gpt-5 계열 폴더명 suffix용 (예: none, low, medium, high, xhigh)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="open_model 전용: 로그 폴더 temp- (미지정 시 open_model_configs 기본값)",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="open_model 전용: 로그 폴더 topp- (미지정 시 open_model_configs 기본값)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    selected_model = args.model
    reasoning_effort = args.reasoning_effort
    t, tp = open_sampling_params_for_accuracy(
        selected_model, args.temperature, args.top_p, open_model_configs
    )
    root_dirs = get_root_dir_by_model(selected_model)
    with_dir_effort = _apply_reasoning_suffix_if_needed(
        selected_model, root_dirs["with_sys_mes"], reasoning_effort
    )
    wo_dir_effort = _apply_reasoning_suffix_if_needed(
        selected_model, root_dirs["wo_sys_mes"], reasoning_effort
    )
    if selected_model in open_model_configs:
        with_dir_effort = insert_sampling_before_sys_mes_folder(with_dir_effort, t, tp)
        wo_dir_effort = insert_sampling_before_sys_mes_folder(wo_dir_effort, t, tp)
    with_dir = _resolve_existing_dir(with_dir_effort, root_dirs["with_sys_mes"])
    wo_dir = _resolve_existing_dir(wo_dir_effort, root_dirs["wo_sys_mes"])

    if os.path.exists(with_dir):
        process_all_files(with_dir, f"accuracy_results_{selected_model}_with_sys_mes.xlsx")
    else:
        print(f"with_sys_mes folder does not exist for {selected_model}")

    if os.path.exists(wo_dir):
        process_all_files(wo_dir, f"accuracy_results_{selected_model}_wo_sys_mes.xlsx")
    else:
        print(f"wo_sys_mes folder does not exist for {selected_model}")


if __name__ == "__main__":
    main()
