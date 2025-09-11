import json
import os
import pandas as pd

def calculate_accuracy(file_path):
    result = []
    try:
        with open(file_path, 'r', encoding='utf8') as data_file:
            for line in data_file:
                result.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    accuracies = []
    for i in range(len(result)):
        for data in result[i].get('test_output_infos', []):
            # GLUE 태스크는 correct 필드를 사용
            if 'correct' in data:
                accuracy = 1.0 if data['correct'] else 0.0
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
                                if 'correct' in test_info:
                                    accuracy = 1.0 if test_info['correct'] else 0.0
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
                file_path = os.path.join(root, file)
                
                try:
                    # 파일명에서 모델과 메소드 추출
                    model = file.split("_model-")[1].split("_")[0]
                    method = file.split("__method-")[1].split("_model")[0]
                    
                    # GLUE 서브셋 추출 (glue_cola, glue_sst2 등에서 cola, sst2 추출)
                    task_part = file.split("__method-")[0]
                    if "glue_" in task_part:
                        subset = task_part.split("glue_")[1].split("_")[0]
                    else:
                        subset = "unknown"
                    
                    print(f"\nProcessing file: {file}")
                    print(f"Model: {model}")
                    print(f"Method: {method}")
                    print(f"GLUE Subset: {subset}")

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
                            "GLUE_Subset": subset,
                            "Result Index": idx + 1,
                            "Accuracy": acc
                        })
                        print(f"Accuracy for result {idx + 1}: {acc:.2f}")

                    data.append({
                        "File": file,
                        "Model": model,
                        "Method": method,
                        "GLUE_Subset": subset,
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

def get_root_dir_by_model_and_subset(model_name, subset):
    # GLUE 로그 디렉토리 경로 (각 서브셋별로 폴더가 분리됨)
    base_dir = 'logs'
    
    if model_name == 'gpt-4o':
        return {
            'with_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-4o-2024-08-06_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-4o-2024-08-06_wo_sys_mes')
        }
    elif model_name == 'gpt35-turbo':
        return {
            'with_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-3.5-turbo_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-3.5-turbo_wo_sys_mes')
        }
    elif model_name == 'gpt-4o-mini':
        return {
            'with_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-4o-mini_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-4o-mini_wo_sys_mes')
        }
    elif model_name == 'o1-mini':
        return {
            'with_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'o1-mini_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'o1-mini_wo_sys_mes')
        }
    elif model_name == 'gpt-4.1':
        return {
            'with_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-4.1_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-4.1_wo_sys_mes')
        }
    elif model_name == 'gpt-4.1-mini':
        return {
            'with_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-4.1-mini_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'gpt-4.1-mini_wo_sys_mes')
        }
    elif model_name == 'llama3.1-8b-inst':
        return {
            'with_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'llama3.1-8b-inst_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'llama3.1-8b-inst_wo_sys_mes')
        }
    elif model_name == 'qwen2.5-7b-instruct':
        return {
            'with_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'qwen2.5-7b-instruct_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, f'glue_{subset}', 'qwen2.5-7b-instruct_wo_sys_mes')
        }
    else:
        raise ValueError(f"Model {model_name} is not supported.")

def process_glue_subset_accuracy(root_dir, subset, output_file_name):
    """특정 GLUE 서브셋의 정확도만 계산"""
    data = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                
                try:
                    model = file.split("_model-")[1].split("_")[0]
                    method = file.split("__method-")[1].split("_model")[0]
                    
                    print(f"\nProcessing file: {file}")
                    print(f"Model: {model}")
                    print(f"Method: {method}")
                    print(f"GLUE Subset: {subset}")

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
                            "GLUE_Subset": subset,
                            "Result Index": idx + 1,
                            "Accuracy": acc
                        })

                    data.append({
                        "File": file,
                        "Model": model,
                        "Method": method,
                        "GLUE_Subset": subset,
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

# Set the model you want to use (gpt-4.1, gpt-4.1-mini, gpt-4o, gpt35-turbo, gpt-4o-mini, o1-mini, llama3.1-8b-inst, qwen2.5-7b-instruct)
selected_model = 'gpt-4o'

# GLUE 서브셋 목록
glue_subsets = ['cola', 'sst2', 'mrpc', 'qqp', 'rte', 'qnli']

try:
    # 모든 GLUE 서브셋에 대해 처리
    for subset in glue_subsets:
        print(f"\n{'='*50}")
        print(f"Processing GLUE subset: {subset.upper()}")
        print(f"{'='*50}")
        
        root_dirs = get_root_dir_by_model_and_subset(selected_model, subset)
        
        if os.path.exists(root_dirs['with_sys_mes']):
            process_glue_subset_accuracy(root_dirs['with_sys_mes'], subset, f"glue_{subset}_accuracy_results_{selected_model}_with_sys_mes.xlsx")
        else:
            print(f"with_sys_mes folder does not exist for {selected_model} in {subset}")

        if os.path.exists(root_dirs['wo_sys_mes']):
            process_glue_subset_accuracy(root_dirs['wo_sys_mes'], subset, f"glue_{subset}_accuracy_results_{selected_model}_wo_sys_mes.xlsx")
        else:
            print(f"wo_sys_mes folder does not exist for {selected_model} in {subset}")


except Exception as e:
    print(f"Error during processing: {e}")
