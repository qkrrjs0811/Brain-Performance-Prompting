import json
import os
import pandas as pd

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
            accuracy = data['matched_count'] / data['target_count']
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
                    if 'guesser_logs' in data and 'answer_1' in data['guesser_logs']:
                        answer_1 = data['guesser_logs']['answer_1']
                        if 'test_output_infos' in answer_1:
                            for test_info in answer_1['test_output_infos']:
                                matched_count = test_info.get('matched_count', 0)
                                target_count = test_info.get('target_count', 0)
                                if target_count > 0:
                                    accuracy = matched_count / target_count
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
    base_dir = 'logs/codenames_collaborative'
    
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
            'with_sys_mes': os.path.join(base_dir, 'qwen2.5-7b-instruct_w_sys_mes'),
            'wo_sys_mes': os.path.join(base_dir, 'qwen2.5-7b-instruct_wo_sys_mes')
        }
    else:
        raise ValueError(f"Model {model_name} is not supported.")


# Set the model you want to use (gpt-4.1, gpt-4.1-mini, gpt-4o, gpt35-turbo, gpt-4o-mini, o1-mini, llama3.1-8b-inst, qwen2.5-7b-instruct)
selected_model = 'llama3.1-8b-inst' 


root_dirs = get_root_dir_by_model(selected_model)

if os.path.exists(root_dirs['with_sys_mes']):
    process_all_files(root_dirs['with_sys_mes'], f"accuracy_results_{selected_model}_with_sys_mes.xlsx")
else:
    print(f"with_sys_mes folder does not exist for {selected_model}")

if os.path.exists(root_dirs['wo_sys_mes']):
    process_all_files(root_dirs['wo_sys_mes'], f"accuracy_results_{selected_model}_wo_sys_mes.xlsx")
else:
    print(f"wo_sys_mes folder does not exist for {selected_model}")
