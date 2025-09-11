import os
import json
import argparse
from models import OpenAIWrapper, OpenModelWrapper
from tasks import get_task
import time
from configs import gpt_configs, open_model_configs, default_gpt_config, default_open_model_config


SLEEP_RATE = 10 # sleep between calls


import os

# 로그 파일을 저장할 경로를 설정하는 함수
def setup_log_file(task_data_file, method, model_config, start_idx, end_idx, additional_output_note, system_message, output_dir):
    # 모델 이름을 기반으로 폴더 생성
    model_name_for_output = model_config['model'].replace("/", "-")
    
    # system_message 유무에 따라 상위 폴더명 결정
    sys_mes_folder = f"{model_name_for_output}_{'w_sys_mes' if system_message else 'wo_sys_mes'}"
    
    # 로그 디렉토리 경로 설정
    log_dir = os.path.normpath(os.path.join(output_dir, sys_mes_folder))
    os.makedirs(log_dir, exist_ok=True)
    
    # temperature와 top_p를 체크하여 없는 경우 생략
    temp_str = f"_temp-{model_config['temperature']}" if 'temperature' in model_config else ""
    top_p_str = f"_topp-{model_config['top_p']}" if 'top_p' in model_config else ""

    # 로그 파일 이름 설정
    if system_message == "":
        log_file_name = f"{task_data_file}__method-{method}_model-{model_name_for_output}_temp-{temp_str}_topp-{top_p_str}_start{start_idx}-end{end_idx}{additional_output_note}__wo_sys.jsonl"
        log_file = os.path.normpath(os.path.join(log_dir, log_file_name))
    else:
        log_file_name = f"{task_data_file}__method-{method}_model-{model_name_for_output}_temp-{temp_str}_topp-{top_p_str}_start{start_idx}-end{end_idx}{additional_output_note}__w_sys.jsonl"
        log_file = os.path.normpath(os.path.join(log_dir, log_file_name))
        
    # 경로 디버깅 출력
    print(f"Log file path: {log_file}")

    return log_file



def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def _post_process_raw_response(task, raw_output_batch, method, **kwargs):
    unwrapped_output_batch = []
    if_success_batch = []
    for output in raw_output_batch:
        unwrapped_output, if_success_flag = task.prompt_unwrap(output, method, **kwargs)
        unwrapped_output_batch.append(unwrapped_output)
        if_success_batch.append(if_success_flag)
    return unwrapped_output_batch, if_success_batch


### default task runners ###

def _get_response_default(model, task, i, method, num_generation, prompt, test_output=True, **kwargs):
    # 시작 시간 기록
    start_time = time.time()
    
    raw_output_batch, raw_response_batch = model.run(prompt=prompt, n=num_generation)
    if raw_output_batch == [] or raw_response_batch == []: # handle exception
        return {}    
    # get parsed response, and the success flags (whether or not the parsing is success) (standard prompt always success)
    unwrapped_output_batch, if_success_batch = _post_process_raw_response(task, raw_output_batch, method, **kwargs)
    # compute automatic metric (different for each task), e.g., if the output contains all the answers
    if test_output:
        test_output_infos = [task.test_output(i, output) for output in unwrapped_output_batch]
    else:
        test_output_infos = []
    
    # 종료 시간 기록 및 실행 시간 계산
    end_time = time.time()
    execution_time = end_time - start_time
    
    # log output
    log_output = {
        "idx": i,
        "raw_response": raw_response_batch,
        "unwrapped_output": unwrapped_output_batch,
        "parsing_success_flag": if_success_batch,
        "test_output_infos": test_output_infos,
        "usage_info": model.compute_gpt_usage(),
        "execution_time": execution_time
    }
    return log_output

def _run_task_default(model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, test_output=True):
    # get prompt
    prompt = task.get_input_prompt(i, method=method)
    # get response and parsed output 
    return _get_response_default(model, task, i, method, num_generation, prompt, test_output=test_output)

def _run_task_codenames(model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, test_output=True):
    # 전체 codenames 시작 시간
    total_start_time = time.time()
    
    # get spymaster hint word
    spymaster_prompt = task.get_input_prompt(i, method=method, role='spymaster')
    raw_spymaster_output, raw_response_spymaster = model.run(prompt=spymaster_prompt, n=1)
    if raw_spymaster_output == [] or raw_response_spymaster == []: # handle exception
        return {}
    spymaster_output, if_success_batch_spymaster = _post_process_raw_response(task, raw_spymaster_output, method)
    hint_word = spymaster_output[0].replace(".", "").strip()
    # print(f"\tidx: {i} | done spymaster, hint word: {hint_word}")
    # sleep before calling guesser
    time.sleep(sleep_rate)
    # get guesser result
    guesser_prompt = task.get_input_prompt(i, method=method, role='guesser', hint_word=hint_word)
    raw_guesser_output, raw_response_batch_guesser = model.run(prompt=guesser_prompt, n=num_generation)
    if raw_guesser_output == [] or raw_response_batch_guesser == []: # handle exception
        return {}
    guesser_output_batch, if_success_batch_guesser = _post_process_raw_response(task, raw_guesser_output, method)
    # print(f"\tidx: {i} | done guesser, result: {guesser_output_batch}")
    # compute automatic metric (different for each task), e.g., if the output contains all the answers
    if test_output:
        test_output_infos = [task.test_output(i, output) for output in guesser_output_batch]
    else:
        test_output_infos = []
    
    # 전체 codenames 종료 시간 및 총 실행 시간 계산
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # log output
    log_output = {
        "idx": i,
        "raw_response_spymaster": raw_response_spymaster,
        "raw_response_guesser": raw_response_batch_guesser,
        "spymaster_output": spymaster_output,
        "guesser_output": guesser_output_batch,
        "hint_word": hint_word,
        "parsing_success_flag_spymaster": if_success_batch_spymaster,
        "parsing_success_flag_guesser": if_success_batch_guesser,
        "test_output_infos": test_output_infos,
        "total_execution_time": total_execution_time,
        "total_usage_info": model.compute_gpt_usage()
    }
    return log_output

##############################

### self_refine task runners ###

def _run_self_refine_default(model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, num_refine=1, **kwargs):
    print("\tidx:", i, "start self refine...")
    log_outputs = {}
    
    # 전체 self_refine 시작 시간
    total_start_time = time.time()
    
    ## get initial response
    init_prompt = task.get_input_prompt(i, method=method, phase="init", **kwargs)
    init_output = _get_response_default(model, task, i, method, num_generation=1, prompt=init_prompt, test_output=True, phase="init")
    if init_output == {}:
        return {}
    log_outputs["answer_0"] = init_output

    time.sleep(sleep_rate)
    context_prompt = init_output['raw_response'][0]['prompt'] + "\n" + init_output["raw_response"][0]['choices'][0]['message']['content'] # Q + A0
    for j in range(num_refine):
        print("\t\tstep:", j)
        # get feedback
        feedback_prompt = task.get_input_prompt(i, method=method, phase="feedback", question_answer=context_prompt, **kwargs)
        feedback_output = _get_response_default(model, task, i, method, num_generation=1, prompt=feedback_prompt, test_output=False, phase="feedback")
        if feedback_output == {}:
            return log_outputs
        log_outputs[f"feedback_{j}"] = feedback_output
        time.sleep(sleep_rate)

        # get refined response
        refine_prompt = task.get_input_prompt(i, method=method, phase="refine", question_answer=context_prompt, feedback=feedback_output["unwrapped_output"][0], **kwargs) # Q + A0 + F
        refine_output = _get_response_default(model, task, i, method, num_generation=1, prompt=refine_prompt, test_output=True, phase="refine")
        if refine_output == {}:
            return log_outputs
        log_outputs[f"answer_{j+1}"] = refine_output
        time.sleep(sleep_rate)

        # update context
        context_prompt = refine_prompt + refine_output["raw_response"][0]['choices'][0]['message']['content'] # Q + A0 + F + A1

    # 전체 self_refine 종료 시간 및 총 실행 시간 계산
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # 총 실행 시간과 사용량 정보를 메타데이터로 추가
    log_outputs["total_execution_time"] = total_execution_time
    log_outputs["total_usage_info"] = model.compute_gpt_usage()

    return log_outputs

def _run_self_refine_codenames(model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, num_refine=1, test_output=True):
    # 전체 self_refine_codenames 시작 시간
    total_start_time = time.time()
    
    # get spymaster hint word
    spy_master_log_outputs = _run_self_refine_default(model, task, i, method, num_generation, sleep_rate, num_refine, role='spymaster')
    if f"answer_{num_refine}" not in spy_master_log_outputs:
        return {}
    hint_word = spy_master_log_outputs[f"answer_{num_refine}"]["unwrapped_output"][0].replace(".", "").strip()
    print(f"\tidx: {i} | num_refine: {num_refine} | done spymaster, hint word: {hint_word}")
    # sleep before calling guesser
    time.sleep(sleep_rate)
    # get guesser result
    guesser_log_outputs = _run_self_refine_default(model, task, i, method, num_generation, sleep_rate, num_refine, role='guesser', hint_word=hint_word)
    if f"answer_{num_refine}" not in guesser_log_outputs:
        return {}
    guesser_output = guesser_log_outputs[f"answer_{num_refine}"]["unwrapped_output"][0]
    # compute automatic metric (different for each task), e.g., if the output contains all the answers
    if test_output:
        test_output_infos = [task.test_output(i, guesser_output)]
    else:
        test_output_infos = []
    
    # 전체 self_refine_codenames 종료 시간 및 총 실행 시간 계산
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # log output
    log_output = {
        "idx": i,
        "spymaster_logs": spy_master_log_outputs,
        "guesser_logs": guesser_log_outputs,
        "hint_word": hint_word,
        "parsing_success_flag_spymaster": spy_master_log_outputs[f"answer_{num_refine}"]["parsing_success_flag"],
        "parsing_success_flag_guesser": guesser_log_outputs[f"answer_{num_refine}"]["parsing_success_flag"],
        "test_output_infos": test_output_infos,
        "total_execution_time": total_execution_time,
        "total_usage_info": model.compute_gpt_usage()
    }
    return log_output
##############################



def _run_task(task_name, model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, **kwargs):
    if task_name in ['trivia_creative_writing', 'logic_grid_puzzle']:
        if method == "self_refine":
            log_output = _run_self_refine_default(model, task, i, method, num_generation, sleep_rate, num_refine = kwargs['num_refine'])
        else:
            log_output = _run_task_default(model, task, i, method, num_generation, sleep_rate)
    elif task_name == 'codenames_collaborative':
        if method == "self_refine":
            log_output = _run_self_refine_codenames(model, task, i, method, num_generation, sleep_rate, num_refine = kwargs['num_refine'])
        else:
            log_output = _run_task_codenames(model, task, i, method, num_generation, sleep_rate)
    elif task_name.startswith('glue_'):
        # GLUE 태스크는 standard 방식으로 처리
        log_output = _run_task_default(model, task, i, method, num_generation, sleep_rate)
    else:
        raise NotImplementedError(f"task {task_name} not implemented; please choose from ['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative', 'glue_*']")

    # log everything else that is related
    if "open_model_config" in args:
        args["open_model_config"]["torch_dtype"] = str(args["open_model_config"]["torch_dtype"])
    log_output.update(args)
    log_output.update({"task_data":task.get_input(i)})
    return log_output

def run(args):
    # get configs
    model_type = args['model_type']
    task_name = args['task']
    method = args['method']
    start_idx, end_idx = args['task_start_index'], args['task_end_index']
    task_data_file = args['task_data_file']
    num_generation = args['num_generation']
    
    output_dir = args['output_dir']
    if output_dir == "":
        output_dir = f"logs/{task_name}"

    additional_output_note = args['additional_output_note']
    system_message = args['system_message']
    print(f"setting default system message: {system_message}")
    
    # setup model and output log file
    if model_type == 'gpt':
        model_config = args['gpt_config']
        model = OpenAIWrapper(config=model_config, system_message=system_message)
        # setup log file
        log_file = setup_log_file(task_data_file, method, model_config, start_idx, end_idx, additional_output_note, system_message, output_dir)
        sleep_rate = SLEEP_RATE

    elif model_type == 'open_model':
        model_config = args['open_model_config']
        # 로컬 모델 경로가 설정되어 있으면 사용
        local_model_path = args.get('local_model_path', None)
        if local_model_path:
            model_config['local_model_path'] = local_model_path
        model = OpenModelWrapper(config=model_config, local_model_path=local_model_path)
        # setup log file
        log_file = setup_log_file(task_data_file, method, model_config, start_idx, end_idx, additional_output_note, system_message, output_dir)
        sleep_rate = 0

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # setup task
    task = get_task(task_name, file=task_data_file)
    
    all_logs = []
    print("start running ... log file:", log_file)
    print("sleep rate:", sleep_rate)

    print()
    start = max(start_idx, 0)
    end = min(end_idx, len(task))
    total_instances = end - start
    print("total num of instances:", total_instances)
    print("method:", method)
    
    # 타이머 시작
    start_time = time.time()
    
    for i in range(start, end):
        log_output = _run_task(task_name, model, task, i, method, num_generation, sleep_rate, num_refine = args['num_refine'])
        all_logs.append(log_output)
        
        # 현재 진행률 계산
        progress = ((i - start + 1) / total_instances) * 100
        elapsed_time = time.time() - start_time  # 경과 시간
        estimated_total_time = elapsed_time / (i - start + 1) * total_instances  # 총 예상 시간
        remaining_time = estimated_total_time - elapsed_time  # 남은 시간
        
        # 정확도 계산 및 출력
        if 'test_output_infos' in log_output:
            # trivia_creative_writing
            if task_name == 'trivia_creative_writing':
                correct_count = log_output['test_output_infos'][0]['correct_count']
                question_count = log_output['test_output_infos'][0]['question_count']
                accuracy = correct_count / question_count * 100 if question_count > 0 else 0
                print(f"\tidx: {i} | done | Accuracy: {accuracy:.2f}%")
            
            # codenames_collaborative
            elif task_name == 'codenames_collaborative':
                matched_count = log_output['test_output_infos'][0]['matched_count']
                target_count = log_output['test_output_infos'][0]['target_count']
                accuracy = matched_count / target_count * 100 if target_count > 0 else 0
                print(f"\tidx: {i} | done | Accuracy: {accuracy:.2f}%")
            
            # logic_grid_puzzle
            elif task_name == 'logic_grid_puzzle':
                correct = log_output['test_output_infos'][0]['correct']
                accuracy = 100 if correct else 0
                print(f"\tidx: {i} | done | Accuracy: {accuracy}% | Correct: {correct}")
            
            # GLUE tasks
            elif task_name.startswith('glue_'):
                correct = log_output['test_output_infos'][0]['correct']
                true_label = log_output['test_output_infos'][0].get('true_label', 'N/A')
                predicted_label = log_output['test_output_infos'][0].get('predicted_label', 'N/A')
                accuracy = 100 if correct else 0
                subset = task_name.replace('glue_', '').upper()
                print(f"\tidx: {i} | done | Accuracy: {accuracy}% | Correct: {correct} | True: {true_label}, Predicted: {predicted_label} | Task: {subset}")
        
        print(f"\tidx: {i}, done | Progress: {progress:.2f}% | Elapsed time: {elapsed_time:.2f}s | Estimated total time: {estimated_total_time:.2f}s | Estimated remaining time: {remaining_time:.2f}s | Total usage so far: {model.compute_gpt_usage()}")
        # output log at each iteration
        output_log_jsonl(log_file, all_logs)
        # sleep
        time.sleep(sleep_rate)


def parse_args():
    model_choices = list(gpt_configs.keys()) + list(open_model_configs.keys())
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=model_choices, required=True) # gpt-4o, gpt35-turbo, meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct
    args.add_argument('--output_dir', type=str, required=False, default="")
    args.add_argument('--model_type', type=str, choices=['gpt','open_model'], default='gpt')
    args.add_argument('--method', type=str, choices=['standard','cot','spp', 'self_refine', 'bpp1', 'bpp2', 'bpp3', 'bpp', 'bpp_w_r_demo', 'bpp_w_k_demo', 'bpp_two_k_demo', 'bpp_two_r_demo'], required=True)
    args.add_argument('--task', type=str, choices=['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative', 'glue_cola', 'glue_sst2', 'glue_mrpc', 'glue_qqp', 'glue_rte', 'glue_qnli'], required=True)
    args.add_argument('--task_data_file', type=str, required=True)
    args.add_argument('--task_start_index', type=int, required=True)
    args.add_argument('--task_end_index', type=int, required=True)
    args.add_argument('--num_generation', type=int, default=1)
    args.add_argument('--additional_output_note', type=str, default="")
    args.add_argument('--temperature', type=float, default=0.0)
    args.add_argument('--top_p', type=float, default=1.0)
    args.add_argument('--system_message', type=str, default="") 
    # "You are an AI assistant that helps people find information",
    args.add_argument('--local_model_path', type=str, default="") # 로컬 모델 경로

    args.add_argument('--num_refine', type=int, default=1) # Perform how many iterations of the self-refinement
    
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = vars(parse_args())
    model_name = args['model']
    model_type = args['model_type']
    
    ### gpt config ###
    if model_type == 'gpt':
        if model_name in gpt_configs:
            args['gpt_config'] = gpt_configs[model_name] # gpt configs
        else:
            args['gpt_config'] = default_gpt_config
            args['gpt_config']['model'] = model_name

        # overwrite temperature and top_p
        args['gpt_config']['temperature'] = args['temperature']
        args['gpt_config']['top_p'] = args['top_p']
    
    elif model_type == 'open_model':
        ### open model config ###
        if model_name in open_model_configs:
            args['open_model_config'] = open_model_configs[model_name] # open model configs
        else:
            args['open_model_config'] = default_open_model_config
            args['open_model_config']['model'] = model_name

    print("run args:", args)
    run(args)