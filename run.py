import copy
import os
import json
import argparse
import re
from models import AnthropicWrapper, OpenAIWrapper, OpenModelWrapper
from tasks import get_task
import time
from configs import (
    claude_configs,
    default_claude_config,
    default_gpt_config,
    default_open_model_config,
    gpt_configs,
    open_model_configs,
)
from log_path_utils import model_log_folder_stem, normalize_additional_output_note


SLEEP_RATE = 10 # sleep between calls


import os

# 로그 파일을 저장할 경로를 설정하는 함수
def _dataset_name_from_task_data_file(task_data_file: str) -> str:
    base = os.path.splitext(os.path.basename(task_data_file))[0]
    # 예: gsm8k.jsonl -> gsm8k, mmlu_pro_100 -> mmlu_pro
    m = re.match(r"^(.*)_\d+$", base)
    return (m.group(1) if m else base) or base


def setup_log_file(task_name, task_data_file, method, model_config, start_idx, end_idx, additional_output_note, system_message, output_dir):
    additional_output_note = normalize_additional_output_note(additional_output_note)
    # 모델 이름을 기반으로 폴더 생성
    model_name_for_output = model_log_folder_stem(model_config)
    
    # system_message 유무에 따라 상위 폴더명 결정
    sys_mes_folder = f"{model_name_for_output}_{'w_sys_mes' if system_message else 'wo_sys_mes'}"
    
    # hf 태스크는 모델/시스템메시지구분 + 데이터셋 기준으로 하위폴더를 분리해서 저장
    # 예: logs/hf_numeric_math/<model>_{w|wo}_sys_mes/<dataset>/...
    if task_name in ["hf_numeric_math", "hf_multiple_choice"]:
        dataset_name = _dataset_name_from_task_data_file(task_data_file)
        log_dir = os.path.normpath(os.path.join(output_dir, sys_mes_folder, dataset_name))
    else:
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


def load_existing_logs_for_resume(log_file, start_idx, end_idx):
    """
    기존 JSONL 로그를 읽어 재개 지점을 계산한다.
    - 목표 구간의 idx가 모두 존재하면 complete=True
    - 일부만 존재하면 첫 미완료 idx를 resume_start로 반환
    """
    if not os.path.exists(log_file):
        return [], start_idx, False

    existing_logs = []
    observed_idx = set()

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # 깨진 라인은 무시하고 진행
                continue
            if isinstance(row, dict):
                existing_logs.append(row)
                idx = row.get("idx")
                if isinstance(idx, int) and start_idx <= idx < end_idx:
                    observed_idx.add(idx)

    resume_start = start_idx
    for i in range(start_idx, end_idx):
        if i not in observed_idx:
            resume_start = i
            break
    else:
        return existing_logs, end_idx, True

    return existing_logs, resume_start, False

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
    if task_name in ['trivia_creative_writing', 'logic_grid_puzzle', 'hf_numeric_math', 'hf_multiple_choice']:
        if method == "self_refine":
            log_output = _run_self_refine_default(
                model, task, i, method, num_generation, sleep_rate, num_refine=kwargs['num_refine']
            )
        else:
            log_output = _run_task_default(model, task, i, method, num_generation, sleep_rate)
    elif task_name == 'codenames_collaborative':
        if method == "self_refine":
            log_output = _run_self_refine_codenames(model, task, i, method, num_generation, sleep_rate, num_refine = kwargs['num_refine'])
        else:
            log_output = _run_task_codenames(model, task, i, method, num_generation, sleep_rate)
    else:
        raise NotImplementedError(
            f"task {task_name} not implemented; please choose from ['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative', 'hf_*']"
        )

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
    
    # setup model and output log file
    if model_type == 'gpt':
        model_config = args['gpt_config']
        model = OpenAIWrapper(config=model_config, system_message=system_message)
        # setup log file
        log_file = setup_log_file(task_name, task_data_file, method, model_config, start_idx, end_idx, additional_output_note, system_message, output_dir)
        sleep_rate = SLEEP_RATE

    elif model_type == 'open_model':
        model_config = args['open_model_config']
        print(f"setting default system message: {system_message}")
        
        local_model_path = args.get('local_model_path') or None
        model = OpenModelWrapper(config=model_config, local_model_path=local_model_path)
        # setup log file
        log_file = setup_log_file(task_name, task_data_file, method, model_config, start_idx, end_idx, additional_output_note, system_message, output_dir)
        sleep_rate = 0

    elif model_type == "claude":
        model_config = args["claude_config"]
        model = AnthropicWrapper(config=model_config, system_message=system_message)
        log_file = setup_log_file(
            task_name,
            task_data_file,
            method,
            model_config,
            start_idx,
            end_idx,
            additional_output_note,
            system_message,
            output_dir,
        )
        sleep_rate = SLEEP_RATE

    else:
        raise ValueError(f"unknown model_type: {model_type}")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # setup task
    task = get_task(task_name, file=task_data_file)
    
    all_logs = []
    print("start running ... log file:", log_file)
    print("sleep rate:", sleep_rate)

    print()
    start = max(start_idx, 0)
    end = min(end_idx, len(task))
    all_logs, resume_start, is_complete = load_existing_logs_for_resume(log_file, start, end)
    if is_complete:
        print(f"skip: 이미 모든 샘플이 완료되어 있습니다. ({start}~{end-1})")
        return
    if resume_start > start:
        print(f"resume: 기존 로그를 감지했습니다. idx {resume_start}부터 이어서 실행합니다.")
        start = resume_start
    else:
        # 기존 로그가 없거나(또는 시작 idx 이전만 존재) 신규 시작
        if not all_logs:
            all_logs = []

    total_instances = end - start
    if total_instances <= 0:
        print(f"skip: 실행할 유효 샘플이 없습니다. start={start}, end={end}")
        return
    print("total num of instances:", total_instances)
    print("method:", method)
    
    # 타이머 시작
    start_time = time.time()
    
    for i in range(start, end):
        log_output = _run_task(task_name, model, task, i, method, num_generation, sleep_rate, num_refine = args['num_refine'])
        
        # 현재 진행률 계산
        progress = ((i - start + 1) / total_instances) * 100
        elapsed_time = time.time() - start_time  # 경과 시간
        estimated_total_time = elapsed_time / (i - start + 1) * total_instances  # 총 예상 시간
        remaining_time = estimated_total_time - elapsed_time  # 남은 시간
        
        # 시간 정보를 log_output에 추가
        if log_output:
            log_output['progress'] = progress
            log_output['elapsed_time'] = elapsed_time
            log_output['estimated_total_time'] = estimated_total_time
            log_output['remaining_time'] = remaining_time
        
        all_logs.append(log_output)
        
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

            # hf_numeric_math
            elif task_name == 'hf_numeric_math':
                correct = log_output['test_output_infos'][0]['correct']
                accuracy = 100 if correct else 0
                print(f"\tidx: {i} | done | Accuracy: {accuracy}% | Correct: {correct}")

            # hf_multiple_choice
            elif task_name == 'hf_multiple_choice':
                correct = log_output['test_output_infos'][0]['correct']
                accuracy = 100 if correct else 0
                print(f"\tidx: {i} | done | Accuracy: {accuracy}% | Correct: {correct}")
            
        print(f"\tidx: {i}, done | Progress: {progress:.2f}% | Elapsed time: {elapsed_time:.2f}s | Estimated total time: {estimated_total_time:.2f}s | Estimated remaining time: {remaining_time:.2f}s | Total usage so far: {model.compute_gpt_usage()}")
        # output log at each iteration
        output_log_jsonl(log_file, all_logs)
        # sleep
        time.sleep(sleep_rate)


def parse_args():
    model_choices = (
        list(gpt_configs.keys())
        + list(open_model_configs.keys())
        + list(claude_configs.keys())
    )
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=model_choices, required=True) # gpt-4o, gpt35-turbo, meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct
    args.add_argument('--output_dir', type=str, required=False, default="")
    args.add_argument(
        '--model_type',
        type=str,
        choices=['gpt', 'open_model', 'claude'],
        default='gpt',
    )
    args.add_argument('--method', type=str, choices=['standard','cot','spp', 'self_refine', 'bpp3', 'bpp', 'bpp_w_r_demo', 'bpp_w_k_demo', 'bpp_two_k_demo', 'bpp_two_r_demo'], required=True)
    args.add_argument(
        '--task',
        type=str,
        choices=[
            'trivia_creative_writing',
            'logic_grid_puzzle',
            'codenames_collaborative',
            'hf_numeric_math',
            'hf_multiple_choice',
        ],
        required=True
    )
    args.add_argument('--task_data_file', type=str, required=True)
    args.add_argument('--task_start_index', type=int, required=True)
    args.add_argument('--task_end_index', type=int, required=True)
    args.add_argument('--num_generation', type=int, default=1)
    args.add_argument(
        '--additional_output_note',
        type=str,
        default="",
        help="로그 파일명에 start-end 뒤에 붙는 접미사. 비우면 _run01 이 자동 적용됩니다.",
    )
    args.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='gpt: API temperature. open_model은 configs.open_model_configs의 모델별 값 사용.',
    )
    args.add_argument(
        '--top_p',
        type=float,
        default=1.0,
        help='gpt: API top_p. open_model은 configs.open_model_configs의 모델별 값 사용.',
    )
    args.add_argument(
        '--reasoning_effort',
        type=str,
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="gpt-5 계열 reasoning effort override. 미지정 시 configs.py 값 사용.",
    )
    args.add_argument('--system_message', type=str, default="") 
    # "You are an AI assistant that helps people find information",
    args.add_argument('--local_model_path', type=str, default="") # 로컬 모델 경로

    args.add_argument('--num_refine', type=int, default=1) # Perform how many iterations of the self-refinement
    args.add_argument(
        '--seed',
        type=int,
        default=None,
        help=(
            "open_model: vLLM SamplingParams.seed. "
            "gpt: Chat Completions seed(지원 모델만). "
            "claude: 미사용. 생략 시 엔진 기본 동작."
        ),
    )
    args.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=None,
        help=(
            "open_model 전용: vLLM LLM(..., gpu_memory_utilization=값). "
            "예: 0.85"
        ),
    )

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
        if args['reasoning_effort'] is not None:
            args['gpt_config']['reasoning_effort'] = args['reasoning_effort']
        if args['seed'] is not None:
            args['gpt_config']['seed'] = args['seed']

    elif model_type == 'open_model':
        ### open model config ###
        if model_name in open_model_configs:
            args['open_model_config'] = copy.deepcopy(open_model_configs[model_name])
        else:
            args['open_model_config'] = copy.deepcopy(default_open_model_config)
            args['open_model_config']['model'] = model_name
        # temperature / top_p는 모델별 configs.py만 사용 (DeepSeek 등 기본값 유지)
        if args['seed'] is not None:
            args['open_model_config']['seed'] = args['seed']
        if args['gpu_memory_utilization'] is not None:
            gm = args['gpu_memory_utilization']
            if not (0.0 < gm <= 1.0):
                raise ValueError("--gpu_memory_utilization must be in (0, 1].")
            args['open_model_config']['gpu_memory_utilization'] = gm

    elif model_type == "claude":
        if model_name in claude_configs:
            args["claude_config"] = copy.deepcopy(claude_configs[model_name])
        else:
            args["claude_config"] = copy.deepcopy(default_claude_config)
            args["claude_config"]["model"] = model_name
        args["claude_config"]["temperature"] = args["temperature"]

    print("run args:", args)
    run(args)