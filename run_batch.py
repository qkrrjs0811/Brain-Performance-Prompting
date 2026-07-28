"""
배치 API로 실험 실행. 동기 run.py와 동일한 로그 스키마를 목표로 함.

- OpenAI Batch (Chat Completions): https://developers.openai.com/api/docs/guides/batch
- Anthropic Message Batches: https://platform.claude.com/docs/ko/build-with-claude/batch-processing

지원: model_type=gpt | claude, trivia_creative_writing, logic_grid_puzzle, hf_*, codenames (2단계)
미지원: model_type=open_model, method=self_refine
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Dict, List

from openai import OpenAI

from configs import (
    claude_configs,
    default_claude_config,
    default_gpt_config,
    gpt_configs,
)
from models import claude_message_batch_usage_totals_cost, gpt_usage_totals_cost
from anthropic_batch import (
    build_codenames_phase1_claude_requests,
    build_codenames_phase2_claude_requests,
    build_single_turn_claude_requests,
    run_claude_batches_and_collect,
)
from openai_batch import (
    build_codenames_phase1_jsonl,
    build_codenames_phase2_jsonl,
    build_single_turn_batch_jsonl,
    merge_codenames_results,
    merge_single_turn_parts,
    run_jsonl_batch_and_collect,
)
from run import (
    load_existing_logs_for_resume,
    _post_process_raw_response,
    output_log_jsonl,
    setup_log_file,
)
from tasks import get_task


def _build_log_updates(args: dict) -> dict:
    u = dict(args)
    if "open_model_config" in u and u["open_model_config"] is not None:
        u["open_model_config"] = dict(u["open_model_config"])
        if "torch_dtype" in u["open_model_config"]:
            u["open_model_config"]["torch_dtype"] = str(u["open_model_config"]["torch_dtype"])
    return u


def _print_progress_line(task_name: str, i: int, log_output: dict) -> None:
    if "test_output_infos" not in log_output or not log_output["test_output_infos"]:
        return
    info = log_output["test_output_infos"][0]
    if task_name == "trivia_creative_writing":
        cc, qc = info["correct_count"], info["question_count"]
        acc = cc / qc * 100 if qc > 0 else 0
        print(f"\tidx: {i} | done | Accuracy: {acc:.2f}%")
    elif task_name == "codenames_collaborative":
        mc, tc = info["matched_count"], info["target_count"]
        acc = mc / tc * 100 if tc > 0 else 0
        print(f"\tidx: {i} | done | Accuracy: {acc:.2f}%")
    elif task_name == "logic_grid_puzzle":
        c = info["correct"]
        print(f"\tidx: {i} | done | Accuracy: {100 if c else 0}% | Correct: {c}")
    elif task_name in ("hf_numeric_math", "hf_multiple_choice"):
        c = info["correct"]
        print(f"\tidx: {i} | done | Accuracy: {100 if c else 0}% | Correct: {c}")


def _choices_texts_from_body(body: dict) -> List[str]:
    texts = []
    for ch in body.get("choices") or []:
        msg = ch.get("message") or {}
        texts.append(msg.get("content") or "")
    return texts


def run_batch(args: dict) -> None:
    mt = args["model_type"]
    if mt not in ("gpt", "claude"):
        raise SystemExit("Batch 실행은 --model_type gpt 또는 claude 만 지원합니다.")
    if args["method"] == "self_refine":
        raise SystemExit("method=self_refine 는 다단계 의존으로 Batch 미지원입니다. python run.py 를 사용하세요.")

    is_claude = mt == "claude"
    task_name = args["task"]
    method = args["method"]
    start_idx = args["task_start_index"]
    end_idx = args["task_end_index"]
    task_data_file = args["task_data_file"]
    num_generation = args["num_generation"]
    system_message = args["system_message"]
    model_config = args["claude_config"] if is_claude else args["gpt_config"]
    model_name = model_config["model"]

    merge_cost_kw = (
        {
            "usage_cost_fn": claude_message_batch_usage_totals_cost,
            "apply_openai_batch_half_price": False,
        }
        if is_claude
        else {}
    )

    output_dir = args["output_dir"] or f"logs/{task_name}"
    log_file = setup_log_file(
        task_name,
        task_data_file,
        method,
        model_config,
        start_idx,
        end_idx,
        args["additional_output_note"],
        system_message,
        output_dir,
    )
    # Batch 실행 결과만 분리하기 위해,
    # 기존 setup_log_file()이 만든 "모델별(sys_mes 유무 포함) 폴더" 바로 아래에 batch_api 폴더를 추가합니다.
    # 예: logs/{task}/{model}_{wo|w}_sys_mes/batch_api/<...>.jsonl
    log_dir = os.path.dirname(log_file)
    batch_api_dir = os.path.join(log_dir, "batch_api")
    os.makedirs(batch_api_dir, exist_ok=True)
    log_file = os.path.join(batch_api_dir, os.path.basename(log_file))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    task = get_task(task_name, file=task_data_file)
    start = max(start_idx, 0)
    end = min(end_idx, len(task))
    if start >= end:
        raise SystemExit(f"유효한 구간이 없습니다: start={start}, end={end}, len(task)={len(task)}")

    existing_logs, resume_start, is_complete = load_existing_logs_for_resume(log_file, start, end)
    if is_complete:
        print(f"skip: 이미 모든 샘플이 완료되어 있습니다. ({start}~{end-1})")
        return
    if resume_start > start:
        print(f"resume: 기존 로그를 감지했습니다. idx {resume_start}부터 이어서 실행합니다.")
        start = resume_start
    if start >= end:
        print(f"skip: 실행할 유효 샘플이 없습니다. start={start}, end={end}")
        return

    if is_claude:
        try:
            import anthropic
        except ImportError as e:
            raise SystemExit("anthropic 패키지가 필요합니다. pip install anthropic") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Anthropic 배치 실행은 ANTHROPIC_API_KEY 가 필요합니다.")
        client = anthropic.Anthropic()
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY 환경 변수를 설정하세요.")
        client = OpenAI()

    poll = args["batch_poll_interval"]
    base_meta = {
        "task": task_name,
        "method": method,
        "run": "brain-performance-prompting batch",
    }
    batch_base_path = log_file + ".batch"
    t0 = time.time()
    all_logs: List[dict] = list(existing_logs)

    if task_name == "codenames_collaborative":
        if is_claude:
            lines1, spy_prompts = build_codenames_phase1_claude_requests(
                claude_config=model_config,
                system_message=system_message,
                start_idx=start,
                end_idx=end,
                get_spymaster_prompt=lambda i: task.get_input_prompt(i, method=method, role="spymaster"),
            )
            with open(batch_base_path + "_phase1_anthropic_requests.jsonl", "w", encoding="utf-8") as f:
                for r in lines1:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            spy_results, bid1 = run_claude_batches_and_collect(
                client,
                lines1,
                poll,
                model_name,
                phase_label="codenames_spymaster",
            )
        else:
            lines1, spy_prompts = build_codenames_phase1_jsonl(
                gpt_config=model_config,
                system_message=system_message,
                start_idx=start,
                end_idx=end,
                get_spymaster_prompt=lambda i: task.get_input_prompt(i, method=method, role="spymaster"),
            )
            spy_results, bid1 = run_jsonl_batch_and_collect(
                client,
                lines1,
                batch_base_path + "_phase1_input.jsonl",
                poll_interval_sec=poll,
                batch_metadata={**base_meta, "phase": "codenames_spymaster"},
            )
        hint_by_idx: Dict[int, str] = {}
        for i in range(start, end):
            br = spy_results.get(f"spy-{i}")
            texts: List[str] = []
            if br and br.ok and br.body:
                texts = _choices_texts_from_body(br.body)
            unwrapped, _ = _post_process_raw_response(task, texts, method)
            if unwrapped:
                hint_by_idx[i] = unwrapped[0].replace(".", "").strip()
            else:
                hint_by_idx[i] = ""

        if is_claude:
            lines2, guess_prompt_map = build_codenames_phase2_claude_requests(
                claude_config=model_config,
                system_message=system_message,
                start_idx=start,
                end_idx=end,
                get_guesser_prompt=lambda i, h: task.get_input_prompt(
                    i, method=method, role="guesser", hint_word=h
                ),
                hint_by_idx=hint_by_idx,
                num_generation=num_generation,
            )
            with open(batch_base_path + "_phase2_anthropic_requests.jsonl", "w", encoding="utf-8") as f:
                for r in lines2:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            guess_results, bid2 = run_claude_batches_and_collect(
                client,
                lines2,
                poll,
                model_name,
                phase_label="codenames_guesser",
            )
        else:
            lines2, guess_prompt_map = build_codenames_phase2_jsonl(
                gpt_config=model_config,
                system_message=system_message,
                start_idx=start,
                end_idx=end,
                get_guesser_prompt=lambda i, h: task.get_input_prompt(
                    i, method=method, role="guesser", hint_word=h
                ),
                hint_by_idx=hint_by_idx,
                num_generation=num_generation,
            )
            guess_results, bid2 = run_jsonl_batch_and_collect(
                client,
                lines2,
                batch_base_path + "_phase2_input.jsonl",
                poll_interval_sec=poll,
                batch_metadata={**base_meta, "phase": "codenames_guesser"},
            )
        merged = merge_codenames_results(
            spy_results,
            guess_results,
            start,
            end,
            spy_prompts,
            guess_prompt_map,
            system_message,
            model_name,
            **merge_cost_kw,
        )
        meta = {
            "phase1_batch_id": bid1,
            "phase2_batch_id": bid2,
            "log_file": log_file,
        }
        for i in range(start, end):
            block = merged[i]
            sp_out, sp_flags = _post_process_raw_response(task, block["spymaster_output"], method)
            g_out, g_flags = _post_process_raw_response(task, block["guesser_output"], method)
            hint_word = sp_out[0].replace(".", "").strip() if sp_out else ""
            test_infos = [task.test_output(i, o) for o in g_out] if g_out else []
            execution_time = time.time() - t0
            log_output = {
                "idx": i,
                "raw_response_spymaster": block["raw_response_spymaster"],
                "raw_response_guesser": block["raw_response_guesser"],
                "spymaster_output": sp_out,
                "guesser_output": g_out,
                "hint_word": hint_word,
                "parsing_success_flag_spymaster": sp_flags,
                "parsing_success_flag_guesser": g_flags,
                "test_output_infos": test_infos,
                "total_execution_time": execution_time,
                "total_usage_info": block["total_usage_info"],
            }
            log_output.update(_build_log_updates(args))
            log_output["task_data"] = task.get_input(i)
            all_logs.append(log_output)
            _print_progress_line(task_name, i, log_output)

    else:
        if is_claude:
            lines, prompts_by_idx = build_single_turn_claude_requests(
                claude_config=model_config,
                system_message=system_message,
                start_idx=start,
                end_idx=end,
                get_prompt=lambda i: task.get_input_prompt(i, method=method),
                num_generation=num_generation,
            )
            with open(batch_base_path + "_anthropic_requests.jsonl", "w", encoding="utf-8") as f:
                for r in lines:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            results, bid = run_claude_batches_and_collect(
                client, lines, poll, model_name, phase_label="single_turn"
            )
        else:
            lines, prompts_by_idx = build_single_turn_batch_jsonl(
                gpt_config=model_config,
                system_message=system_message,
                start_idx=start,
                end_idx=end,
                get_prompt=lambda i: task.get_input_prompt(i, method=method),
                num_generation=num_generation,
            )
            results, bid = run_jsonl_batch_and_collect(
                client,
                lines,
                batch_base_path + "_input.jsonl",
                poll_interval_sec=poll,
                batch_metadata=base_meta,
            )
        meta = {"batch_id": bid, "log_file": log_file}
        merged = merge_single_turn_parts(
            results,
            start,
            end,
            prompts_by_idx,
            system_message,
            model_name,
            **merge_cost_kw,
        )
        for i in range(start, end):
            m = merged[i]
            raw_out = m["raw_output_batch"]
            raw_resp = m["raw_response_batch"]
            unwrapped, flags = _post_process_raw_response(task, raw_out, method)
            test_infos = [task.test_output(i, o) for o in unwrapped] if unwrapped else []
            execution_time = time.time() - t0
            log_output = {
                "idx": i,
                "raw_response": raw_resp,
                "unwrapped_output": unwrapped,
                "parsing_success_flag": flags,
                "test_output_infos": test_infos,
                "usage_info": m["usage_info"],
                "execution_time": execution_time,
            }
            log_output.update(_build_log_updates(args))
            log_output["task_data"] = task.get_input(i)
            all_logs.append(log_output)
            _print_progress_line(task_name, i, log_output)

    output_log_jsonl(log_file, all_logs)
    total_pt = total_ct = 0.0
    for row in all_logs:
        u = row.get("usage_info") or row.get("total_usage_info") or {}
        total_pt += u.get("prompt_tokens", 0)
        total_ct += u.get("completion_tokens", 0)
    if is_claude:
        totals = claude_message_batch_usage_totals_cost(
            model_name, int(total_pt), int(total_ct)
        )
    else:
        totals = gpt_usage_totals_cost(model_name, int(total_pt), int(total_ct))
        totals["cost"] *= 0.5
    meta["aggregate_usage_info"] = totals
    meta_path = log_file + ".batch_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {log_file}")
    print(f"배치 메타: {meta_path}")
    print(f"집계 usage(추정): {totals}")


def parse_args():
    model_choices = list(gpt_configs.keys()) + list(claude_configs.keys())
    p = argparse.ArgumentParser(description="OpenAI 또는 Anthropic Message Batches로 실험 실행")
    p.add_argument("--model", type=str, choices=model_choices, required=True)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument(
        "--model_type",
        type=str,
        choices=["gpt", "claude"],
        default="gpt",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=[
            "standard",
            "cot",
            "spp",
            "bpp",
            "bpp3",
            "macro_bpp",
            "meso_bpp",
            "micro_bpp",
            "bpp_w_r_demo",
            "bpp_w_k_demo",
            "bpp_two_k_demo",
            "bpp_two_r_demo",
        ],
        required=True,
    )
    p.add_argument(
        "--task",
        type=str,
        choices=[
            "trivia_creative_writing",
            "logic_grid_puzzle",
            "codenames_collaborative",
            "hf_numeric_math",
            "hf_multiple_choice",
        ],
        required=True,
    )
    p.add_argument("--task_data_file", type=str, required=True)
    p.add_argument("--task_start_index", type=int, required=True)
    p.add_argument("--task_end_index", type=int, required=True)
    p.add_argument("--num_generation", type=int, default=1)
    p.add_argument(
        "--additional_output_note",
        type=str,
        default="",
        help="로그 파일명 접미사. 비우면 _run01 이 자동 적용됩니다.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="gpt-5 계열 reasoning effort override. 미지정 시 configs.py 값 사용.",
    )
    p.add_argument("--system_message", type=str, default="")
    p.add_argument(
        "--batch_poll_interval",
        type=float,
        default=15.0,
        help="배치 상태 폴링 간격(초)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = vars(parse_args())
    name = args["model"]
    if args["model_type"] == "gpt":
        if name in gpt_configs:
            args["gpt_config"] = dict(gpt_configs[name])
        else:
            args["gpt_config"] = dict(default_gpt_config)
            args["gpt_config"]["model"] = name
        args["gpt_config"]["temperature"] = args["temperature"]
        args["gpt_config"]["top_p"] = args["top_p"]
        if args["reasoning_effort"] is not None:
            args["gpt_config"]["reasoning_effort"] = args["reasoning_effort"]
    else:
        if name in claude_configs:
            args["claude_config"] = copy.deepcopy(claude_configs[name])
        else:
            args["claude_config"] = copy.deepcopy(default_claude_config)
            args["claude_config"]["model"] = name
        args["claude_config"]["temperature"] = args["temperature"]
    print("run_batch args:", args)
    run_batch(args)
