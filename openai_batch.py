"""
OpenAI Batch API 헬퍼 (Chat Completions).
가이드: https://developers.openai.com/api/docs/guides/batch
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from openai import OpenAI

# OpenAIWrapper.run 과 동일: 한 번에 최대 10 completion (n>10이면 여러 요청으로 분할)
_MAX_N_PER_REQUEST = 10


def build_chat_messages(user_prompt: str, system_message: str = "") -> List[dict]:
    if system_message:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]
    return [{"role": "user", "content": user_prompt}]


def gpt_config_to_chat_completion_body(
    gpt_config: dict,
    messages: List[dict],
    n: int,
) -> dict:
    """Batch 한 줄의 body에 넣을 Chat Completions 파라미터 (stream 미사용)."""
    body: Dict[str, Any] = {
        "model": gpt_config["model"],
        "messages": messages,
        "n": n,
    }
    for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"):
        if key in gpt_config and gpt_config[key] is not None:
            body[key] = gpt_config[key]
    # gpt-5 계열 reasoning 제어를 배치 경로에서도 동기(run.py)와 동일하게 전달한다.
    if "reasoning_effort" in gpt_config and gpt_config["reasoning_effort"] is not None:
        body["reasoning_effort"] = gpt_config["reasoning_effort"]
    if "max_completion_tokens" in gpt_config:
        body["max_completion_tokens"] = gpt_config["max_completion_tokens"]
    elif "max_tokens" in gpt_config:
        body["max_tokens"] = gpt_config["max_tokens"]
    return body


def _attach_prompt_metadata(raw_body: dict, prompt: str, system_message: str) -> dict:
    out = dict(raw_body)
    out["prompt"] = prompt
    if system_message:
        out["system_message"] = system_message
    return out


def iter_n_chunks(total_n: int) -> Iterator[int]:
    rem = total_n
    while rem > 0:
        c = min(rem, _MAX_N_PER_REQUEST)
        yield c
        rem -= c


def _parse_custom_id_single(custom_id: str) -> Tuple[int, int]:
    """형식 i{idx}-p{part}"""
    m = re.match(r"^i(\d+)-p(\d+)$", custom_id)
    if not m:
        raise ValueError(f"unexpected custom_id: {custom_id}")
    return int(m.group(1)), int(m.group(2))


def _parse_custom_id_spy(custom_id: str) -> int:
    m = re.match(r"^spy-(\d+)$", custom_id)
    if not m:
        raise ValueError(f"unexpected custom_id: {custom_id}")
    return int(m.group(1))


def _parse_custom_id_guess(custom_id: str) -> Tuple[int, int]:
    m = re.match(r"^guess-(\d+)-p(\d+)$", custom_id)
    if not m:
        raise ValueError(f"unexpected custom_id: {custom_id}")
    return int(m.group(1)), int(m.group(2))


@dataclass
class BatchLineResult:
    custom_id: str
    ok: bool
    body: Optional[dict]
    error: Optional[dict]


def write_batch_jsonl_lines(
    lines: List[dict],
    path: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def submit_chat_completions_batch(
    client: OpenAI,
    jsonl_path: str,
    completion_window: str = "24h",
    metadata: Optional[dict] = None,
) -> Any:
    with open(jsonl_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    return client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata=metadata or {},
    )


def wait_for_batch(
    client: OpenAI,
    batch_id: str,
    poll_interval_sec: float = 15.0,
    on_status: Optional[Callable[[Any], None]] = None,
) -> Any:
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        b = client.batches.retrieve(batch_id)
        if on_status:
            on_status(b)
        if b.status in terminal:
            return b
        time.sleep(poll_interval_sec)


def download_batch_output_text(client: OpenAI, file_id: str) -> str:
    return client.files.content(file_id).text


def parse_batch_output_jsonl(text: str) -> Dict[str, BatchLineResult]:
    by_id: Dict[str, BatchLineResult] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cid = row["custom_id"]
        err = row.get("error")
        resp = row.get("response")
        if err:
            by_id[cid] = BatchLineResult(cid, False, None, err if isinstance(err, dict) else {"message": str(err)})
            continue
        if not resp or resp.get("status_code") != 200:
            by_id[cid] = BatchLineResult(
                cid,
                False,
                None,
                {"message": "non-200 or missing response", "response": resp},
            )
            continue
        body = resp.get("body")
        if isinstance(body, dict):
            by_id[cid] = BatchLineResult(cid, True, body, None)
        else:
            by_id[cid] = BatchLineResult(cid, False, None, {"message": "body not dict"})
    return by_id


def parse_batch_error_jsonl(text: str) -> List[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def merge_single_turn_parts(
    results: Dict[str, BatchLineResult],
    start_idx: int,
    end_idx: int,
    prompts_by_idx: Dict[int, str],
    system_message: str,
    model_name: str,
    *,
    usage_cost_fn=None,
    apply_openai_batch_half_price: bool = True,
) -> Dict[int, dict]:
    """
    i{{idx}}-p{{part}} 결과를 idx별로 합쳐 run.py의 raw_response 리스트 형태로 만든다.
    반환: idx -> { raw_output_batch, raw_response_batch, usage_info (해당 idx만) }
    """
    from models import gpt_usage_totals_cost

    cost_fn = usage_cost_fn or gpt_usage_totals_cost

    # 그룹: idx -> list of (part, BatchLineResult)
    groups: Dict[int, List[Tuple[int, BatchLineResult]]] = {i: [] for i in range(start_idx, end_idx)}
    for cid, br in results.items():
        try:
            idx, part = _parse_custom_id_single(cid)
        except ValueError:
            continue
        if start_idx <= idx < end_idx:
            groups.setdefault(idx, []).append((part, br))

    out = {}
    for idx in range(start_idx, end_idx):
        parts = sorted(groups.get(idx, []), key=lambda x: x[0])
        raw_output_batch: List[str] = []
        raw_response_batch: List[dict] = []
        pt = ct = 0
        user_prompt = prompts_by_idx[idx]

        for _part_num, br in parts:
            if not br.ok or not br.body:
                continue
            b = br.body
            choices = b.get("choices") or []
            for ch in choices:
                msg = (ch.get("message") or {})
                content = msg.get("content") or ""
                raw_output_batch.append(content)
            usage = b.get("usage") or {}
            pt += int(usage.get("prompt_tokens") or 0)
            ct += int(usage.get("completion_tokens") or 0)
            raw_response_batch.append(
                _attach_prompt_metadata(b, user_prompt, system_message)
            )

        usage_info = dict(cost_fn(model_name, pt, ct))
        if apply_openai_batch_half_price:
            usage_info["cost"] *= 0.5
        out[idx] = {
            "raw_output_batch": raw_output_batch,
            "raw_response_batch": raw_response_batch,
            "usage_info": usage_info,
        }
    return out


def build_single_turn_batch_jsonl(
    *,
    gpt_config: dict,
    system_message: str,
    start_idx: int,
    end_idx: int,
    get_prompt: Callable[[int], str],
    num_generation: int,
) -> Tuple[List[dict], Dict[int, str]]:
    lines: List[dict] = []
    prompts_by_idx: Dict[int, str] = {}
    for i in range(start_idx, end_idx):
        prompt = get_prompt(i)
        prompts_by_idx[i] = prompt
        messages = build_chat_messages(prompt, system_message)
        part = 0
        for c in iter_n_chunks(num_generation):
            cid = f"i{i}-p{part}"
            body = gpt_config_to_chat_completion_body(gpt_config, messages, c)
            lines.append(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
            part += 1
    return lines, prompts_by_idx


def build_codenames_phase1_jsonl(
    *,
    gpt_config: dict,
    system_message: str,
    start_idx: int,
    end_idx: int,
    get_spymaster_prompt: Callable[[int], str],
) -> Tuple[List[dict], Dict[int, str]]:
    lines: List[dict] = []
    prompts_by_idx: Dict[int, str] = {}
    for i in range(start_idx, end_idx):
        prompt = get_spymaster_prompt(i)
        prompts_by_idx[i] = prompt
        messages = build_chat_messages(prompt, system_message)
        body = gpt_config_to_chat_completion_body(gpt_config, messages, 1)
        lines.append(
            {
                "custom_id": f"spy-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
    return lines, prompts_by_idx


def build_codenames_phase2_jsonl(
    *,
    gpt_config: dict,
    system_message: str,
    start_idx: int,
    end_idx: int,
    get_guesser_prompt: Callable[[int, str], str],
    hint_by_idx: Dict[int, str],
    num_generation: int,
) -> Tuple[List[dict], Dict[Tuple[int, int], str]]:
    """키 (idx, part) -> user_prompt (로그용)"""
    lines: List[dict] = []
    prompts: Dict[Tuple[int, int], str] = {}
    for i in range(start_idx, end_idx):
        hint = hint_by_idx.get(i, "")
        prompt = get_guesser_prompt(i, hint)
        part = 0
        for c in iter_n_chunks(num_generation):
            cid = f"guess-{i}-p{part}"
            messages = build_chat_messages(prompt, system_message)
            body = gpt_config_to_chat_completion_body(gpt_config, messages, c)
            lines.append(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
            prompts[(i, part)] = prompt
            part += 1
    return lines, prompts


def merge_codenames_results(
    spy_results: Dict[str, BatchLineResult],
    guess_results: Dict[str, BatchLineResult],
    start_idx: int,
    end_idx: int,
    spy_prompts: Dict[int, str],
    guess_prompts: Dict[Tuple[int, int], str],
    system_message: str,
    model_name: str,
    *,
    usage_cost_fn=None,
    apply_openai_batch_half_price: bool = True,
) -> Dict[int, dict]:
    from models import gpt_usage_totals_cost

    cost_fn = usage_cost_fn or gpt_usage_totals_cost

    out: Dict[int, dict] = {}
    for i in range(start_idx, end_idx):
        spy_br = spy_results.get(f"spy-{i}")
        raw_spy_list: List[dict] = []
        spy_texts: List[str] = []
        s_pt = s_ct = 0
        sp = spy_prompts[i]
        if spy_br and spy_br.ok and spy_br.body:
            b = spy_br.body
            choices = b.get("choices") or []
            for ch in choices:
                msg = (ch.get("message") or {})
                spy_texts.append(msg.get("content") or "")
            u = b.get("usage") or {}
            s_pt += int(u.get("prompt_tokens") or 0)
            s_ct += int(u.get("completion_tokens") or 0)
            raw_spy_list.append(_attach_prompt_metadata(b, sp, system_message))

        groups: List[Tuple[int, BatchLineResult]] = []
        for cid, br in guess_results.items():
            try:
                gi, gp = _parse_custom_id_guess(cid)
            except ValueError:
                continue
            if gi == i:
                groups.append((gp, br))
        groups.sort(key=lambda x: x[0])

        guess_texts: List[str] = []
        raw_guess_list: List[dict] = []
        g_pt = g_ct = 0
        for gp, br in groups:
            prompt = guess_prompts.get((i, gp), "")
            if not br.ok or not br.body:
                continue
            b = br.body
            choices = b.get("choices") or []
            for ch in choices:
                msg = (ch.get("message") or {})
                guess_texts.append(msg.get("content") or "")
            u = b.get("usage") or {}
            g_pt += int(u.get("prompt_tokens") or 0)
            g_ct += int(u.get("completion_tokens") or 0)
            raw_guess_list.append(_attach_prompt_metadata(b, prompt, system_message))

        usage = dict(cost_fn(model_name, s_pt + g_pt, s_ct + g_ct))
        if apply_openai_batch_half_price:
            usage["cost"] *= 0.5
        out[i] = {
            "raw_response_spymaster": raw_spy_list,
            "raw_response_guesser": raw_guess_list,
            "spymaster_output": spy_texts,
            "guesser_output": guess_texts,
            "total_usage_info": usage,
        }
    return out


def run_jsonl_batch_and_collect(
    client: OpenAI,
    lines: List[dict],
    input_jsonl_path: str,
    poll_interval_sec: float,
    batch_metadata: dict,
) -> Tuple[Dict[str, BatchLineResult], str]:
    write_batch_jsonl_lines(lines, input_jsonl_path)
    batch = submit_chat_completions_batch(
        client,
        input_jsonl_path,
        metadata=batch_metadata,
    )
    bid = batch.id
    print(f"Batch submitted: {bid}, status={batch.status}")

    def _cb(b: Any) -> None:
        rc = getattr(b, "request_counts", None)
        total = getattr(rc, "total", None) if rc else None
        completed = getattr(rc, "completed", None) if rc else None
        if total is not None and completed is not None:
            print(f"  batch {b.id} status={b.status} completed={completed}/{total}")
        else:
            print(f"  batch {b.id} status={b.status}")

    final = wait_for_batch(client, bid, poll_interval_sec=poll_interval_sec, on_status=_cb)
    if final.status != "completed":
        err = getattr(final, "errors", None)
        raise RuntimeError(f"Batch ended with status={final.status} errors={err}")

    out_f = final.output_file_id
    err_f = final.error_file_id
    if not out_f:
        raise RuntimeError("Batch completed but output_file_id is missing")

    text = download_batch_output_text(client, out_f)
    by_id = parse_batch_output_jsonl(text)
    if err_f:
        err_text = download_batch_output_text(client, err_f)
        err_rows = parse_batch_error_jsonl(err_text)
        if err_rows:
            print(f"Warning: batch error file has {len(err_rows)} line(s) (partial failures).")
    return by_id, bid
