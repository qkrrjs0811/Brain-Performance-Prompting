"""
Anthropic Message Batches API 헬퍼.
가이드: https://platform.claude.com/docs/ko/build-with-claude/batch-processing
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai_batch import BatchLineResult, iter_n_chunks


def _msg_to_dict(message: Any) -> dict:
    if message is None:
        return {}
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump()
    dump = getattr(message, "to_dict", None)
    if callable(dump):
        return dump()
    raise TypeError(f"unsupported message type: {type(message)}")


def claude_succeeded_message_to_openai_shaped_body(
    message: Any, model_fallback: str = ""
) -> dict:
    """배치 결과 message → OpenAI batch merge가 기대하는 choices/usage 형태."""
    msg = _msg_to_dict(message)
    parts: List[str] = []
    for block in msg.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    text = "".join(parts)
    u = msg.get("usage") or {}
    in_tok = int(u.get("input_tokens") or 0)
    out_tok = int(u.get("output_tokens") or 0)
    usage_dict = {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }
    stop_r = msg.get("stop_reason")
    return {
        "id": msg.get("id", ""),
        "object": "chat.completion",
        "model": msg.get("model") or model_fallback,
        "choices": [
            {
                "index": 0,
                "finish_reason": str(stop_r) if stop_r is not None else "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": usage_dict,
    }


def claude_config_to_message_params(
    claude_config: dict, user_prompt: str, system_message: str
) -> dict:
    params: Dict[str, Any] = {
        "model": claude_config["model"],
        "max_tokens": int(claude_config["max_tokens"]),
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if system_message:
        params["system"] = system_message
    thinking = claude_config.get("thinking")
    output_config = claude_config.get("output_config")
    has_reasoning_controls = isinstance(thinking, dict) or isinstance(output_config, dict)
    if isinstance(thinking, dict):
        params["thinking"] = dict(thinking)
    if isinstance(output_config, dict):
        params["output_config"] = dict(output_config)
    # Claude에서 thinking/effort 제어를 쓸 때는 temperature를 함께 보내지 않는다.
    if (
        not has_reasoning_controls
        and "temperature" in claude_config
        and claude_config["temperature"] is not None
    ):
        params["temperature"] = float(claude_config["temperature"])
    stops = claude_config.get("stop_sequences")
    if stops:
        params["stop_sequences"] = stops
    return params


def build_single_turn_claude_requests(
    *,
    claude_config: dict,
    system_message: str,
    start_idx: int,
    end_idx: int,
    get_prompt: Callable[[int], str],
    num_generation: int,
) -> Tuple[List[dict], Dict[int, str]]:
    requests: List[dict] = []
    prompts_by_idx: Dict[int, str] = {}
    for i in range(start_idx, end_idx):
        prompt = get_prompt(i)
        prompts_by_idx[i] = prompt
        part = 0
        for c in iter_n_chunks(num_generation):
            for _ in range(c):
                cid = f"i{i}-p{part}"
                params = claude_config_to_message_params(
                    claude_config, prompt, system_message
                )
                requests.append({"custom_id": cid, "params": params})
                part += 1
    return requests, prompts_by_idx


def build_codenames_phase1_claude_requests(
    *,
    claude_config: dict,
    system_message: str,
    start_idx: int,
    end_idx: int,
    get_spymaster_prompt: Callable[[int], str],
) -> Tuple[List[dict], Dict[int, str]]:
    requests: List[dict] = []
    prompts_by_idx: Dict[int, str] = {}
    for i in range(start_idx, end_idx):
        prompt = get_spymaster_prompt(i)
        prompts_by_idx[i] = prompt
        params = claude_config_to_message_params(
            claude_config, prompt, system_message
        )
        requests.append({"custom_id": f"spy-{i}", "params": params})
    return requests, prompts_by_idx


def build_codenames_phase2_claude_requests(
    *,
    claude_config: dict,
    system_message: str,
    start_idx: int,
    end_idx: int,
    get_guesser_prompt: Callable[[int, str], str],
    hint_by_idx: Dict[int, str],
    num_generation: int,
) -> Tuple[List[dict], Dict[Tuple[int, int], str]]:
    requests: List[dict] = []
    prompts: Dict[Tuple[int, int], str] = {}
    for i in range(start_idx, end_idx):
        hint = hint_by_idx.get(i, "")
        prompt = get_guesser_prompt(i, hint)
        part = 0
        for c in iter_n_chunks(num_generation):
            for _ in range(c):
                cid = f"guess-{i}-p{part}"
                params = claude_config_to_message_params(
                    claude_config, prompt, system_message
                )
                requests.append({"custom_id": cid, "params": params})
                prompts[(i, part)] = prompt
                part += 1
    return requests, prompts


def _result_item_custom_id(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("custom_id") or "")
    return str(getattr(row, "custom_id", "") or "")


def _result_item_inner_result(row: Any) -> Any:
    if isinstance(row, dict):
        return row.get("result")
    return getattr(row, "result", None)


def _anthropic_results_to_map(
    client: Any,
    batch_id: str,
    *,
    model_name: str,
) -> Dict[str, BatchLineResult]:
    out: Dict[str, BatchLineResult] = {}

    stream = client.messages.batches.results(batch_id)
    for item in stream:
        cid = _result_item_custom_id(item)
        inner = _result_item_inner_result(item)
        if inner is None:
            out[cid] = BatchLineResult(cid, False, None, {"message": "missing result"})
            continue

        rtype = getattr(inner, "type", None)
        if rtype is None and isinstance(inner, dict):
            rtype = inner.get("type")
        if hasattr(rtype, "value"):
            rtype = rtype.value
        rtype = str(rtype) if rtype is not None else ""

        if rtype == "succeeded":
            msg = getattr(inner, "message", None)
            if msg is None and isinstance(inner, dict):
                msg = inner.get("message")
            try:
                body = claude_succeeded_message_to_openai_shaped_body(msg, model_name)
            except Exception as e:
                out[cid] = BatchLineResult(
                    cid, False, None, {"message": "parse succeeded message", "error": str(e)}
                )
                continue
            out[cid] = BatchLineResult(cid, True, body, None)
            continue

        if rtype == "errored":
            err = getattr(inner, "error", None)
            if err is None and isinstance(inner, dict):
                err = inner.get("error")
            ed: Any = err
            if hasattr(ed, "model_dump"):
                ed = ed.model_dump()
            elif not isinstance(ed, dict):
                ed = {"repr": repr(ed)}
            out[cid] = BatchLineResult(cid, False, None, {"type": "errored", "error": ed})
            continue

        if rtype in ("canceled", "expired"):
            out[cid] = BatchLineResult(
                cid, False, None, {"type": rtype},
            )
            continue

        out[cid] = BatchLineResult(
            cid, False, None, {"message": "unknown batch result type", "type": rtype},
        )
    return out


def wait_anthropic_batch_ended(
    client: Any,
    batch_id: str,
    poll_interval_sec: float,
    on_status: Optional[Callable[[Any], None]] = None,
) -> Any:
    """processing_status 가 ended 가 될 때까지 폴링."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if on_status:
            on_status(batch)
        st = getattr(batch, "processing_status", None)
        if isinstance(batch, dict):
            st = batch.get("processing_status")
        if st == "ended":
            return batch
        time.sleep(poll_interval_sec)


def submit_claude_message_batch(client: Any, requests: List[dict]) -> str:
    if not requests:
        raise ValueError("requests 비어 있음")
    mb = client.messages.batches.create(requests=requests)
    bid = getattr(mb, "id", None)
    if bid is None and isinstance(mb, dict):
        bid = mb.get("id")
    if not bid:
        raise RuntimeError("batch id 없음")
    return bid


def run_claude_batches_and_collect(
    client: Any,
    requests: List[dict],
    poll_interval_sec: float,
    model_name: str,
    *,
    phase_label: str = "",
) -> Tuple[Dict[str, BatchLineResult], str]:
    bid = submit_claude_message_batch(client, requests)
    label = phase_label or "claude-batch"
    print(f"{label}: submitted batch_id={bid}")

    def _cb(b: Any) -> None:
        st = getattr(b, "processing_status", None)
        rc = getattr(b, "request_counts", None)
        if isinstance(b, dict):
            st, rc = b.get("processing_status"), b.get("request_counts")
        if rc is not None and hasattr(rc, "model_dump"):
            rc = rc.model_dump()
        print(f"  [{label}] processing_status={st} request_counts={rc}")

    wait_anthropic_batch_ended(client, bid, poll_interval_sec=poll_interval_sec, on_status=_cb)
    by_id = _anthropic_results_to_map(client, bid, model_name=model_name)

    errs = sum(1 for v in by_id.values() if not v.ok)
    if errs:
        print(f"Warning: [{label}] {errs}/{len(by_id)} 항목이 실패(errored/skipped)")
    return by_id, bid
