"""
run.py / accuracy 스크립트가 동일한 규칙으로 로그 상위 폴더명을 만들 때 사용합니다.

- gpt-5*: reasoning_effort -> ..._re-{effort}_...
- claude-*: thinking.type, output_config.effort -> ..._think-{type}_eff-{effort}_... (설정된 항목만)
- open_model(vLLM): temperature, top_p -> ..._temp-{t}_topp-{p}_...
  (temperature==0 이면 top_p는 폴더명에 넣지 않음 — 결정적 샘플링에서 의미 없음)
"""


def normalize_additional_output_note(note) -> str:
    """
    additional_output_note가 비어 있으면 단일 레플리케이트 규격에 맞춰 _run01을 부여합니다.
    (스크립트에서 명시적으로 _run02 등을 넘긴 경우는 그대로 둠.)
    """
    if note is None or str(note).strip() == "":
        return "_run01"
    return str(note)


def _is_zero_temperature(temperature) -> bool:
    if temperature is None:
        return False
    if isinstance(temperature, bool):
        return False
    if isinstance(temperature, (int, float)):
        return float(temperature) == 0.0
    return False


def fmt_folder_num(v) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return format(v, "g")
    return str(v)


def sampling_segment_from_values(temperature, top_p) -> str:
    parts = []
    if temperature is not None:
        parts.append(f"temp-{fmt_folder_num(temperature)}")
    if top_p is not None and not _is_zero_temperature(temperature):
        parts.append(f"topp-{fmt_folder_num(top_p)}")
    return "_".join(parts)


def sampling_segment_from_config(model_config: dict) -> str:
    t = model_config.get("temperature") if "temperature" in model_config else None
    tp = model_config.get("top_p") if "top_p" in model_config else None
    if t is None and tp is None:
        return ""
    return sampling_segment_from_values(t, tp)


def is_open_model_config(model_config: dict) -> bool:
    return "max_model_len" in model_config and "max_new_tokens" in model_config


def claude_stem_segments_from_config(model_config: dict) -> str:
    """thinking.type / output_config.effort 를 gpt-5 의 _re- 와 같은 식으로 stem 에 붙인다."""
    parts = []
    thinking = model_config.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") is not None:
        parts.append(f"think-{fmt_folder_num(thinking['type'])}")
    oc = model_config.get("output_config")
    if isinstance(oc, dict) and oc.get("effort") is not None:
        parts.append(f"eff-{fmt_folder_num(oc['effort'])}")
    return "_".join(parts)


def model_log_folder_stem(model_config: dict) -> str:
    """로그 상위 폴더에서 '_{w|wo}_sys_mes' 앞까지의 stem (HF id의 '/'는 '-')."""
    model_name = model_config["model"].replace("/", "-")
    reasoning_effort = model_config.get("reasoning_effort")
    if model_name.startswith("gpt-5") and reasoning_effort:
        return f"{model_name}_re-{reasoning_effort}"
    if model_name.startswith("claude-"):
        seg = claude_stem_segments_from_config(model_config)
        if seg:
            return f"{model_name}_{seg}"
        return model_name
    if is_open_model_config(model_config):
        seg = sampling_segment_from_config(model_config)
        if seg:
            return f"{model_name}_{seg}"
    return model_name


def open_sampling_params_for_accuracy(model_name: str, cli_temp, cli_top_p, open_model_configs: dict):
    """accuracy CLI와 configs.open_model_configs 기본값을 합쳐 temp/top_p를 정한다."""
    t, tp = None, None
    if model_name in open_model_configs:
        oc = open_model_configs[model_name]
        t = oc.get("temperature")
        tp = oc.get("top_p")
    if cli_temp is not None:
        t = cli_temp
    if cli_top_p is not None:
        tp = cli_top_p
    return t, tp


def insert_sampling_before_sys_mes_folder(folder_path: str, temperature, top_p) -> str:
    """..._{w|wo}_sys_mes 경로에 temp/top_p 세그먼트를 stem과 sys_mes 사이에 삽입."""
    seg = sampling_segment_from_values(temperature, top_p)
    if not seg:
        return folder_path
    if folder_path.endswith("_w_sys_mes"):
        base = folder_path[: -len("_w_sys_mes")]
        return f"{base}_{seg}_w_sys_mes"
    if folder_path.endswith("_wo_sys_mes"):
        base = folder_path[: -len("_wo_sys_mes")]
        return f"{base}_{seg}_wo_sys_mes"
    return folder_path
