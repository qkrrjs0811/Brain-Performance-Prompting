#!/usr/bin/env python3
"""
logs/<task>/<model_dir>/ 에 있는 accuracy 엑셀(accuracy_results_*.xlsx 등)을 읽어
모델(log 폴더)별로 Dataset × Method × Run 정확도(%)를 고정폭 ASCII 표로 출력합니다.
각 행 끝에 run 간 평균(Avg)과 표준편차(Std, run이 2개 이상일 때만)가 붙습니다.
토큰은 동일 File 이름의 jsonl에서 집계합니다. 각 줄의 raw_response(및 codenames용
raw_response_* , self_refine 중첩 dict)의 usage를 우선 합산하고, 없을 때만
usage_info/total_usage_info(누적·배치) 휴리스틱을 씁니다. 입력+출력은
prompt_tokens+completion_tokens(또는 Claude input/output, 또는 total_tokens).
맨 아래에는 Task/모델 요약 표가 이어집니다.

특정 로그 폴더(모델)와 Method만 보려면:
  python result_all.py --model gpt-4o-2024-08-06 --methods bpp
  python result_all.py --tasks logic_grid_puzzle --model gpt-4o --methods bpp,cot

특정 run 만 보려면 (`--runs`):
  python result_all.py --runs 1,2,3
  python result_all.py --runs run01,run03
  python result_all.py --runs default,1,2
(`--model` 은 폴더명 부분 문자열, 대소문자 무시. `--methods` 는 엑셀 Method 값과 정확히 일치.
 `--runs` 는 숫자(1,2,...), `runNN`(run01,run02,...), 또는 `default` 를 쉼표로 구분.)

토큰 표·요약 없이 정확도 표만:
  python result_all.py --accuracy-only
  python result_all.py --accuracy-only --no-overview

의존성: pandas + openpyxl (pip install openpyxl)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from statistics import stdev
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

NORMAL_TASK_DIRS = frozenset(
    {
        "trivia_creative_writing",
        "codenames_collaborative",
        "logic_grid_puzzle",
        "hf_numeric_math",
        "hf_multiple_choice",
    }
)

# 모델 폴더 안에서 집계에 쓸 엑셀 파일명 패턴 (__accuracy_results__ 포함)
XLSX_NAME_SUBSTR = "accuracy_results"


def extract_run_number(file_name: str) -> Optional[int]:
    """원본 jsonl 파일명(File 컬럼)에서 run 번호. 없으면 None (= default)."""
    if not isinstance(file_name, str):
        return None
    m = re.search(r"(?:^|[_-])run[_-]?(\d+)(?:[_-]|\.|$)", file_name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def run_label(run_id: Optional[int]) -> str:
    return "default" if run_id is None else f"run{run_id:02d}"


def data_key_from_file_cell(file_cell: str) -> Optional[str]:
    """엑셀 File 컬럼 값에서 __method- 앞부분(데이터 파일 식별)."""
    if not isinstance(file_cell, str) or "__method-" not in file_cell:
        return None
    return file_cell.split("__method-")[0]


def accuracy_to_percent(val: float) -> float:
    """엑셀 Accuracy: 0~1 비율이면 %로 변환, 이미 % 스케일이면 그대로."""
    if val <= 1.0 + 1e-9:
        return float(val) * 100.0
    return float(val)


def list_task_roots(logs_dir: str) -> List[str]:
    roots: List[str] = []
    if not os.path.isdir(logs_dir):
        return roots
    for name in sorted(os.listdir(logs_dir)):
        path = os.path.join(logs_dir, name)
        if not os.path.isdir(path):
            continue
        if name in NORMAL_TASK_DIRS:
            roots.append(path)
    return roots


def is_accuracy_xlsx(filename: str) -> bool:
    return filename.endswith(".xlsx") and XLSX_NAME_SUBSTR in filename.lower()


def _skip_jsonl_for_index(filename: str) -> bool:
    """배치 메타/요청 jsonl은 실제 응답 로그가 아니므로 인덱싱에서 제외."""
    if not filename.endswith(".jsonl"):
        return True
    bad = (
        "batch_input",
        "batch_meta",
        "batch_phase",
        "batch_anthropic_requests",
    )
    return any(s in filename for s in bad)


def build_jsonl_basename_index(search_root: str) -> Dict[str, str]:
    """basename -> 절대경로 (동명이면 나중에 발견된 경로가 남음)."""
    out: Dict[str, str] = {}
    if not os.path.isdir(search_root):
        return out
    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames.sort()
        for fn in sorted(filenames):
            if _skip_jsonl_for_index(fn):
                continue
            out[fn] = os.path.join(dirpath, fn)
    return out


def tokens_from_usage_dict(u: object) -> Optional[int]:
    """usage_info 등에서 입력+출력 토큰 합. 없으면 None."""
    if not isinstance(u, dict):
        return None
    pt = u.get("prompt_tokens")
    ct = u.get("completion_tokens")
    if pt is None and u.get("input_tokens") is not None:
        pt = u.get("input_tokens")
    if ct is None and u.get("output_tokens") is not None:
        ct = u.get("output_tokens")
    if pt is None and ct is None:
        tt = u.get("total_tokens")
        if tt is None:
            return None
        try:
            return int(tt)
        except (TypeError, ValueError):
            return None
    try:
        return int(pt or 0) + int(ct or 0)
    except (TypeError, ValueError):
        return None


_SKIP_RECURSE_KEYS = frozenset(
    {
        "task_data",
        "gpt_config",
        "claude_config",
        "open_model_config",
        "args",
    }
)


def _sum_usage_from_raw_response_list(responses: object) -> int:
    if not isinstance(responses, list):
        return 0
    s = 0
    for response in responses:
        if isinstance(response, dict) and "usage" in response:
            t = tokens_from_usage_dict(response["usage"])
            if t is not None:
                s += t
    return s


def line_tokens_from_raw_paths(rec: dict) -> int:
    """한 레코드 안의 API 호출 usage(prompt+completion) 합. raw_response·codenames·self_refine 중첩."""
    total = 0
    for key in ("raw_response", "raw_response_spymaster", "raw_response_guesser"):
        total += _sum_usage_from_raw_response_list(rec.get(key))
    for k, v in rec.items():
        if k in _SKIP_RECURSE_KEYS or k in ("raw_response", "raw_response_spymaster", "raw_response_guesser"):
            continue
        if isinstance(v, dict):
            total += line_tokens_from_raw_paths(v)
    return total


def file_total_tokens_from_jsonl(jsonl_path: str) -> Optional[int]:
    """
    jsonl 한 파일의 총 토큰(입력+출력).
    - 기본: 각 줄에서 raw_response(및 codenames 전용 필드, self_refine 중첩 dict)의 usage 합을
      줄마다 더해 파일 전체 합을 구한다 (배치·동기 공통, 누적 usage와 혼동 없음).
    - raw usage 가 한 줄도 없으면: total_usage_info / usage_info / aggregate_usage_info 만으로
      줄별 값을 모아, 비감소이면 마지막 값(세션 누적), 아니면 줄별 합(배치 추정).
    """
    series: List[int] = []
    from_raw_total = 0
    any_raw_line = False
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                raw_line = line_tokens_from_raw_paths(rec)
                if raw_line > 0:
                    from_raw_total += raw_line
                    any_raw_line = True
                    continue
                u = (
                    rec.get("total_usage_info")
                    or rec.get("aggregate_usage_info")
                    or rec.get("usage_info")
                )
                t = tokens_from_usage_dict(u) if u else None
                if t is not None:
                    series.append(t)
    except OSError as e:
        print(f"경고: jsonl 읽기 실패 {jsonl_path}: {e}")
        return None

    if any_raw_line:
        return from_raw_total

    if not series:
        return None

    if len(series) == 1:
        return series[0]

    if all(series[i] <= series[i + 1] for i in range(len(series) - 1)):
        return series[-1]
    return sum(series)


def read_accuracy_workbook(
    xlsx_path: str, jsonl_index: Dict[str, str]
) -> List[Tuple[str, str, float, Optional[int]]]:
    """
    엑셀에서 (run_label, column_label, accuracy_pct, total_tokens_opt) 행 목록 반환.
    Result Index == Mean 행만 사용.
    total_tokens_opt 는 File 이름에 대응하는 jsonl 이 있으면 그 파일 집계, 없으면 None.
    """
    out: List[Tuple[str, str, float, Optional[int]]] = []
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"경고: 엑셀 읽기 실패 {xlsx_path}: {e}")
        return out

    need = {"File", "Method", "Accuracy"}
    if not need.issubset(set(df.columns)):
        print(
            f"경고: 필수 컬럼 부족 {xlsx_path} "
            f"(필요: {sorted(need)}, 실제: {list(df.columns)})"
        )
        return out

    ridx = "Result Index"
    if ridx not in df.columns:
        print(f"경고: '{ridx}' 없음 → File·Method별 Accuracy 평균: {xlsx_path}")
        iter_df = df.groupby(["File", "Method"], dropna=False)["Accuracy"].mean().reset_index()
    else:
        mean_mask = df[ridx].astype(str).str.strip() == "Mean"
        df_mean = df[mean_mask]
        if df_mean.empty:
            print(f"경고: Mean 행 없음 → File·Method별 Accuracy 평균: {xlsx_path}")
            sub = df[df[ridx].astype(str).str.strip() != "Mean"]
            iter_df = sub.groupby(["File", "Method"], dropna=False)["Accuracy"].mean().reset_index()
        else:
            iter_df = df_mean

    for _, row in iter_df.iterrows():
        file_cell = row["File"]
        method = str(row["Method"])
        dkey = data_key_from_file_cell(str(file_cell))
        if not dkey:
            continue
        run_id = extract_run_number(str(file_cell))
        rl = run_label(run_id)
        col = f"{dkey} | {method}"
        try:
            pct = accuracy_to_percent(float(row["Accuracy"]))
        except (TypeError, ValueError):
            continue
        tok: Optional[int] = None
        jpath = jsonl_index.get(str(file_cell))
        if jpath and os.path.isfile(jpath):
            tok = file_total_tokens_from_jsonl(jpath)
        elif str(file_cell).endswith(".jsonl"):
            # 엑셀 File 이 basename 만 아닌 경우(거의 없음)
            bn = os.path.basename(str(file_cell))
            jpath2 = jsonl_index.get(bn)
            if jpath2 and os.path.isfile(jpath2):
                tok = file_total_tokens_from_jsonl(jpath2)
        out.append((rl, col, pct, tok))

    return out


def collect_results(
    logs_dir: str,
    task_filter: Optional[Set[str]],
    model_subdir_filter: Optional[str],
    run_filter: Optional[Set[str]] = None,
) -> Dict[Tuple[str, str], Dict[str, Dict[str, Tuple[float, Optional[int]]]]]:
    """
    (task_name, model_subdir) -> { run_label -> { column_label -> (accuracy_pct, total_tokens_opt) } }

    run_filter 가 주어지면 해당 run_label (예: "default", "run01") 들만 남깁니다.
    """
    flat: Dict[Tuple[str, str, str, str], Tuple[float, Optional[int]]] = {}

    for task_root in list_task_roots(logs_dir):
        task_name = os.path.basename(task_root)
        if task_filter is not None and task_name not in task_filter:
            continue

        for sub in sorted(os.listdir(task_root)):
            sub_path = os.path.join(task_root, sub)
            if not os.path.isdir(sub_path):
                continue
            if model_subdir_filter:
                # 로그 폴더명은 run.py 기준 대소문자 혼용 (예: Qwen-Qwen2.5-7B-Instruct_*)
                if model_subdir_filter.lower() not in sub.lower():
                    continue

            jsonl_index = build_jsonl_basename_index(sub_path)

            for _dirpath, _dirnames, filenames in os.walk(sub_path):
                for fn in filenames:
                    if not is_accuracy_xlsx(fn):
                        continue
                    xlsx_path = os.path.join(_dirpath, fn)
                    rows = read_accuracy_workbook(xlsx_path, jsonl_index)
                    for rl, col, pct, tok in rows:
                        if run_filter is not None and rl not in run_filter:
                            continue
                        key = (task_name, sub, rl, col)
                        flat[key] = (pct, tok)

    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Tuple[float, Optional[int]]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (task, sub, rl, col), pair in flat.items():
        grouped[(task, sub)][rl][col] = pair
    return {k: dict(v) for k, v in grouped.items()}


def normalize_run_token(token: str) -> Optional[str]:
    """사용자 입력(`1`, `01`, `run1`, `run01`, `default` 등) -> run_label 표준 형식."""
    s = token.strip().lower()
    if not s:
        return None
    if s == "default":
        return "default"
    m = re.match(r"^(?:run[_-]?)?(\d+)$", s)
    if m:
        return f"run{int(m.group(1)):02d}"
    return s


def run_sort_key(label: str) -> Tuple[int, str]:
    if label == "default":
        return (-1, label)
    m = re.match(r"run(\d+)", label, flags=re.IGNORECASE)
    if m:
        return (int(m.group(1)), label)
    return (9999, label)


def _shorten_dataset_key(dkey: str) -> str:
    s = str(dkey).strip()
    if s.endswith(".jsonl"):
        return s[: -len(".jsonl")]
    return s


def _parse_col_label(col: str) -> Optional[Tuple[Optional[str], str, str]]:
    """열 이름 -> (subset or None, data_key, method)."""
    parts = col.split(" | ")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    return None


def _format_num(val: Optional[float], decimals: int) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _print_ascii_table(
    headers: List[str],
    rows: List[List[str]],
    sep_char: str = "=",
    left_align_upto: int = 0,
) -> None:
    """고정폭 | 구분 표. left_align_upto개 열은 왼쪽, 나머지는 오른쪽 정렬."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    inner = sum(widths) + 3 * (len(headers) - 1)
    bar = sep_char * inner

    def line(cells: List[str], header_row: bool = False) -> str:
        parts = []
        for i, c in enumerate(cells):
            if header_row or i < left_align_upto:
                parts.append(c.ljust(widths[i]))
            else:
                parts.append(c.rjust(widths[i]))
        return " | ".join(parts)

    print(bar)
    print(line(headers, header_row=True))
    print("-" * inner)
    for row in rows:
        print(line(row, header_row=False))
    print(bar)


def _run_values_std(vals: List[Optional[float]]) -> Optional[float]:
    """유효 run 값만으로 표본 표준편차. 2개 미만이면 None."""
    valid = [v for v in vals if v is not None]
    return stdev(valid) if len(valid) >= 2 else None


def _build_section_table(
    block: Dict[str, Dict[str, Tuple[float, Optional[int]]]],
    decimals: int,
    method_filter: Optional[Set[str]] = None,
) -> Tuple[List[str], List[List[str]]]:
    """(Dataset, Method) 행 × run 열 + 평균열 + 표준편차열(run 간). method_filter 가 있으면 해당 method 행만."""
    runs = sorted(block.keys(), key=run_sort_key)
    # (subset, dkey, method) -> run -> pct
    cells: Dict[Tuple[Optional[str], str, str], Dict[str, float]] = defaultdict(dict)
    for run in runs:
        for col, cell in block[run].items():
            parsed = _parse_col_label(col)
            if parsed is None:
                continue
            subset, dkey, method = parsed
            pct = cell[0] if isinstance(cell, tuple) else float(cell)
            cells[(subset, dkey, method)][run] = pct

    row_keys_all = sorted(cells.keys(), key=lambda x: (x[0] or "", _shorten_dataset_key(x[1]), x[2]))
    row_keys = [
        rk
        for rk in row_keys_all
        if method_filter is None or rk[2] in method_filter
    ]
    headers = ["Dataset", "Method"] + runs + ["Avg", "Std"]
    rows_out: List[List[str]] = []
    for subset, dkey, method in row_keys:
        ds = _shorten_dataset_key(dkey)
        if subset:
            ds = f"{subset} / {ds}"
        vals: List[Optional[float]] = [cells[(subset, dkey, method)].get(r) for r in runs]
        valid = [v for v in vals if v is not None]
        avg: Optional[float] = sum(valid) / len(valid) if valid else None
        sd = _run_values_std(vals)
        row = (
            [ds, method]
            + [_format_num(v, decimals) for v in vals]
            + [_format_num(avg, decimals), _format_num(sd, decimals)]
        )
        rows_out.append(row)
    return headers, rows_out


def _format_tokens_cell(val: Optional[int]) -> str:
    if val is None:
        return "—"
    return f"{val:,}"


def _build_section_token_table(
    block: Dict[str, Dict[str, Tuple[float, Optional[int]]]],
    method_filter: Optional[Set[str]] = None,
) -> Tuple[List[str], List[List[str]]]:
    """정확도 표와 동일한 행·run 열, 값은 jsonl 집계 총 토큰(입력+출력)."""
    runs = sorted(block.keys(), key=run_sort_key)
    cells: Dict[Tuple[Optional[str], str, str], Dict[str, Optional[int]]] = defaultdict(dict)
    for run in runs:
        for col, cell in block[run].items():
            parsed = _parse_col_label(col)
            if parsed is None:
                continue
            subset, dkey, method = parsed
            tok = cell[1] if isinstance(cell, tuple) else None
            cells[(subset, dkey, method)][run] = tok

    row_keys_all = sorted(cells.keys(), key=lambda x: (x[0] or "", _shorten_dataset_key(x[1]), x[2]))
    row_keys = [
        rk
        for rk in row_keys_all
        if method_filter is None or rk[2] in method_filter
    ]
    headers = ["Dataset", "Method"] + runs + ["Avg", "Std"]
    rows_out: List[List[str]] = []
    for subset, dkey, method in row_keys:
        ds = _shorten_dataset_key(dkey)
        if subset:
            ds = f"{subset} / {ds}"
        vals_i: List[Optional[int]] = [cells[(subset, dkey, method)].get(r) for r in runs]
        vals_f: List[Optional[float]] = [float(v) if v is not None else None for v in vals_i]
        valid = [v for v in vals_i if v is not None]
        avg: Optional[float] = (sum(valid) / len(valid)) if valid else None
        sd = _run_values_std(vals_f)
        row = (
            [ds, method]
            + [_format_tokens_cell(v) for v in vals_i]
            + [
                _format_tokens_cell(int(round(avg))) if avg is not None else "—",
                _format_num(sd, 1) if sd is not None else "—",
            ]
        )
        rows_out.append(row)
    return headers, rows_out


def _overview_row(
    task: str,
    model_subdir: str,
    block: Dict[str, Dict[str, Tuple[float, Optional[int]]]],
    decimals: int,
    method_filter: Optional[Set[str]] = None,
) -> List[str]:
    """요약 표 한 행: task, model, Mean Acc(%), N runs, N settings."""
    runs = sorted(block.keys(), key=run_sort_key)
    all_pct: List[float] = []
    for run in runs:
        for col, cell in block[run].items():
            parsed = _parse_col_label(col)
            if parsed is None:
                continue
            _subset, _dkey, method = parsed
            if method_filter is not None and method not in method_filter:
                continue
            pct = cell[0] if isinstance(cell, tuple) else float(cell)
            all_pct.append(pct)
    setting_keys: Set[Tuple[Optional[str], str, str]] = set()
    for run in runs:
        for c in block[run].keys():
            parsed = _parse_col_label(c)
            if parsed is None:
                continue
            if method_filter is not None and parsed[2] not in method_filter:
                continue
            setting_keys.add(parsed)
    n_settings = len(setting_keys)
    mean_all = sum(all_pct) / len(all_pct) if all_pct else None
    return [
        task,
        model_subdir,
        _format_num(mean_all, decimals),
        str(len(runs)),
        str(n_settings),
    ]


def print_tables(
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Tuple[float, Optional[int]]]]],
    float_format: int,
    method_filter: Optional[Set[str]] = None,
    *,
    accuracy_only: bool = False,
    no_overview: bool = False,
) -> None:
    if not grouped:
        print("집계할 accuracy 엑셀 결과가 없습니다.")
        return

    overview_headers = ["Task", "Model (log dir)", "Mean Acc (%)", "#runs", "#dataset×method"]
    overview_rows: List[List[str]] = []

    for (task, model_subdir) in sorted(grouped.keys()):
        block = grouped[(task, model_subdir)]
        headers, rows = _build_section_table(block, float_format, method_filter)

        print()
        print("Accuracy by dataset / method / run (%)")
        print(f"Task: {task}  |  Log folder: {model_subdir}")
        if not rows:
            print("(표시할 행 없음: --methods 필터와 일치하는 Method 가 없습니다.)")
            continue
        _print_ascii_table(headers, rows, sep_char="=", left_align_upto=2)

        if not accuracy_only:
            tok_headers, tok_rows = _build_section_token_table(block, method_filter)
            print()
            print("Total tokens by dataset / method / run (prompt + completion, from jsonl)")
            _print_ascii_table(tok_headers, tok_rows, sep_char="=", left_align_upto=2)

        overview_rows.append(
            _overview_row(task, model_subdir, block, float_format, method_filter)
        )

    if not no_overview:
        print()
        print("Overview (all sections)")
        _print_ascii_table(overview_headers, overview_rows, sep_char="=", left_align_upto=2)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "logs 내 task별 모델 폴더의 accuracy_results*.xlsx 만 읽어 "
            "run xx (데이터셋|method) 정확도(%) 표를 출력합니다."
        )
    )
    p.add_argument("--logs-dir", type=str, default="logs", help="로그 루트 (기본: logs)")
    p.add_argument(
        "--tasks",
        type=str,
        default="",
        help="쉼표로 구분한 task만 (예: trivia_creative_writing,codenames_collaborative)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="로그 하위 폴더명에 포함될 문자열 (대소문자 무시, 예: gpt-4o-2024-08-06)",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="",
        help="엑셀 Method 컬럼과 정확히 일치하는 이름만 (쉼표 구분, 예: bpp,cot,standard)",
    )
    p.add_argument(
        "--runs",
        type=str,
        default="",
        help=(
            "표시할 run 만 (쉼표 구분). "
            "허용 형식: 숫자(1,2,...), runNN(run01,run02,...), default. "
            "예: --runs 1,2,3  /  --runs run01,run03  /  --runs default,1"
        ),
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="표 소수 자릿수 (기본 2)",
    )
    p.add_argument(
        "--accuracy-only",
        action="store_true",
        help="정확도(%%) 표만 출력 (run별 jsonl 토큰 집계 표 생략)",
    )
    p.add_argument(
        "--no-overview",
        action="store_true",
        help="맨 아래 Task/모델 요약(Overview) 표 생략",
    )
    args = p.parse_args()

    logs_dir = os.path.abspath(args.logs_dir)
    task_filter: Optional[Set[str]] = None
    if args.tasks.strip():
        task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()}

    model_filter = args.model.strip() or None

    method_filter: Optional[Set[str]] = None
    if args.methods.strip():
        method_filter = {m.strip() for m in args.methods.split(",") if m.strip()}

    run_filter: Optional[Set[str]] = None
    if args.runs.strip():
        run_filter = set()
        for tok in args.runs.split(","):
            norm = normalize_run_token(tok)
            if norm:
                run_filter.add(norm)
        if not run_filter:
            run_filter = None

    grouped = collect_results(logs_dir, task_filter, model_filter, run_filter)
    print_tables(
        grouped,
        args.decimals,
        method_filter,
        accuracy_only=args.accuracy_only,
        no_overview=args.no_overview,
    )


if __name__ == "__main__":
    main()
