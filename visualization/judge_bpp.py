#!/usr/bin/env python3
"""
bpp_llm_neuro_judge.py 결과를 읽어 심판 모델 폴더별로 runNN·best_over_runs 평균 표를 stdout에 출력한다.

레이아웃(신규): logs/judge/<judge-model>/<judge-prompt>/run01/…
  (열 헤더 예: template/run01, explanation/run01)

레이아웃(레거시): logs/judge/<judge-model>/run01/… , logs/judge/<judge-model>/best_over_runs/…
  (동일 모델 폴더에 직접 runNN 이 있으면 그대로 지원)

레거시(judge 루트 바로 아래 runNN): 단일 표 `(legacy_direct_runs)`.
그 외 평탄 트리는 `(legacy_flat)` 단일 열.

사용 (저장소 루트에서):
  python visualization/judge_bpp.py
  python visualization/judge_bpp.py --judge-root logs/judge
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import DefaultDict, Dict, List, Optional, Tuple

_RUN_DIR_NAME_RE = re.compile(r"^run\d+$", re.IGNORECASE)
_RESERVED_JUDGE_CHILDREN = frozenset({"best_over_runs"})
_MISSING = "—"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_run_subdirs(parent: Path) -> List[Tuple[str, Path]]:
    """parent 직계 자식 중 runNN 이름만 (이름, 경로), 번호 순 정렬."""
    found: List[Tuple[str, Path]] = []
    if not parent.is_dir():
        return found
    for child in parent.iterdir():
        if child.is_dir() and _RUN_DIR_NAME_RE.match(child.name):
            found.append((child.name, child.resolve()))

    def _run_sort_key(item: Tuple[str, Path]) -> Tuple[int, str]:
        name = item[0]
        mo = re.search(r"\d+", name)
        return (int(mo.group()) if mo else 0, name.lower())

    found.sort(key=_run_sort_key)
    return found


def iter_model_subdirs(judge_root: Path) -> List[Tuple[str, Path]]:
    """judge_root 직계 자식 중 runNN·best_over_runs 가 아닌 디렉터리."""
    out: List[Tuple[str, Path]] = []
    if not judge_root.is_dir():
        return out
    for child in sorted(judge_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in _RESERVED_JUDGE_CHILDREN:
            continue
        if _RUN_DIR_NAME_RE.match(child.name):
            continue
        out.append((child.name, child.resolve()))
    return out


def iter_judge_jsonl_files(under: Path) -> List[Path]:
    """bpp_neuro_judge.jsonl 및 *_neuro_judge.jsonl 탐색."""
    if not under.is_dir():
        return []
    out: List[Path] = []
    for p in under.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name == "bpp_neuro_judge.jsonl" or name.endswith("_neuro_judge.jsonl"):
            out.append(p)
    return sorted(out)


def task_label_for_row(row: Dict, path: Path, scope_root: Path) -> str:
    """행 단위 태스크 라벨 (task_profile 우선, 없으면 경로에서 추론)."""
    tp = row.get("task_profile")
    if isinstance(tp, str) and tp.strip():
        return tp.strip()

    try:
        rel = path.relative_to(scope_root)
    except ValueError:
        return path.parent.name or path.stem

    parts = rel.parts[:-1]
    if parts and parts[0] == "1_5":
        parts = parts[1:]
    if len(parts) >= 2 and parts[-1] in ("trivia_n5", "trivia_n10"):
        return parts[-1]
    if parts:
        return "/".join(parts)
    return path.stem


def load_scores_by_task(
    scope_root: Path, *, emit_warnings: bool = True
) -> Tuple[Dict[str, List[int]], List[str]]:
    """
    scope_root 이하 트리에서 태스크 라벨 -> overall_score 목록.
    judge_parse_ok 가 True 이고 overall_score 가 정수인 행만 사용.
    """
    buckets: DefaultDict[str, List[int]] = defaultdict(list)
    warnings: List[str] = []

    for path in iter_judge_jsonl_files(scope_root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            warnings.append(f"skip read {path}: {e}")
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.append(f"{path}:{line_no} JSON error: {e}")
                continue
            if not row.get("judge_parse_ok", False):
                continue
            score = row.get("overall_score")
            if not isinstance(score, int):
                continue
            label = task_label_for_row(row, path, scope_root)
            buckets[label].append(score)

    if emit_warnings:
        for w in warnings[:20]:
            print(w, file=sys.stderr)
        if len(warnings) > 20:
            print(f"... 외 경고 {len(warnings) - 20}건", file=sys.stderr)

    return dict(buckets), warnings


def _task_sort_key(label: str) -> Tuple[int, str]:
    order = {
        "trivia_n5": 0,
        "trivia_n10": 1,
        "codenames": 2,
        "logic_grid": 3,
    }
    return (order.get(label, 100), label)


def _is_run_column_label(header: str) -> bool:
    """run 데이터 열인지: runNN 또는 …/runNN. best_over_runs 는 제외."""
    if header == "best_over_runs" or header.endswith("/best_over_runs"):
        return False
    tail = header.rsplit("/", 1)[-1]
    return bool(_RUN_DIR_NAME_RE.match(tail))


def _avg_source_labels(run_order: List[str]) -> List[str]:
    """Avg 계산에 쓸 열 이름: runNN(또는 prompt/runNN) 만. 없으면 best_over_runs 제외한 나머지, 그것도 없으면 전체."""
    nn = [h for h in run_order if _is_run_column_label(h)]
    if nn:
        return nn
    rest = [h for h in run_order if h != "best_over_runs"]
    return rest if rest else list(run_order)


def _gather_scores_per_run(
    scopes: List[Tuple[str, Path]],
) -> Tuple[Dict[str, Dict[str, List[int]]], List[str]]:
    """run_label -> task -> scores. run 순서는 scopes 순서 유지."""
    per_run: Dict[str, Dict[str, List[int]]] = {}
    run_order: List[str] = []
    for i, (run_label, scope_path) in enumerate(scopes):
        buckets, _ = load_scores_by_task(scope_path, emit_warnings=(i == 0))
        run_order.append(run_label)
        per_run[run_label] = buckets
    return per_run, run_order


def _print_table(
    judge_root: Path,
    per_run: Dict[str, Dict[str, List[int]]],
    run_order: List[str],
    *,
    judge_model_folder: str,
) -> bool:
    """태스크 × run 평균 표 출력. 데이터 없으면 False."""
    all_tasks: List[str] = sorted(
        {t for b in per_run.values() for t in b},
        key=_task_sort_key,
    )
    if not all_tasks:
        return False

    cells: Dict[str, Dict[str, Optional[float]]] = {}
    for task in all_tasks:
        cells[task] = {}
        for run_label in run_order:
            scores = per_run.get(run_label, {}).get(task)
            cells[task][run_label] = mean(scores) if scores else None

    sep = " | "
    run_col_w = max(len(_MISSING), len("99.99"), max((len(h) for h in run_order), default=0))
    task_col_w = max(len("Task"), max(len(t) for t in all_tasks), 21)

    def row_line(left: str, fields: List[str]) -> str:
        return left.ljust(task_col_w) + sep + sep.join(f.rjust(run_col_w) for f in fields)

    run_headers = [h.rjust(run_col_w) for h in run_order]
    header_line = row_line("Task", run_headers + ["Avg".rjust(run_col_w)])
    rule_len = len(header_line)
    rule_eq = "=" * rule_len
    rule_dash = "-" * rule_len

    print("Mean overall_score by task / run (1–5 scale)")
    print(f"Judge root: {judge_root}")
    print(f"Judge model folder: {judge_model_folder}")
    print(rule_eq)
    print(header_line)
    print(rule_dash)

    avg_cols = _avg_source_labels(run_order)
    for task in all_tasks:
        row_vals = [cells[task].get(r) for r in run_order]
        present = [cells[task][c] for c in avg_cols if cells[task].get(c) is not None]
        avg_s = f"{mean(present):.2f}" if present else _MISSING
        disp = [f"{v:.2f}" if v is not None else _MISSING for v in row_vals]
        print(row_line(task, disp + [avg_s]))

    print(rule_eq)
    return True


def _build_scopes_for_model_dir(model_path: Path) -> List[Tuple[str, Path]]:
    """
    스코프 (열 헤더, 점수를 읽을 디렉터리).

    - 레거시: <model>/runNN/…
    - 신규: <model>/<judge-prompt>/runNN/… (헤더는 template/run01 형식)
    - best_over_runs: 모델 직하 또는 <model>/<prompt>/best_over_runs/
    """
    scopes: List[Tuple[str, Path]] = []

    for run_name, run_path in iter_run_subdirs(model_path):
        scopes.append((run_name, run_path))

    for child in sorted(model_path.iterdir()):
        if not child.is_dir():
            continue
        if child.name in _RESERVED_JUDGE_CHILDREN:
            continue
        if _RUN_DIR_NAME_RE.match(child.name):
            continue
        for run_name, run_path in iter_run_subdirs(child):
            scopes.append((f"{child.name}/{run_name}", run_path))
        best_nested = (child / "best_over_runs").resolve()
        if best_nested.is_dir():
            scopes.append((f"{child.name}/best_over_runs", best_nested))

    best_path = (model_path / "best_over_runs").resolve()
    if best_path.is_dir():
        scopes.append(("best_over_runs", best_path))
    return scopes


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(
        description="logs/judge/<모델>[/프롬프트변형]/runNN·best_over_runs별 평균 표 출력"
    )
    p.add_argument(
        "--judge-root",
        type=str,
        default=str(root / "logs" / "judge"),
        help="심판 스테이징 루트 (기본: <repo>/logs/judge)",
    )
    args = p.parse_args()

    judge_root = Path(args.judge_root).resolve()
    if not judge_root.is_dir():
        print(f"경로가 없습니다: {judge_root}", file=sys.stderr)
        sys.exit(1)

    models = iter_model_subdirs(judge_root)
    blocks: List[Tuple[str, List[Tuple[str, Path]]]] = []

    if models:
        for model_name, model_path in models:
            scopes = _build_scopes_for_model_dir(model_path)
            if scopes:
                blocks.append((model_name, scopes))
    else:
        runs = iter_run_subdirs(judge_root)
        best_top = (judge_root / "best_over_runs").resolve()
        if runs:
            scopes = list(runs)
            if best_top.is_dir():
                scopes.append(("best_over_runs", best_top))
            blocks.append(("(legacy_direct_runs)", scopes))
        elif best_top.is_dir():
            blocks.append(("(legacy_direct_runs)", [("best_over_runs", best_top)]))
        else:
            blocks.append(("(legacy_flat)", [("(no runNN subdir)", judge_root)]))

    any_ok = False
    for bi, (model_folder, scopes) in enumerate(blocks):
        per_run, run_order = _gather_scores_per_run(scopes)
        if _print_table(judge_root, per_run, run_order, judge_model_folder=model_folder):
            any_ok = True
        if bi < len(blocks) - 1:
            print()

    if not any_ok:
        print(f"{judge_root} 에서 유효한 심판 행을 찾지 못했습니다.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
