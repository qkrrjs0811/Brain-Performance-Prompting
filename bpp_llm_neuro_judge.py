#!/usr/bin/env python3
"""
BPP 실행 로그(jsonl)에서 뇌 영역 persona 대화를 읽어, Claude(Haiku 등)로
뇌 기능 정합성 1~5점 LLM-as-a-Judge 평가를 수행한다.

필요: ANTHROPIC_API_KEY, pip install anthropic
  (심판 모델 하이퍼파라미터는 본 파일 내장 `claude_configs` 만 사용; configs.py 와 무관)

심판 호출은 항상 Anthropic Message Batches API 로만 수행한다(할인 요금 구간).
실행 종료 시 표준 출력에 배치 응답 usage 기준 토큰 합계·USD 추정 비용을 출력한다.

사용 예 (인자 없음 — 4개 태스크 BPP 로그 자동 탐색 후 미처리 건만 배치 제출):
  python bpp_llm_neuro_judge.py

  단일 로그만 (기본: logs/judge/<심판설정키>/<judge-prompt>/runNN/trivia_creative_writing/…):
  python bpp_llm_neuro_judge.py \\
    --log-file logs/trivia_creative_writing/.../__method-bpp_model-....jsonl

  결과를 한 파일에만 쓰기:
  python bpp_llm_neuro_judge.py --log-file .../run.jsonl --output logs/judge/scores.jsonl

  각 태스크 로그마다 idx 0,1,2 만 시험(포함 상한):
  python bpp_llm_neuro_judge.py --upto-idx 2

  뇌영역 참조 루브릭이 포함된 심판 프롬프트 사용:
  python bpp_llm_neuro_judge.py --judge-prompt explanation

배치 자동 태스크: trivia n=5 / trivia n=10 / codenames_collaborative / logic_grid_puzzle
(`logs/` 아래에서 BPP 로그만: 생성 모델은 **gpt-4o**(snapshot `gpt-4o-2024-08-06`, mini 제외),
추가 출력 노트 **run01**(`_run01__` in filename); 동일 조건 후보가 여러 개면 수정 시각 최신 1개)

기본 결과 형식: JSONL(한 줄에 하나의 UTF-8 JSON 객체).
단일 JSON 배열 파일이 필요하면 `--output-format json` 을 쓴다.

기본 저장 위치: `logs/judge/<--judge-model 설정키>/<--judge-prompt>/runNN/<태스크폴더>/…`
  (예: `logs/judge/claude-haiku-4-5/template/run01/trivia_creative_writing/trivia_n5/bpp_neuro_judge.jsonl`).
  `run01`, `run02` … 는 자동 할당: **해당 run 폴더만** 스캔해 (source_log_file, idx) 가 모두 채워져 있으면
  다음 번호 run 폴더를 쓴다. 미완료면 같은 run 폴더에서 이어쓴다.
  `--output` 에 디렉터리를 주면 그 아래에 `<judge-model>/<judge-prompt>/runNN/` 을 붙인 뒤 동일 규칙으로 만든다.
  `--output` 이 `.jsonl`/`.json` 파일 경로면 run 접두 없이 예전처럼 단일 파일에만 기록한다.

이어하기: 선택된 run 폴더(또는 단일 파일 모드일 때 그 파일) 안의 기존 결과만 보고
  (source_log_file, idx) 가 이미 있으면 건너뜀.

평가 대상: method 가 `bpp` 이거나, 파일명에 `__method-bpp_model-` 가 있는데 JSON의 method 만 `bpp`+숫자로 어긋난 행.
  코드네임 로그는 `raw_response` 대신 `raw_response_spymaster`·`raw_response_guesser` 를 이어 붙여 심판한다.
  그 밖의 파일명·방법론은 심판하지 않음.
심판 프롬프트는 영어로 고정되어 있다. `--judge-prompt template`(기본) 또는 `explanation`(뇌영역 참조 루브릭 포함)으로 선택.
심판 모델은 JSON 없이 전체 대화에 대한 단일 점수(정수 1–5)만 출력한다.
결과 각 레코드에는 메타데이터와 함께 `overall_score`(정수 1–5)만 포함된다(페르소나별 점수·score_payload 없음).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from anthropic_batch import claude_config_to_message_params, run_claude_batches_and_collect
from openai_batch import BatchLineResult

# 심판 전용: 항상 아래 딕셔너리만 사용한다 (configs.claude_configs 는 참조하지 않음).
claude_configs: Dict[str, Dict[str, Any]] = {
    "claude-haiku-4-5": {
        "model": "claude-haiku-4-5",
        "temperature": 0.0,
        "max_tokens": 30,
    },
    "claude-sonnet-4-6": {
        "model": "claude-sonnet-4-6",
        "temperature": 0.0,
        "max_tokens": 30,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    },
    "claude-opus-4-7": {
        "model": "claude-opus-4-7",
        "temperature": 0.0,
        "max_tokens": 30,
    },
}

_FINAL_ANSWER_SPLIT = re.compile(r"\bfinal\s+answer\s*", re.IGNORECASE | re.DOTALL)

# (결과 필드 task_profile, logs 하위 디렉터리, 파일명에 포함되어야 할 접두 — bpp 단일 방법만)
DEFAULT_BATCH_TASK_SPECS: List[Tuple[str, str, str]] = [
    ("trivia_n5", "trivia_creative_writing", "trivia_creative_writing_100_n_5.jsonl__method-bpp_model-"),
    ("trivia_n10", "trivia_creative_writing", "trivia_creative_writing_100_n_10.jsonl__method-bpp_model-"),
    ("codenames", "codenames_collaborative", "codenames_50.jsonl__method-bpp_model-"),
    ("logic_grid", "logic_grid_puzzle", "logic_grid_puzzle_200.jsonl__method-bpp_model-"),
]

# task_profile → logs/ 바로 아래 태스크 폴더명 (judge/<모델>/<judge-prompt>/<run>/<이 이름>/… 에 대응)
_PROFILE_TO_LOG_SUBDIR: Dict[str, str] = {
    profile: subdir for profile, subdir, _ in DEFAULT_BATCH_TASK_SPECS
}

DEFAULT_JUDGE_OUTPUT_ROOT = os.path.join("logs", "judge")

# 배치 심판용 소스 로그: run.py 기본 스냅샷·추가 노트 run01 과 맞춤
_BATCH_SOURCE_MODEL_RE = re.compile(r"model-gpt-4o-(?!mini)")
_BATCH_RUN_MARKER = "_run01__"


def find_latest_bpp_log_under_task(logs_root: str, task_subdir: str, filename_needle: str) -> Optional[str]:
    """task_subdir 아래에서 filename_needle + gpt-4o(비-mini) + run01 파일명 조건을 만족하는 jsonl 중 최신 mtime."""
    base = os.path.join(logs_root, task_subdir)
    if not os.path.isdir(base):
        return None
    best: Optional[str] = None
    best_mtime = -1.0
    for root, _, files in os.walk(base):
        for name in files:
            if not name.endswith(".jsonl"):
                continue
            if filename_needle not in name:
                continue
            if _BATCH_RUN_MARKER not in name:
                continue
            if not _BATCH_SOURCE_MODEL_RE.search(name):
                continue
            path = os.path.join(root, name)
            try:
                m = os.path.getmtime(path)
            except OSError:
                continue
            if m > best_mtime:
                best_mtime = m
                best = path
    return best


def resolve_default_batch_logs(logs_root: str) -> List[Tuple[str, str]]:
    """[(task_profile, abs_log_path), ...] — 하나라도 없으면 SystemExit."""
    logs_root = os.path.abspath(logs_root)
    resolved: List[Tuple[str, str]] = []
    missing: List[str] = []
    for profile, subdir, needle in DEFAULT_BATCH_TASK_SPECS:
        p = find_latest_bpp_log_under_task(logs_root, subdir, needle)
        if not p:
            missing.append(
                f"{profile}: logs/{subdir}/...*{needle}*gpt-4o(비-mini)*{_BATCH_RUN_MARKER}*"
            )
        else:
            resolved.append((profile, os.path.abspath(p)))
    if missing:
        msg = "배치 모드에서 다음 BPP 로그를 찾지 못했습니다:\n  " + "\n  ".join(missing)
        raise SystemExit(msg)
    return resolved


def make_anthropic_sdk_client() -> Any:
    """Message Batches 제출용 Anthropic 클라이언트."""
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit(
            "anthropic 패키지가 필요합니다. pip install anthropic 후 다시 실행하세요."
        ) from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY 가 설정되어 있어야 합니다. 환경 변수를 확인하세요."
        )
    return anthropic.Anthropic(api_key=api_key)


def _assistant_text_from_completion_dict(rr: Dict[str, Any]) -> Optional[str]:
    """OpenAI 형식 completion dict → 어시스턴트 텍스트."""
    choices = rr.get("choices")
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        if parts:
            return "".join(parts)
    return None


def _assistant_texts_from_raw_field(raw: Any) -> List[str]:
    """raw_response 계열 필드(list[completion] | completion)에서 문자열 목록."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        t = _assistant_text_from_completion_dict(raw)
        return [t] if t else []
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            if isinstance(item, dict):
                t = _assistant_text_from_completion_dict(item)
                if t:
                    out.append(t)
        return out
    return []


def assistant_content_from_record(record: Dict[str, Any]) -> Optional[str]:
    """run.py 로그: raw_response 또는 코드네임 전용 raw_response_spymaster/guesser."""
    rr = record.get("raw_response")
    if rr is not None:
        if isinstance(rr, list):
            rr = rr[0] if rr else None
        if isinstance(rr, dict):
            t = _assistant_text_from_completion_dict(rr)
            if t:
                return t

    sp = record.get("raw_response_spymaster")
    gg = record.get("raw_response_guesser")
    if sp is None and gg is None:
        return None

    sections: List[str] = []
    sp_texts = _assistant_texts_from_raw_field(sp)
    if sp_texts:
        sections.append(
            "=== Codenames (spymaster phase) ===\n\n" + "\n\n---\n\n".join(sp_texts)
        )
    gg_texts = _assistant_texts_from_raw_field(gg)
    if gg_texts:
        sections.append(
            "=== Codenames (guesser phase) ===\n\n" + "\n\n---\n\n".join(gg_texts)
        )
    if sections:
        return "\n\n".join(sections)
    return None


def extract_collaboration_for_judge(full_text: str, *, strip_after_final_answer: bool = True) -> str:
    """
    BPP 협업 장을 심판용으로 준비한다.
    기본: 'Final answer:' 이후(최종 스토리/답안 본문)는 제외해 persona 발화 평가에 집중한다.
    """
    if not full_text:
        return ""
    t = full_text.strip()
    if strip_after_final_answer:
        parts = _FINAL_ANSWER_SPLIT.split(t, maxsplit=1)
        t = parts[0].strip()
    return t


JUDGE_PROMPT_TEMPLATE_EN = """You are a cognitive neuroscience evaluator.

Read the dialogue below between brain-region personas (and any assistant turns). Produce ONE holistic score from 1 to 5 for how well the brain-region personas' contributions collectively align with real cognitive/neuroscientific roles implied by their assigned regions. Weight all speaking brain-region personas together into a single judgment.

Scoring rubric (same scale for the single aggregate score):
1 = Almost no alignment with plausible neural/cognitive roles for those personas
2 = Weakly related but largely inaccurate or misleading overall
3 = Somewhat related but generic, vague, or incomplete overall
4 = Largely consistent with real functions overall
5 = Strongly consistent, specific, and appropriate overall

Do not score fluency, style, or role-play naturalness. Judge only neuroscientific plausibility of what the brain-region personas say.

Output format: output exactly one integer in {{1,2,3,4,5}} with no other characters, OR a single line of the form `overall: <digit>`.

Dialogue:
{dialogue}
"""


JUDGE_PROMPT_EXPLANATION = """You are a cognitive neuroscience evaluator.

Read the dialogue below between brain-region personas and any assistant turns. Produce ONE holistic score from 1 to 5 for how well the brain-region personas' contributions collectively align with the cognitive/neuroscientific functions of their assigned brain regions.

Use the region descriptions below as the primary rubric. Score the functional match of each persona's utterance, not whether it merely uses neuroscientific wording or sounds plausible as role-play.

Reference functions of brain regions:
- Frontal Lobe: Responsible for goal setting, strategic organization, prioritization, planning, executive control, and monitoring the overall progress of problem solving.
- Temporal Lobe: Responsible for semantic knowledge, language comprehension, factual recall, and conceptual association.
- Hippocampus: Responsible for binding context and relationships, maintaining sequence, connecting scenes or events, and supporting coherent relational memory.
- Occipital Lobe: Responsible for visual representation, scene construction, visual imagery, and image-based checking.
- Limbic System: Responsible for emotion, motivation, value relevance, affective salience, and emotion-memory integration.
- Amygdala: Responsible for rapid evaluation of threat, reward, emotional salience, and socially relevant affective cues.
- Dorsolateral Prefrontal Cortex: Responsible for maintaining rules and intermediate states in working memory, goal-directed attention, task switching, cognitive control, and step-by-step problem solving.
- Superior Parietal Lobule: Responsible for visuospatial attention, position comparison, coordinate transformation, spatial layout checking, and spatial relation integration.
- Parietal Lobe: Responsible for spatial attention, tracking external relations, adaptive visual processing, multisensory integration, and relation/state updating.
- Primary Visual Cortex: Responsible for representing early visual features and precise layouts, and for supporting top-down visual checking.

Scoring rubric:
1 = Almost no alignment with the reference functions of the assigned brain regions. The personas' contributions are arbitrary, misleading, or unrelated to their assigned regions.
2 = Weak alignment. Some utterances are loosely related, but most are generic, inaccurate, or do not clearly match the assigned brain-region functions.
3 = Moderate alignment. The contributions are somewhat related to the assigned regions, but are often generic, shallow, or only partially matched to the reference functions.
4 = Strong alignment. Most persona contributions are consistent with the reference functions of their assigned brain regions and are functionally appropriate for the task.
5 = Very strong alignment. The persona contributions are specific, functionally appropriate, and clearly differentiated according to the reference functions of their assigned brain regions.

Do not score fluency, style, or role-play naturalness. Judge only neuroscientific plausibility of what the brain-region personas say.

Output format: output exactly one integer in {{1,2,3,4,5}} with no other characters, OR a single line of the form `overall: <digit>`.

Dialogue:
{dialogue}
"""


JUDGE_PROMPT_VARIANTS: Dict[str, str] = {
    "template": JUDGE_PROMPT_TEMPLATE_EN,
    "explanation": JUDGE_PROMPT_EXPLANATION,
}


def _strip_json_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


_LINE_OVERALL = re.compile(r"^\s*overall\s*:\s*([1-5])\s*$", re.IGNORECASE)


def parse_judge_overall_score(text: str) -> int:
    """모델 응답에서 단일 정수 점수 1–5만 추출한다."""
    s = _strip_json_fences(text).strip()
    if not s:
        raise ValueError("심판 응답이 비어 있음")

    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        mo_o = _LINE_OVERALL.match(line)
        if mo_o:
            return int(mo_o.group(1))
        if re.fullmatch(r"[1-5]", line):
            return int(line)

    compact = "".join(s.split())
    if re.fullmatch(r"[1-5]", compact):
        return int(compact)

    raise ValueError("overall 점수(1–5 한 자리)를 파싱하지 못함")


def load_results_json_array(path: str) -> List[Dict[str, Any]]:
    """단일 JSON 파일에서 결과 객체 배열을 읽는다."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"결과 JSON 을 파싱할 수 없음 ({path}): {e}") from e
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [x for x in data["results"] if isinstance(x, dict)]
    raise SystemExit(
        f"결과 JSON 은 배열이거나 {{\"results\": [...]}} 형태여야 함: {path}"
    )


def _result_row_key(row: Dict[str, Any]) -> Tuple[str, int]:
    """이어하기·upsert 용 (source_log_file, idx)."""
    src = row.get("source_log_file")
    src_n = os.path.normpath(str(src)) if src else ""
    ix = row.get("idx")
    return (src_n, int(ix) if ix is not None else -1)


def upsert_result(rows: List[Dict[str, Any]], row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """동일 (source_log_file, idx) 이면 교체, 없으면 추가 후 정렬."""
    key = _result_row_key(row)
    out: List[Dict[str, Any]] = []
    replaced = False
    for r in rows:
        if _result_row_key(r) == key:
            out.append(row)
            replaced = True
        else:
            out.append(r)
    if not replaced:
        out.append(row)
    out.sort(key=_result_row_key)
    return out


def save_results_json_array(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def loaded_done_keys(path: str, output_format: str) -> Set[Tuple[str, int]]:
    """기존 출력에서 처리된 (source_log_file, idx) 모음."""
    done: Set[Tuple[str, int]] = set()
    if not path or not os.path.isfile(path):
        return done

    def _consume_row(o: Dict[str, Any]) -> None:
        if "idx" not in o:
            return
        if o.get("source_log_file"):
            done.add(
                (os.path.normpath(str(o["source_log_file"])), int(o["idx"]))
            )

    if output_format == "json":
        for o in load_results_json_array(path):
            _consume_row(o)
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            _consume_row(o)
    return done


def loaded_done_keys_under_judge_root(judge_root: str, output_format: str) -> Set[Tuple[str, int]]:
    """지정 심판 출력 디렉터리 트리 아래 모든 심판 결과에서 (source_log_file, idx) 합집합."""
    done: Set[Tuple[str, int]] = set()
    if not judge_root or not os.path.isdir(judge_root):
        return done
    suf = ".jsonl" if output_format == "jsonl" else ".json"
    for root, _, files in os.walk(judge_root):
        for name in files:
            if not name.endswith(suf):
                continue
            done |= loaded_done_keys(os.path.join(root, name), output_format)
    return done


def infer_task_subdir_from_log_path(log_abs: str, logs_root: str) -> Optional[str]:
    """logs/<태스크폴더>/... 경로에서 태스크 폴더명 추출."""
    log_abs = os.path.abspath(log_abs)
    logs_abs = os.path.abspath(logs_root)
    prefix = logs_abs + os.sep
    if not log_abs.startswith(prefix):
        return None
    rel = log_abs[len(prefix) :]
    first = rel.split(os.sep, 1)[0]
    return first if first and first != ".." else None


def judge_output_path_for_item(
    item: JudgeWorkItem,
    judge_root: str,
    fmt: str,
    *,
    single_file_path: Optional[str],
    single_log_stem: Optional[str],
    logs_root: str,
) -> str:
    """
    기본: <judge_root>/<태스크폴더>/…/bpp_neuro_judge.jsonl (judge_root 에 보통 <judge-prompt>/runNN 포함)
    trivia_n5/n10: …/trivia_creative_writing/<profile>/bpp_neuro_judge.jsonl
    """
    if single_file_path:
        return single_file_path

    ext = ".jsonl" if fmt == "jsonl" else ".json"
    base = os.path.abspath(judge_root)
    prof = item.task_profile

    if prof in ("trivia_n5", "trivia_n10"):
        task_subdir = _PROFILE_TO_LOG_SUBDIR[prof]
        rel = os.path.join(task_subdir, prof, f"bpp_neuro_judge{ext}")
        return os.path.join(base, rel)

    if prof and prof in _PROFILE_TO_LOG_SUBDIR:
        task_subdir = _PROFILE_TO_LOG_SUBDIR[prof]
        rel = os.path.join(task_subdir, f"bpp_neuro_judge{ext}")
        return os.path.join(base, rel)

    task_subdir = infer_task_subdir_from_log_path(item.log_abs, logs_root)
    if not task_subdir:
        task_subdir = (
            re.sub(r"[^\w\-]+", "_", str(item.task or "unknown_task")).strip("_")
            or "unknown_task"
        )
    stem = (single_log_stem or "").strip() or "bpp_neuro_judge"
    if stem.endswith(ext):
        fname = stem
    else:
        fname = f"{stem}_neuro_judge{ext}"
    return os.path.join(base, task_subdir, fname)


def iter_pending_indices_for_log(
    log_abs: str,
    start: int,
    end: int,
    done_keys: Set[Tuple[str, int]],
) -> Iterable[int]:
    log_abs = os.path.normpath(os.path.abspath(log_abs))
    for i in range(start, end):
        if (log_abs, i) not in done_keys:
            yield i


_CANONICAL_BPP_FILE_MARKER = "__method-bpp_model-"


def log_filename_is_canonical_bpp(log_path: str) -> bool:
    """basename 에 canonical BPP 마커가 있는지."""
    return _CANONICAL_BPP_FILE_MARKER in os.path.basename(log_path)


def bpp_method_ok(method: Optional[str], log_path: Optional[str] = None) -> bool:
    if not method:
        return False
    m = method.strip().lower()
    if m == "bpp":
        return True
    if log_path and log_filename_is_canonical_bpp(log_path) and re.fullmatch(r"bpp\d+", m):
        return True
    return False


@dataclass(frozen=True)
class JudgeWorkItem:
    """심판 1건(로그 한 줄). Message Batches 의 custom_id 는 API 상 최대 64자."""

    log_abs: str
    idx: int
    task_profile: Optional[str]
    method: Any
    task: Any
    dialogue: str

    @property
    def custom_id(self) -> str:
        key = f"{os.path.normpath(self.log_abs)}\0{self.idx}"
        # Anthropic: custom_id 길이 ≤ 64 — SHA256 hex 만으로 정확히 64자
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def iter_judge_work_items_for_log(
    *,
    log_path: str,
    start_idx: int,
    end_idx: int,
    include_final_answer_section: bool,
    allow_non_bpp_method: bool,
    done_keys: Set[Tuple[str, int]],
    task_profile: Optional[str],
) -> Iterator[JudgeWorkItem]:
    """평가 대상으로 확정된 로그 행만 JudgeWorkItem 으로 낸다 (스킵 시 로그 출력)."""
    log_abs = os.path.abspath(log_path)
    if not os.path.isfile(log_path):
        raise SystemExit(f"log 가 없음: {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    n = len(lines)
    end_local = end_idx if end_idx >= 0 else n
    start_local = max(0, start_idx)
    end_local = min(end_local, n)

    if start_local >= end_local:
        print(
            f"[warn] 유효한 구간 없음: {log_path} start={start_local} end={end_local}",
            flush=True,
        )
        return

    prefix = f"[{task_profile}] " if task_profile else ""

    for idx in iter_pending_indices_for_log(log_abs, start_local, end_local, done_keys):
        line = lines[idx].strip()
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"{prefix}[skip idx={idx}] JSON 파싱 실패: {e}", flush=True)
            continue

        method = rec.get("method")
        if not allow_non_bpp_method and not bpp_method_ok(method, log_path=log_abs):
            hint = ""
            if isinstance(method, str) and re.fullmatch(r"bpp\d+", method.strip().lower()):
                if not log_filename_is_canonical_bpp(log_abs):
                    hint = (
                        " — bpp+N 은 파일명에 "
                        f"{_CANONICAL_BPP_FILE_MARKER!r} 가 있을 때만 허용"
                    )
            print(
                f"{prefix}[skip idx={idx}] method 가 bpp 가 아님 (현재: {method!r}){hint}",
                flush=True,
            )
            continue

        raw_text = assistant_content_from_record(rec)
        if not raw_text:
            print(f"{prefix}[skip idx={idx}] raw_response 에서 텍스트 없음", flush=True)
            continue

        dialogue = extract_collaboration_for_judge(
            raw_text, strip_after_final_answer=not include_final_answer_section
        )
        if not dialogue.strip():
            print(f"{prefix}[skip idx={idx}] 협업 대화 비어 있음", flush=True)
            continue

        yield JudgeWorkItem(
            log_abs=log_abs,
            idx=idx,
            task_profile=task_profile,
            method=method,
            task=rec.get("task"),
            dialogue=dialogue,
        )


def count_pending_judge_items(
    *,
    jobs: List[Tuple[Optional[str], str]],
    done_keys: Set[Tuple[str, int]],
    start_idx: int,
    end_idx: int,
    include_final_answer_section: bool,
    allow_non_bpp_method: bool,
) -> int:
    """done_keys 를 제외한 뒤 실제 심판 요청으로 나갈 JudgeWorkItem 개수."""
    n = 0
    for task_profile, log_path in jobs:
        for _ in iter_judge_work_items_for_log(
            log_path=log_path,
            start_idx=start_idx,
            end_idx=end_idx,
            include_final_answer_section=include_final_answer_section,
            allow_non_bpp_method=allow_non_bpp_method,
            done_keys=done_keys,
            task_profile=task_profile,
        ):
            n += 1
    return n


def resolve_judge_run_directory_and_done_keys(
    staging_base: str,
    output_format: str,
    jobs: List[Tuple[Optional[str], str]],
    *,
    start_idx: int,
    end_idx: int,
    include_final_answer_section: bool,
    allow_non_bpp_method: bool,
    max_run_index: int = 9999,
) -> Tuple[str, Set[Tuple[str, int]]]:
    """
    staging_base(…/<judge-model>/<judge-prompt>/) 아래 run01, run02, … 중 하나를 고른다.

    해당 run 디렉터리만 loaded_done_keys 로 스캔했을 때
    아직 제출할 심판 건이 남아 있으면 그 디렉터리를 사용한다(미완료면 이어쓰기).
    이전 run 들은 모두 완료(pending==0)인데 아직 해야 할 소스 행이 남아 있으면
    다음 번호 run 폴더를 연다(예: run02 가 가득 차 있으면 run03).
    """
    base = os.path.abspath(staging_base)
    os.makedirs(base, exist_ok=True)
    n = 1
    while n <= max_run_index:
        run_dir = os.path.join(base, f"run{n:02d}")
        done = loaded_done_keys_under_judge_root(run_dir, output_format)
        pending_here = count_pending_judge_items(
            jobs=jobs,
            done_keys=done,
            start_idx=start_idx,
            end_idx=end_idx,
            include_final_answer_section=include_final_answer_section,
            allow_non_bpp_method=allow_non_bpp_method,
        )
        if pending_here > 0:
            return run_dir, done
        n += 1
    raise SystemExit(
        f"심판 출력용 runNN 폴더를 run{max_run_index:02d} 까지 모두 확인했으나 "
        "모든 run 에서 처리할 항목이 없습니다(전부 완료). "
        f"staging_base={base}"
    )


def _method_label_for_judge_output(method: Any) -> Any:
    """소스가 bpp4 로 기록돼 있어도 심판 결과 method 는 bpp 로 통일."""
    if isinstance(method, str) and method.strip().lower() == "bpp4":
        return "bpp"
    return method


def _judge_text_from_batch_body(body: Optional[Dict[str, Any]]) -> str:
    if not body:
        return ""
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    c = msg.get("content")
    return c if isinstance(c, str) else ""


def _tokens_from_judge_openai_shaped_body(body: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    """claude_succeeded_message_to_openai_shaped_body 의 usage → (prompt, completion)."""
    if not body:
        return 0, 0
    u = body.get("usage") or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def estimate_message_batch_judge_cost_usd(
    pricing_model_id: str, prompt_tokens: int, completion_tokens: int
) -> Dict[str, Any]:
    """
    Message Batches 단가 기준 추정 비용(USD). models.claude_message_batch_usage_totals_cost 와 동일하게 두되,
    torch 미설치 시 내장 표만 사용한다.
    """
    try:
        from models import claude_message_batch_usage_totals_cost

        return dict(
            claude_message_batch_usage_totals_cost(
                pricing_model_id, prompt_tokens, completion_tokens
            )
        )
    except Exception:
        batch_prices: Dict[str, Tuple[float, float]] = {
            "claude-haiku-4-5": (0.50, 2.50),
            "claude-sonnet-4-6": (1.50, 7.50),
            "claude-opus-4-7": (2.50, 12.50),
        }
        if pricing_model_id in batch_prices:
            pin, pout = batch_prices[pricing_model_id]
            cost = completion_tokens / 1_000_000 * pout + prompt_tokens / 1_000_000 * pin
        else:
            cost = 0.0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        }


def _write_judge_error_row(
    *,
    item: JudgeWorkItem,
    judge_model_id: str,
    err_msg: str,
    out_path: str,
    fmt: str,
    accumulated_by_path: Dict[str, List[Dict[str, Any]]],
    dialogue_chars: int,
    anthropic_batch_id: Optional[str] = None,
) -> None:
    err_row: Dict[str, Any] = {
        "idx": item.idx,
        "source_log_file": item.log_abs,
        "method": _method_label_for_judge_output(item.method),
        "task": item.task,
        "judge_model": judge_model_id,
        "dialogue_chars": dialogue_chars,
        "judge_parse_ok": False,
        "error": err_msg,
    }
    if item.task_profile:
        err_row["task_profile"] = item.task_profile
    if anthropic_batch_id:
        err_row["anthropic_message_batch_id"] = anthropic_batch_id
    if fmt == "jsonl":
        with open(out_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(err_row, ensure_ascii=False) + "\n")
    else:
        acc = accumulated_by_path.setdefault(out_path, load_results_json_array(out_path))
        acc = upsert_result(acc, err_row)
        accumulated_by_path[out_path] = acc
        save_results_json_array(out_path, acc)


def _write_judge_ok_row(
    *,
    item: JudgeWorkItem,
    judge_model_id: str,
    overall_score: int,
    judge_raw: Optional[Dict[str, Any]],
    out_path: str,
    fmt: str,
    accumulated_by_path: Dict[str, List[Dict[str, Any]]],
    dialogue_chars: int,
    anthropic_batch_id: Optional[str] = None,
) -> None:
    row: Dict[str, Any] = {
        "idx": item.idx,
        "source_log_file": item.log_abs,
        "method": _method_label_for_judge_output(item.method),
        "task": item.task,
        "judge_model": judge_model_id,
        "dialogue_chars": dialogue_chars,
        "judge_parse_ok": True,
        "overall_score": overall_score,
    }
    if item.task_profile:
        row["task_profile"] = item.task_profile
    if judge_raw is not None:
        row["judge_raw_response"] = judge_raw
    if anthropic_batch_id:
        row["anthropic_message_batch_id"] = anthropic_batch_id
    if fmt == "jsonl":
        with open(out_path, "a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        acc = accumulated_by_path.setdefault(out_path, load_results_json_array(out_path))
        acc = upsert_result(acc, row)
        accumulated_by_path[out_path] = acc
        save_results_json_array(out_path, acc)


def run_judge_message_batches_for_items(
    items: List[JudgeWorkItem],
    *,
    sdk_client: Any,
    claude_cfg: Dict[str, Any],
    judge_model_id: str,
    pricing_model_id: str,
    judge_user_prompt_template: str,
    output_path_for_item: Callable[[JudgeWorkItem], str],
    fmt: str,
    accumulated_by_path: Dict[str, List[Dict[str, Any]]],
    poll_interval_sec: float,
    max_requests_per_batch: int,
) -> None:
    """Anthropic Message Batches 로 묶어 제출·폴링 후 결과를 출력 파일에 반영한다.

    judge_user_prompt_template: `{dialogue}` 플레이스홀더를 포함한 심판 사용자 메시지 전체 문자열.
    """
    if not items:
        return
    n = len(items)
    max_requests_per_batch = max(1, int(max_requests_per_batch))
    num_chunks = (n + max_requests_per_batch - 1) // max_requests_per_batch
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for c_i in range(num_chunks):
        chunk = items[c_i * max_requests_per_batch : (c_i + 1) * max_requests_per_batch]
        chunk_prompt_tokens = 0
        chunk_completion_tokens = 0
        requests: List[dict] = []
        for it in chunk:
            prompt = judge_user_prompt_template.format(dialogue=it.dialogue)
            params = claude_config_to_message_params(claude_cfg, prompt, "")
            requests.append({"custom_id": it.custom_id, "params": params})

        label = f"bpp-neuro-judge {c_i + 1}/{num_chunks}"
        by_id, bid = run_claude_batches_and_collect(
            sdk_client,
            requests,
            poll_interval_sec=poll_interval_sec,
            model_name=judge_model_id,
            phase_label=label,
        )

        for it in chunk:
            prefix = f"[{it.task_profile}] " if it.task_profile else ""
            out_path = output_path_for_item(it)
            br: Optional[BatchLineResult] = by_id.get(it.custom_id)
            dchars = len(it.dialogue)

            if br is None:
                msg = "배치 결과에 custom_id 없음"
                print(f"{prefix}[idx={it.idx}] judge 실패: {msg}", flush=True)
                _write_judge_error_row(
                    item=it,
                    judge_model_id=judge_model_id,
                    err_msg=msg,
                    out_path=out_path,
                    fmt=fmt,
                    accumulated_by_path=accumulated_by_path,
                    dialogue_chars=dchars,
                    anthropic_batch_id=bid,
                )
                continue

            if not br.ok or not br.body:
                err_d = br.error or {"message": "batch item failed"}
                err_msg = json.dumps(err_d, ensure_ascii=False)
                print(f"{prefix}[idx={it.idx}] judge 실패: {err_msg}", flush=True)
                _write_judge_error_row(
                    item=it,
                    judge_model_id=judge_model_id,
                    err_msg=err_msg,
                    out_path=out_path,
                    fmt=fmt,
                    accumulated_by_path=accumulated_by_path,
                    dialogue_chars=dchars,
                    anthropic_batch_id=bid,
                )
                continue

            pt_u, ct_u = _tokens_from_judge_openai_shaped_body(br.body)
            chunk_prompt_tokens += pt_u
            chunk_completion_tokens += ct_u

            text = _judge_text_from_batch_body(br.body)
            try:
                overall_score = parse_judge_overall_score(text)
            except Exception as e:
                _write_judge_error_row(
                    item=it,
                    judge_model_id=judge_model_id,
                    err_msg=f"{type(e).__name__}: {e}",
                    out_path=out_path,
                    fmt=fmt,
                    accumulated_by_path=accumulated_by_path,
                    dialogue_chars=dchars,
                    anthropic_batch_id=bid,
                )
                print(f"{prefix}[idx={it.idx}] judge 실패: {e}", flush=True)
                continue

            _write_judge_ok_row(
                item=it,
                judge_model_id=judge_model_id,
                overall_score=overall_score,
                judge_raw=br.body,
                out_path=out_path,
                fmt=fmt,
                accumulated_by_path=accumulated_by_path,
                dialogue_chars=dchars,
                anthropic_batch_id=bid,
            )
            print(f"{prefix}idx={it.idx} ok | overall={overall_score}", flush=True)

        total_prompt_tokens += chunk_prompt_tokens
        total_completion_tokens += chunk_completion_tokens
        print(
            f"  [심판 토큰] 청크 {c_i + 1}/{num_chunks}: 입력={chunk_prompt_tokens}, "
            f"출력={chunk_completion_tokens}",
            flush=True,
        )

    cost_info = estimate_message_batch_judge_cost_usd(
        pricing_model_id, total_prompt_tokens, total_completion_tokens
    )
    print(
        f"심판 토큰 합계 (배치 성공 응답 기준): 입력={total_prompt_tokens}, "
        f"출력={total_completion_tokens}, 전체={total_prompt_tokens + total_completion_tokens}",
        flush=True,
    )
    print(
        f"심판 비용 추정 (Message Batch 단가, pricing_model={pricing_model_id}): "
        f"USD ${float(cost_info.get('cost', 0.0)):.6f}",
        flush=True,
    )


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="BPP 로그 대상 뇌기능 정합성 LLM 심판 (Claude)")
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="단일 run.py 결과 jsonl 경로. 생략 시 --logs-root 아래 4개 태스크 BPP 로그를 자동 탐색(배치)",
    )
    p.add_argument(
        "--logs-root",
        type=str,
        default="logs",
        help="배치 모드에서 탐색할 루트 (기본: logs)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "결과 디렉터리(기본 logs/judge) 또는 단일 파일 경로(.jsonl/.json). "
            "디렉터리면 그 아래 <judge-model>/<judge-prompt>/run01/run02/… 및 태스크별 bpp_neuro_judge.jsonl 생성"
        ),
    )
    p.add_argument(
        "--output-format",
        type=str,
        choices=("json", "jsonl"),
        default="jsonl",
        help='저장 형식: jsonl=한 줄에 하나의 JSON(기본), json=단일 JSON 배열 파일',
    )
    p.add_argument("--start", type=int, default=0, help="평가 구간 시작 idx (포함)")
    p.add_argument(
        "--end",
        type=int,
        default=-1,
        help="평가 구간 끝의 다음 idx (미포함). -1 이면 파일 끝까지",
    )
    p.add_argument(
        "--upto-idx",
        type=int,
        default=None,
        metavar="N",
        help="포함하여 이 idx 까지만 평가 (각 로그 파일별). 예: 2 → idx 0,1,2. 설정 시 --end 보다 우선",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default="claude-haiku-4-5",
        choices=list(claude_configs.keys()),
        help="Anthropic 설정 키 (본 스크립트 내장 claude_configs)",
    )
    p.add_argument(
        "--judge-prompt",
        type=str,
        default="template",
        choices=tuple(JUDGE_PROMPT_VARIANTS.keys()),
        help=(
            "심판 사용자 메시지 템플릿: template=짧은 루브릭(기본), "
            "explanation=뇌영역별 참조 기능 설명을 포함한 루브릭"
        ),
    )
    p.add_argument(
        "--include-final-answer-section",
        action="store_true",
        help="Final answer: 이후 본문도 대화 문자열에 포함",
    )
    p.add_argument(
        "--allow-non-bpp-method",
        action="store_true",
        help="기본은 method==bpp 만 평가; 이 플래그 시 bpp 외 method 도 평가 (디버그용)",
    )
    p.add_argument(
        "--batch-poll-interval",
        type=float,
        default=30.0,
        help="Message Batch 처리 상태 폴링 간격(초), 기본 30",
    )
    p.add_argument(
        "--batch-max-requests",
        type=int,
        default=50_000,
        help="배치 한 번에 넣는 최대 요청 수(초과 시 여러 배치로 분할). 기본 50000",
    )
    args = p.parse_args(argv)

    end_idx_for_run = args.end
    if args.upto_idx is not None:
        if args.upto_idx < 0:
            raise SystemExit("--upto-idx 는 0 이상이어야 합니다.")
        if args.upto_idx < args.start:
            raise SystemExit("--upto-idx 는 --start 이상이어야 합니다.")
        end_idx_for_run = args.upto_idx + 1

    batch_mode = args.log_file is None
    single_output_file: Optional[str] = None
    judge_output_root = DEFAULT_JUDGE_OUTPUT_ROOT
    if args.output:
        out_abs = os.path.abspath(args.output)
        if out_abs.endswith(".jsonl") or out_abs.endswith(".json"):
            single_output_file = out_abs
            judge_output_root = os.path.dirname(out_abs) or "."
        else:
            judge_output_root = out_abs
    if batch_mode:
        jobs = resolve_default_batch_logs(args.logs_root)
    else:
        log_path = args.log_file
        jobs = [(None, os.path.abspath(log_path))]

    config = dict(claude_configs[args.judge_model])
    sdk_client = make_anthropic_sdk_client()
    judge_model_id = str(config.get("model") or args.judge_model)
    judge_layout_model_key = args.judge_model

    task_labels = [
        profile if profile else os.path.basename(lpath) for profile, lpath in jobs
    ]
    print(f"심판 모델(API): {judge_model_id}", flush=True)
    print(f"심판 설정 키: {args.judge_model}", flush=True)
    print(f"심판 프롬프트: --judge-prompt {args.judge_prompt}", flush=True)
    print(f"평가 태스크: {', '.join(task_labels)}", flush=True)
    if args.upto_idx is not None:
        print(
            f"인덱스 제한: 각 로그에서 idx {args.start} … {args.upto_idx} (끝 포함)",
            flush=True,
        )
    elif args.start != 0 or args.end != -1:
        end_disp = "파일 끝까지" if args.end == -1 else f"{args.end} 미포함"
        print(f"인덱스 구간: start={args.start}, end={end_disp}", flush=True)

    if batch_mode:
        print("배치 모드: 다음 로그를 순서대로 평가합니다.", flush=True)
        for profile, path in jobs:
            print(f"  - {profile}: {path}", flush=True)

    fmt = args.output_format
    if single_output_file:
        done_keys = loaded_done_keys(single_output_file, fmt)
        os.makedirs(os.path.dirname(single_output_file) or ".", exist_ok=True)
        print(f"결과 파일(단일): {single_output_file}", flush=True)
    else:
        staging_base = os.path.join(
            os.path.abspath(judge_output_root),
            judge_layout_model_key,
            args.judge_prompt,
        )
        baseline_pc = count_pending_judge_items(
            jobs=jobs,
            done_keys=set(),
            start_idx=args.start,
            end_idx=end_idx_for_run,
            include_final_answer_section=args.include_final_answer_section,
            allow_non_bpp_method=args.allow_non_bpp_method,
        )
        if baseline_pc == 0:
            judge_output_root = os.path.join(staging_base, "run01")
            done_keys = loaded_done_keys_under_judge_root(judge_output_root, fmt)
        else:
            judge_output_root, done_keys = resolve_judge_run_directory_and_done_keys(
                staging_base,
                fmt,
                jobs,
                start_idx=args.start,
                end_idx=end_idx_for_run,
                include_final_answer_section=args.include_final_answer_section,
                allow_non_bpp_method=args.allow_non_bpp_method,
            )
        os.makedirs(judge_output_root, exist_ok=True)
        print(f"결과 스테이징 디렉터리: {staging_base}", flush=True)
        print(f"결과 루트(이번 실행): {os.path.abspath(judge_output_root)}", flush=True)

    single_log_stem: Optional[str] = None
    if not batch_mode and args.log_file:
        stem_raw = os.path.splitext(os.path.basename(args.log_file))[0][:120]
        single_log_stem = re.sub(r"[^\w\-.]+", "_", stem_raw)

    pending: List[JudgeWorkItem] = []
    for task_profile, log_path in jobs:
        pending.extend(
            list(
                iter_judge_work_items_for_log(
                    log_path=log_path,
                    start_idx=args.start,
                    end_idx=end_idx_for_run,
                    include_final_answer_section=args.include_final_answer_section,
                    allow_non_bpp_method=args.allow_non_bpp_method,
                    done_keys=done_keys,
                    task_profile=task_profile,
                )
            )
        )

    def _path_for_item(it: JudgeWorkItem) -> str:
        return judge_output_path_for_item(
            it,
            judge_output_root,
            fmt,
            single_file_path=single_output_file,
            single_log_stem=single_log_stem,
            logs_root=args.logs_root,
        )

    if pending and not single_output_file:
        for p in {_path_for_item(it) for it in pending}:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    print(
        f"Anthropic Message Batches: 미처리 심판 {len(pending)}건 제출.",
        flush=True,
    )
    if not pending:
        print("처리할 항목이 없습니다(이미 출력에 포함됨).", flush=True)
    else:
        accumulated_by_path: Dict[str, List[Dict[str, Any]]] = {}
        run_judge_message_batches_for_items(
            pending,
            sdk_client=sdk_client,
            claude_cfg=config,
            judge_model_id=judge_model_id,
            pricing_model_id=judge_model_id,
            judge_user_prompt_template=JUDGE_PROMPT_VARIANTS[args.judge_prompt],
            output_path_for_item=_path_for_item,
            fmt=fmt,
            accumulated_by_path=accumulated_by_path,
            poll_interval_sec=max(1.0, float(args.batch_poll_interval)),
            max_requests_per_batch=max(1, int(args.batch_max_requests)),
        )


if __name__ == "__main__":
    main()
