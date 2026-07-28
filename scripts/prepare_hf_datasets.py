import argparse
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


DATA_OUT_NUM_DIR = os.path.join("data", "hf_numeric_math")
DATA_OUT_MC_DIR = os.path.join("data", "hf_multiple_choice")

DEFAULT_SEED = 42


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_filename(s: str) -> str:
    # 파일명 안정성: 영문숫자/언더스코어만 남김
    s = s.strip()
    s = s.replace("-", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _extract_gsm8k_answer(raw_answer: str) -> str:
    # GSM8K는 보통 "#### 18" 형태로 최종답을 포함
    m = re.search(r"####\s*(.+)\s*$", raw_answer, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return raw_answer.strip()


def _extract_simple_answer(raw: str) -> str:
    # 공통적으로 "#### <ans>" 같은 패턴을 우선 제거
    m = re.search(r"####\s*(.+)\s*$", raw, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _sample_indices_uniform(total: int, k: int, seed: int) -> List[int]:
    rnd = random.Random(seed)
    if total <= k:
        return list(range(total))
    return rnd.sample(range(total), k)


def _proportional_targets_by_counts(counts: Dict[str, int], total: int) -> Dict[str, int]:
    # counts에 비례해서 total개의 샘플을 배분(내림 + 큰 fractional 우선)
    sum_counts = sum(counts.values())
    if sum_counts == 0:
        return {k: 0 for k in counts}

    raw = {k: (v / sum_counts) * total for k, v in counts.items()}
    floors = {k: int(math.floor(x)) for k, x in raw.items()}
    current = sum(floors.values())
    remaining = total - current
    if remaining <= 0:
        return floors

    # fractional parts 내림차순으로 remaining개 배분
    fracs = sorted(((raw[k] - floors[k], k) for k in counts.keys()), reverse=True)
    out = dict(floors)
    for i in range(remaining):
        out[fracs[i][1]] += 1
    return out


def _select_random(ds, k: int, seed: int):
    # datasets.Dataset shuffle/select
    if len(ds) <= k:
        return ds
    return ds.shuffle(seed=seed).select(range(k))


def prepare_gsm8k(seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    # 전체 official test split (랜덤 서브샘플 없음)
    _ = seed
    ds = load_dataset("openai/gsm8k", "main", split="test")
    records: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        q = ex["question"]
        raw_a = ex["answer"]
        a = _extract_gsm8k_answer(raw_a)
        records.append(
            {
                "id": f"gsm8k_{i}",
                "question": q,
                "answers": a,
                "metadata": {"raw_answer": raw_a},
            }
        )
    return "gsm8k.jsonl", records


def prepare_math500(seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    # HuggingFace MATH-500: 전체 test split (500문항)
    _ = seed
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    records: List[Dict[str, Any]] = []
    for ex in ds:
        q = ex["problem"]
        a = ex["answer"]
        uid = ex["unique_id"]
        records.append(
            {
                "id": uid,
                "question": q,
                "answers": a,
                "metadata": {
                    "solution": ex.get("solution"),
                    "subject": ex.get("subject"),
                    "level": ex.get("level"),
                },
            }
        )
    return "math500.jsonl", records


def prepare_amc23(seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    ds = load_dataset("math-ai/amc23", split="test")
    ds = _select_random(ds, 100, seed=seed)
    records: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        q = ex["question"]
        raw_a = ex["answer"]
        a = _extract_simple_answer(raw_a)
        records.append(
            {
                "id": ex.get("id", i),
                "question": q,
                "answers": a,
                "metadata": {"url": ex.get("url")},
            }
        )
    return "amc23.jsonl", records


def prepare_aime_2024(seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    ds = _select_random(ds, 100, seed=seed)
    records: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        q = ex["problem"]
        raw_a = ex["answer"]
        a = _extract_simple_answer(str(raw_a))
        records.append(
            {
                "id": ex.get("id", i),
                "question": q,
                "answers": a,
                "metadata": {"solution": ex.get("solution"), "url": ex.get("url"), "year": ex.get("year")},
            }
        )
    return "aime2024.jsonl", records


def prepare_aime_2025(seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    ds_all = load_dataset("allenai/aime-2022-2025", split="train")
    # AIME 2025만: URL에 2025_AIME가 포함되는 row 필터
    idxs_2025 = [idx for idx in range(len(ds_all)) if "2025_AIME" in ds_all[idx]["url"]]
    ds = ds_all.select(idxs_2025)
    ds = _select_random(ds, 100, seed=seed)

    records: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        q = ex["problem"]
        raw_a = ex["answer"]
        a = _extract_simple_answer(str(raw_a))
        records.append(
            {
                "id": ex.get("id", i),
                "question": q,
                "answers": a,
                "metadata": {"solution": ex.get("solution"), "url": ex.get("url"), "year": "2025"},
            }
        )
    return "aime2025.jsonl", records


def prepare_gpqa_diamond(seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    ds = load_dataset("aradhye/gpqa_diamond", split="train")
    ds = _select_random(ds, 100, seed=seed)
    records: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        q = ex["problem"]
        a = str(ex["answer"]).strip().upper()
        records.append(
            {
                "id": ex.get("Unnamed: 0", i),
                "question": q,
                "answers": a,
                "metadata": {},
            }
        )
    return "gpqa_diamond_100.jsonl", records


def _index_to_option_letter(idx: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if idx < 0 or idx >= len(letters):
        raise ValueError(f"옵션 개수가 너무 많습니다: idx={idx}")
    return letters[idx]


def prepare_mmlu_pro(seed: int) -> Tuple[str, List[Dict[str, Any]]]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    # category별로 비례 배분해서 정확히 100개 샘플
    rnd = random.Random(seed)
    indices_by_cat: Dict[str, List[int]] = {}
    for idx in range(len(ds)):
        cat = str(ds[idx]["category"])
        indices_by_cat.setdefault(cat, []).append(idx)

    counts = {cat: len(idxs) for cat, idxs in indices_by_cat.items()}
    targets = _proportional_targets_by_counts(counts, 100)

    selected: List[int] = []
    for cat, k in targets.items():
        if k <= 0:
            continue
        idxs = indices_by_cat[cat]
        if len(idxs) < k:
            raise RuntimeError(f"MMLU-Pro category={cat} 샘플부족: {len(idxs)} < {k}")
        selected.extend(rnd.sample(idxs, k))

    if len(selected) != 100:
        # 라운딩 오차 방지
        selected = selected[:100]
        if len(selected) < 100:
            remaining_pool = [i for i in range(len(ds)) if i not in set(selected)]
            selected.extend(rnd.sample(remaining_pool, 100 - len(selected)))

    rnd.shuffle(selected)
    ds = ds.select(selected)

    records: List[Dict[str, Any]] = []
    for ex in ds:
        q = ex["question"]
        options = list(ex["options"])
        answer_index = int(ex["answer_index"])
        answer_letter = _index_to_option_letter(answer_index)
        records.append(
            {
                "id": ex["question_id"],
                "question": q,
                "answers": answer_letter,
                "metadata": {
                    "options": options,
                    "answer_index": answer_index,
                    "answer_text": ex.get("answer"),
                    "cot_content": ex.get("cot_content"),
                    "category": ex.get("category"),
                    "src": ex.get("src"),
                },
            }
        )
    return "mmlu_pro_100.jsonl", records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="쉼표로 여러 데이터셋 지정(예: gsm8k,math500,mmlu-pro). 비우면 전부.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    want = None
    if args.only.strip():
        want = {x.strip() for x in args.only.split(",") if x.strip()}

    numeric_fns = {
        "gsm8k": prepare_gsm8k,
        "math500": prepare_math500,
        "amc23": prepare_amc23,
        "aime2024": prepare_aime_2024,
        "aime2025": prepare_aime_2025,
    }
    mc_fns = {
        "gpqa-diamond": prepare_gpqa_diamond,
        "mmlu-pro": prepare_mmlu_pro,
    }

    # 사용자가 요청한 목록: gsm8k, math500, amc23, aime2024, aime2025, gpqa-diamond, mmlu-pro
    expected_all = set(numeric_fns.keys()) | set(mc_fns.keys())
    if want is None:
        want = expected_all
    else:
        missing = want - expected_all
        if missing:
            raise SystemExit(f"알 수 없는 데이터셋 요청: {sorted(missing)}")

    for name in sorted(want):
        if name in numeric_fns:
            filename, records = numeric_fns[name](seed=args.seed)
            out_path = os.path.join(DATA_OUT_NUM_DIR, filename)
            print(f"[num] {name}: writing {len(records)} rows -> {out_path}")
            _write_jsonl(out_path, records)
        elif name in mc_fns:
            filename, records = mc_fns[name](seed=args.seed)
            out_path = os.path.join(DATA_OUT_MC_DIR, filename)
            print(f"[mc] {name}: writing {len(records)} rows -> {out_path}")
            _write_jsonl(out_path, records)
        else:
            raise RuntimeError("unreachable")

    print("All requested datasets prepared.")


if __name__ == "__main__":
    main()
