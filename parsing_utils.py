"""
parsing_utils.py

기존 `parsing.py`(다른 프로젝트에서 사용하던 전처리/파싱 로직)를
이 프로젝트에서도 재사용할 수 있도록 필요한 유틸만 옮겨 둔 파일입니다.

주요 제공 기능:
- normalize_answer_eval: LaTeX/수식 문자열 정규화 및 비교용 변환
- extract_math_answer: AIME/MATH 계열에서 completion에서 답을 추출
- extract_answer_gpqa: GPQA-Diamond(A/B/C/D)에서 답 문자 추출
- extract_answer_mmlu_pro: MMLU-Pro(A-J)에서 답 문자 추출
"""

from __future__ import annotations

import re
from typing import Callable, Optional


# ============================================================
# Normalization (normalize_answer_eval)
# ============================================================


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    """
    수식/LaTeX 형식 문자열을 비교 용도로 정규화.
    (원본 `parsing.py`의 로직을 축약/동일하게 이식)
    """

    # linebreaks
    string = string.replace("\n", "")

    # remove thousands separators in numbers (e.g., "125,000" -> "125000")
    string = string.replace(",", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace `\\` with `\`
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # strip leading `k = ...` style assignments (very short LHS only)
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2} ...
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    return string


def normalize_answer_eval(answer: object) -> str:
    """
    답변을 정규화하여 비교 가능하게 만듦.

    - 복합 수식(\\sqrt, \\frac, ^, { 등)이 있으면: 숫자만 추출하지 않고 _strip_string 결과 전체 반환
    - 단순 숫자만 있는 경우: 숫자만 추출 (예: "70.0", " 70 " -> "70")
    - 숫자가 없으면: 소문자·공백 정리
    """

    if answer is None:
        return ""

    answer_str = _strip_string(str(answer).strip())

    # \text{Yes} 같은 단순 텍스트 케이스는 내부 문자열만 추출
    text_only_match = re.fullmatch(r"\\text\{([^}]*)\}", answer_str)
    if text_only_match:
        answer_str = text_only_match.group(1).strip()

    # Yes/No는 텍스트가 길어도 일관되게 판정되도록 우선 처리
    lower_str = answer_str.lower()
    if re.search(r"\byes\b", lower_str):
        return "yes"
    if re.search(r"\bno\b", lower_str):
        return "no"

    # 복합 수식이면 마지막 숫자만 쓰지 않고 전체를 반환
    _COMPOUND_MARKERS = ("\\sqrt", "\\frac", "^", "\\cdot", "\\times", "\\div", "\\pi")
    is_compound = (
        any(m in answer_str for m in _COMPOUND_MARKERS)
        or ("{" in answer_str and "}" in answer_str)
        or bool(re.search(r"\\[a-zA-Z]", answer_str))
    )
    if is_compound:
        return answer_str

    # 단순 숫자만: 마지막 숫자만 추출
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if numbers:
        return numbers[-1]

    return answer_str.lower().strip()


# ============================================================
# Parsing (extract_* functions)
# ============================================================


def _basic_strip(s: object) -> str:
    # SC 단계에서만 쓰는 최소 정제.
    if s is None:
        return ""
    s = str(s).strip()
    if s.startswith("$"):
        s = s[1:].strip()
    if s.endswith("$"):
        s = s[:-1].strip()
    return s


def _extract_braced_content(s: str, start_at: int) -> str:
    """
    s[start_at] == '{' 라고 가정하고 중첩 braces를 허용해 닫는 '}'까지 추출.
    """

    if start_at < 0 or start_at >= len(s) or s[start_at] != "{":
        return ""

    stack = 1
    i = start_at + 1
    buf: list[str] = []
    while i < len(s):
        ch = s[i]
        if ch == "{":
            stack += 1
            buf.append(ch)
        elif ch == "}":
            stack -= 1
            if stack == 0:
                return "".join(buf).strip()
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    return ""


def extract_math_answer(pred_str: Optional[object]) -> str:
    """MATH/AIME 계열에서 reasoning path 텍스트에서 답을 뽑기 위한 규칙."""

    if pred_str is None:
        return ""
    pred_str = str(pred_str).strip()
    if not pred_str:
        return ""

    pred = ""

    # Prefer \boxed{...}
    if "\\boxed" in pred_str:
        # last occurrence of \\boxed to reduce early distractions
        idx = pred_str.rfind("\\boxed")
        # expected next char is '{' (or content begins after keyword)
        brace_start = pred_str.find("{", idx)
        pred = _extract_braced_content(pred_str, brace_start)

    # Also support \framebox{...} (원본 `parsing.py`에는 boxed only가 있었음)
    if not pred and "\\framebox" in pred_str:
        idx = pred_str.rfind("\\framebox")
        brace_start = pred_str.find("{", idx)
        pred = _extract_braced_content(pred_str, brace_start)

    # Fallback: original boxed-split heuristic (covers cases like '...boxed{...}')
    if not pred and "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) and ans[0] == "{":
            pred = _extract_braced_content(ans, 0)
        else:
            pred = ans.split("$")[0].strip()

    if not pred and "The answer is " in pred_str:
        answer_part = pred_str.split("The answer is ")[-1].strip()
        if "." in answer_part:
            pred = answer_part.split(".")[0].strip()
        elif "\n" in answer_part:
            pred = answer_part.split("\n")[0].strip()
        else:
            pred = answer_part

    if not pred and "the answer is " in pred_str:
        answer_part = pred_str.split("the answer is ")[-1].strip()
        if "." in answer_part:
            pred = answer_part.split(".")[0].strip()
        elif "\n" in answer_part:
            pred = answer_part.split("\n")[0].strip()
        else:
            pred = answer_part

    if not pred:
        # 숫자(정수/실수/음수) 중 마지막 값을 후보로 사용
        pattern = r"-?\d*\.?\d+"
        found = re.findall(pattern, pred_str)
        pred = found[-1] if found else ""

    # remove trailing dots/slashes left by some formats
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]

    return _basic_strip(pred)


def extract_answer_gpqa(content: Optional[object]) -> Optional[str]:
    """GPQA-Diamond: A/B/C/D 선택지에서 답을 뽑는 규칙."""

    if content is None:
        return None
    content = str(content)

    if "boxed" in content or "\\boxed" in content or "\\framebox" in content:
        # try: boxed content
        ans = content.split("boxed")[-1] if "boxed" in content else content.split("\\boxed")[-1]
        if len(ans) and ans[0] == "{":
            pred = _extract_braced_content(ans, 0)
        else:
            pred = ans.split("$")[0].strip()

        a = _basic_strip(pred)
        if a:
            text_match = re.search(r"\\text(?:bf|it|tt)?\{([^}]+)\}", a)
            if text_match:
                a = text_match.group(1)

            boxed_match = re.search(r"([A-D])[\.]?", a.strip())
            if boxed_match:
                return boxed_match.group(1).upper()

    # Keyword based: (answer|final answer|correct answer) followed by a letter
    answer_keyword_pattern = (
        r"(?i)(?:answer|final answer|correct answer)(?:\s+\w+)*\s*[:=\-]?\s*"
    )
    answer_matches = list(re.finditer(answer_keyword_pattern, content))
    if answer_matches:
        for answer_match in reversed(answer_matches):
            text_after = content[answer_match.end() :]
            letter_match = re.search(r"[\s\$]*([A-D])[\.]?(?:\s|$)", text_after)
            if letter_match:
                return letter_match.group(1).upper()

    ANSWER_PATTERN = (
        r"(?i)(?:answer|final answer|correct answer)(?:\s+\w+)*\s*[:=\-]?\s*[\$]?\s*([A-D])[\$]?[\.]?(?:\s|$)"
    )
    matches = list(re.finditer(ANSWER_PATTERN, content))
    if matches:
        return matches[-1].group(1).upper()

    stripped_content = content.strip()
    if stripped_content and stripped_content[-1].upper() in ["A", "B", "C", "D"]:
        last_part = stripped_content[-15:].lower()
        if any(keyword in last_part for keyword in ["answer"]):
            return stripped_content[-1].upper()

    return None


def extract_answer_mmlu_pro(content: Optional[object]) -> Optional[str]:
    """MMLU-Pro: A-J 선택지에서 답을 뽑는 규칙."""

    if content is None:
        return None
    content = str(content)

    ANSWER_PATTERN = r"((?:[A-J]\b))"
    matches = list(re.finditer(ANSWER_PATTERN, content, flags=re.IGNORECASE))
    if matches:
        return matches[-1].group(1).upper()
    return None


def get_extractor(dataset_name: Optional[str]) -> Callable[[object], object]:
    """
    dataset_name에 따라 extract 규칙 함수를 반환합니다.

    - 수학 계열(MATH/AIME 등): extract_math_answer
    - GPQA-Diamond: extract_answer_gpqa
    - MMLU-Pro: extract_answer_mmlu_pro
    """

    if dataset_name in ["AIME_2024", "AIME_2025", "MATH500_test", "MATH_train", "MATH2500", "MATH", "MATH_all"]:
        return extract_math_answer
    if dataset_name == "GPQA-Diamond":
        return extract_answer_gpqa
    if dataset_name == "MMLU-Pro":
        return extract_answer_mmlu_pro
    return extract_math_answer


def extract_answer_by_dataset(dataset_name: Optional[str], text: object) -> str:
    """dataset_name에 맞는 추출기를 사용해 문자열 답을 반환합니다."""

    extractor = get_extractor(dataset_name)
    out = extractor(text)  # type: ignore[misc]
    return str(out) if out else ""

