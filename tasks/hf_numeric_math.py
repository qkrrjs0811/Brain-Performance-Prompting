import json
import os
import re
from typing import Any, Dict

from prompts.hf_numeric_math import *  # standard_prompt, cot_prompt, spp_prompt, bpp_prompt ...
from parsing_utils import extract_math_answer, normalize_answer_eval
from tasks.base import Task, DATA_PATH


class HFNumericMathTask(Task):
    """
    HF 수학/숏답형 데이터셋용 태스크.

    데이터 파일 포맷(각 JSONL line):
      - id (필수)
      - question
      - answers
      - metadata (나머지 키들)
    """

    def __init__(self, file: str = "gsm8k.jsonl"):
        super().__init__()
        path = os.path.join(DATA_PATH, "hf_numeric_math", file)
        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    @staticmethod
    def _escape_for_format(text: str) -> str:
        # 질문 텍스트의 리터럴 중괄호가 str.format placeholder로 해석되지 않도록 이스케이프
        return str(text).replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _safe_prompt_format(template: str, **kwargs) -> str:
        """
        템플릿 내부의 예시 텍스트(예: \\boxed{24})에 있는 리터럴 중괄호 때문에
        str.format이 깨지는 것을 방지한다.
        """
        escaped = str(template).replace("{", "{{").replace("}", "}}")
        for key in kwargs:
            escaped = escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}")
        return escaped.format(**kwargs)

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        dp = self.data[idx]
        question = dp.get("question", dp.get("질문"))
        if question is None:
            raise KeyError("질문(question) 키를 찾을 수 없습니다.")
        question = self._escape_for_format(question)

        if method == "standard":
            return self._safe_prompt_format(standard_prompt, input=question)
        if method == "cot":
            return self._safe_prompt_format(cot_prompt, input=question)
        if method == "spp":
            return self._safe_prompt_format(spp_prompt, input=question)
        if method == "bpp":
            return self._safe_prompt_format(bpp_prompt, input=question)
        if method == "self_refine":
            phase = kwargs["phase"]
            if phase == "init":
                return self._safe_prompt_format(standard_prompt, input=question)
            if phase == "feedback":
                return self._safe_prompt_format(
                    self_refine_feedback_prompt,
                    question_answer=kwargs["question_answer"],
                )
            if phase == "refine":
                return self._safe_prompt_format(
                    self_refine_refinement_prompt,
                    question_answer=kwargs["question_answer"],
                    feedback=kwargs["feedback"],
                )
            raise NotImplementedError(f"self_refine phase {phase} not implemented")

        raise NotImplementedError(f"method {method} not implemented for hf_numeric_math")

    @staticmethod
    def _extract_boxed_content(text: str) -> str:
        """
        \\boxed{...} 내부 텍스트를 첫 번째 항목 기준으로 추출.
        중첩 중괄호를 허용하기 위해 간단한 밸런스 파서를 사용.
        """
        s = str(text)
        needle = "\\boxed{"
        start = s.find(needle)
        if start == -1:
            return ""
        i = start + len(needle)
        depth = 1
        buf = []
        while i < len(s):
            ch = s[i]
            if ch == "{":
                depth += 1
                buf.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(buf).strip()
                buf.append(ch)
            else:
                buf.append(ch)
            i += 1
        return ""

    @staticmethod
    def _normalize_answer(s: str) -> str:
        s = str(s).strip()
        boxed = HFNumericMathTask._extract_boxed_content(s)
        if boxed:
            s = boxed
        # 모델이 답만 주지 않고 래핑을 섞어주는 경우 대비
        s = re.sub(r"^(answer\\s*:)", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^final\\s*answer\\s*:?", "", s, flags=re.IGNORECASE).strip()

        # GSM8K/일부 케이스: "#### ..."가 답에 남아있을 수 있음
        m = re.search(r"####\\s*(.+)$", s, flags=re.MULTILINE)
        if m:
            s = m.group(1).strip()

        # 공통적인 비교 정규화
        s = s.replace(",", "")
        s = s.replace(" ", "")
        s = s.replace("$", "")
        s = s.replace("\\\\left", "").replace("\\\\right", "")
        s = s.strip()
        # trailing punctuation 제거
        s = re.sub(r"[\\.!]+$", "", s)
        return s

    @staticmethod
    def _extract_first_int(s: str) -> str:
        m = re.search(r"-?\\d+", s)
        return m.group(0) if m else ""

    def test_output(self, idx: int, output: str) -> Dict[str, Any]:
        dp = self.data[idx]
        gt = dp.get("answers", dp.get("정답", dp.get("answer")))
        if gt is None:
            raise KeyError("answers(answer) 키를 찾을 수 없습니다.")

        # `parsing_utils`에서 먼저 답 후보를 뽑아 prompt_unwrap 단계에서 넘기도록 했지만,
        # 안전하게 output(=unwrapped_output) 자체를 다시 정규화/비교합니다.
        gt_norm = normalize_answer_eval(gt)
        pred_norm = normalize_answer_eval(output)

        # 정답이 정수 형태면, 예측에서 정수만 첫 번째로 뽑아 비교
        # (예: "70.0" -> "70"처럼 통일)
        if re.fullmatch(r"-?\\d+", gt_norm):
            pred_int = self._extract_first_int(pred_norm)
            is_correct = pred_int == gt_norm
        else:
            is_correct = pred_norm == gt_norm

        return {"correct": is_correct}

    @staticmethod
    def prompt_unwrap(response: str, method: str, **kwargs):
        if method == "self_refine":
            phase = kwargs["phase"]
            if phase == "feedback":
                return response, True

        # 파서 공통: completion에서 답 후보를 추출
        unwrapped = extract_math_answer(response)
        if unwrapped:
            return unwrapped, True

        # fallback: 기존 규칙 유지 (모델이 boxed/answer 문구 없이 다른 포맷인 경우 대비)
        if method in ["standard", "cot"]:
            pattern = r"answer\\s*:?\\s*(.*)"
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip(), True
            return response, False

        if method in ["spp", "bpp"]:
            pattern = r"final\\s+answer\\s*:?\\s*(.*)"
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip(), True
            return response, False

        if method == "self_refine":
            phase = kwargs["phase"]
            if phase != "feedback":
                pattern = r"answer\\s*:?\\s*(.*)"
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip(), True
            return response, False

        raise NotImplementedError(f"method {method} not implemented")

