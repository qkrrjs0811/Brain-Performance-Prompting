import json
import os
import re
from typing import Any, Dict, List

from prompts.hf_multiple_choice import *  # standard_prompt, cot_prompt, spp_prompt, bpp_prompt ...
from parsing_utils import extract_answer_gpqa, extract_answer_mmlu_pro
from tasks.base import Task, DATA_PATH


class HFMultipleChoiceTask(Task):
    """
    HF 객관식(선지) 데이터셋용 태스크.

    데이터 파일 포맷(각 JSONL line):
      - id (필수)
      - question
      - answers : 기본적으로 'A','B' 같은 정답 라벨로 저장한다고 가정
      - metadata (나머지 키들)
        - metadata['options'] 를 가진 경우: 프롬프트에 choices를 직접 구성
        - gpqa처럼 문제 텍스트에 Choices가 이미 포함된 경우: options가 없어도 동작
    """

    def __init__(self, file: str = "mmlu_pro_100.jsonl"):
        super().__init__()
        # 데이터셋에 따라 답 문자 추출 규칙이 달라질 수 있어 구분합니다.
        if file.startswith("gpqa_diamond_"):
            self.dataset_name = "GPQA-Diamond"
        elif file.startswith("mmlu_pro_"):
            self.dataset_name = "MMLU-Pro"
        else:
            self.dataset_name = None

        path = os.path.join(DATA_PATH, "hf_multiple_choice", file)
        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    @staticmethod
    def _letters_for_options(options: List[Any]) -> List[str]:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if len(options) > len(letters):
            raise ValueError(f"옵션이 너무 많습니다: {len(options)}")
        return list(letters[: len(options)])

    def _build_input_text(self, dp: Dict[str, Any]) -> str:
        question = dp.get("question", dp.get("질문"))
        if question is None:
            raise KeyError("질문(question) 키를 찾을 수 없습니다.")

        meta = dp.get("metadata") or {}
        options = meta.get("options")
        if options:
            letters = self._letters_for_options(options)
            choices = "\n".join([f"({l}) {opt}" for l, opt in zip(letters, options)])
            return f"{question}\n\nChoices:\n{choices}\n\nAnswer with only the option letter."

        # gpqa처럼 질문 본문에 choices가 이미 포함된 경우가 많음
        return f"{question}\n\nAnswer with only the option letter."

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        dp = self.data[idx]
        input_text = self._build_input_text(dp)

        if method == "standard":
            return standard_prompt.format(input=input_text)
        if method == "cot":
            return cot_prompt.format(input=input_text)
        if method == "spp":
            return spp_prompt.format(input=input_text)
        if method == "bpp":
            return bpp_prompt.format(input=input_text)

        if method == "self_refine":
            phase = kwargs["phase"]
            if phase == "init":
                return standard_prompt.format(input=input_text)
            if phase == "feedback":
                return self_refine_feedback_prompt.format(question_answer=kwargs["question_answer"])
            if phase == "refine":
                return self_refine_refinement_prompt.format(
                    question_answer=kwargs["question_answer"],
                    feedback=kwargs["feedback"],
                )
            raise NotImplementedError(f"self_refine phase {phase} not implemented")

        raise NotImplementedError(f"method {method} not implemented for hf_multiple_choice")

    @staticmethod
    def _extract_answer_field_letter(pred: str) -> str:
        """
        instruction 예시: "answer": "C"
        위 형태를 우선 파싱하고, "The answer is (C)" / answer: C 형태도 허용.
        """
        s = str(pred)

        # Natural sentence style: The answer is (C) / the answer is C
        m = re.search(
            r"\bthe\s+answer\s+is\s*\(?\s*([A-Za-z])\s*\)?",
            s,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

        # JSON-like: "answer": "C"
        m = re.search(r'"answer"\s*:\s*"([A-Za-z])"', s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # key-value: answer: C / Answer : (C)
        m = re.search(r"\banswer\b\s*:\s*\(?\s*([A-Za-z])\s*\)?", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        return ""

    @staticmethod
    def _normalize_option_text(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"\s+", "", s)
        return s

    def _extract_predicted_letter(
        self,
        pred: str,
        allowed: List[str],
        options: List[Any],
    ) -> str:
        pred_u = str(pred).strip().upper()

        # instruction 우선 파싱: "answer": "C"
        answer_field_letter = self._extract_answer_field_letter(pred)
        if answer_field_letter and answer_field_letter in set(allowed):
            return answer_field_letter

        # (A), A, B) 같은 형태를 우선적으로 추출
        allowed_pattern = "|".join([re.escape(x) for x in allowed])
        m = re.search(rf"\(?\s*({allowed_pattern})\s*\)?", pred_u)
        if m:
            return m.group(1)

        # fallback 1) 예측이 A/B/C/... 라벨만 출력
        for a in allowed:
            if pred_u == a:
                return a

        # fallback 2) 예측이 option text를 그대로 말한 경우
        if options:
            pred_norm = self._normalize_option_text(pred)
            for letter, opt in zip(allowed, options):
                opt_norm = self._normalize_option_text(opt)
                if opt_norm and (opt_norm in pred_norm or pred_norm in opt_norm):
                    return letter
        return ""

    def test_output(self, idx: int, output: str) -> Dict[str, Any]:
        dp = self.data[idx]
        meta = dp.get("metadata") or {}
        gt = dp.get("answers", dp.get("정답", dp.get("answer")))
        if gt is None:
            raise KeyError("answers(answer) 키를 찾을 수 없습니다.")
        gt_letter = str(gt).strip().upper()

        options = meta.get("options") or []
        if options:
            allowed = self._letters_for_options(options)
        else:
            allowed = [gt_letter]

        pred_letter = self._extract_predicted_letter(output, allowed=allowed, options=options)
        is_correct = pred_letter == gt_letter
        return {"correct": is_correct}

    def prompt_unwrap(self, response: str, method: str, **kwargs):
        answer_field_letter = HFMultipleChoiceTask._extract_answer_field_letter(response)
        if answer_field_letter:
            return answer_field_letter, True

        # dataset-specific fallback parser
        if self.dataset_name == "GPQA-Diamond":
            letter = extract_answer_gpqa(response)
            if letter:
                return letter, True
        elif self.dataset_name == "MMLU-Pro":
            letter = extract_answer_mmlu_pro(response)
            if letter:
                return letter, True

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
            if phase == "feedback":
                return response, True
            pattern = r"answer\\s*:?\\s*(.*)"
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip(), True
            return response, False

        raise NotImplementedError(f"method {method} not implemented")

