import os
import re
from tasks.base import Task, DATA_PATH
from prompts.glue import *
import json

class GLUETask(Task):
    def __init__(self, file='validataion.jsonl', subset='cola'):
        super().__init__()
        self.subset = subset
        path = os.path.join(DATA_PATH, 'glue', subset, file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        datapoint = self.data[idx]
        
        if self.subset == 'cola':
            sentence = datapoint["sentence"]
            if method == "standard":
                input_prompt = cola_standard_prompt.format(sentence=sentence)
            elif method == "spp":
                input_prompt = cola_spp_prompt.format(sentence=sentence)
            elif method == "bpp":
                input_prompt = cola_bpp_prompt.format(sentence=sentence)
            else:
                raise NotImplementedError(f"method {method} not implemented for {self.subset}")
                
        elif self.subset == 'sst2':
            sentence = datapoint["sentence"]
            if method == "standard":
                input_prompt = sst2_standard_prompt.format(sentence=sentence)
            elif method == "spp":
                input_prompt = sst2_spp_prompt.format(sentence=sentence)
            elif method == "bpp":
                input_prompt = sst2_bpp_prompt.format(sentence=sentence)
            else:
                raise NotImplementedError(f"method {method} not implemented for {self.subset}")
                
        elif self.subset == 'mrpc':
            sentence1 = datapoint["sentence1"]
            sentence2 = datapoint["sentence2"]
            if method == "standard":
                input_prompt = mrpc_standard_prompt.format(sentence1=sentence1, sentence2=sentence2)
            elif method == "spp":
                input_prompt = mrpc_spp_prompt.format(sentence1=sentence1, sentence2=sentence2)
            elif method == "bpp":
                input_prompt = mrpc_bpp_prompt.format(sentence1=sentence1, sentence2=sentence2)
            else:
                raise NotImplementedError(f"method {method} not implemented for {self.subset}")
                
        elif self.subset == 'qqp':
            question1 = datapoint["question1"]
            question2 = datapoint["question2"]
            if method == "standard":
                input_prompt = qqp_standard_prompt.format(question1=question1, question2=question2)
            elif method == "spp":
                input_prompt = qqp_spp_prompt.format(question1=question1, question2=question2)
            elif method == "bpp":
                input_prompt = qqp_bpp_prompt.format(question1=question1, question2=question2)
            else:
                raise NotImplementedError(f"method {method} not implemented for {self.subset}")
                
        elif self.subset == 'rte':
            sentence1 = datapoint["sentence1"]
            sentence2 = datapoint["sentence2"]
            if method == "standard":
                input_prompt = rte_standard_prompt.format(sentence1=sentence1, sentence2=sentence2)
            elif method == "spp":
                input_prompt = rte_spp_prompt.format(sentence1=sentence1, sentence2=sentence2)
            elif method == "bpp":
                input_prompt = rte_bpp_prompt.format(sentence1=sentence1, sentence2=sentence2)
            else:
                raise NotImplementedError(f"method {method} not implemented for {self.subset}")
                
        elif self.subset == 'qnli':
            question = datapoint["question"]
            sentence = datapoint["sentence"]
            if method == "standard":
                input_prompt = qnli_standard_prompt.format(question=question, sentence=sentence)
            elif method == "spp":
                input_prompt = qnli_spp_prompt.format(question=question, sentence=sentence)
            elif method == "bpp":
                input_prompt = qnli_bpp_prompt.format(question=question, sentence=sentence)
            else:
                raise NotImplementedError(f"method {method} not implemented for {self.subset}")
        else:
            raise NotImplementedError(f"subset {self.subset} not implemented")
        
        return input_prompt

    def test_output(self, idx: int, output: str):
        # GLUE 태스크의 정확도 평가
        instance = self.data[idx]
        true_label = instance["label"]
        
        # 예측된 라벨 추출
        predicted_label = self._extract_label(output, self.subset)
        
        # 정확도 계산
        is_correct = (predicted_label == true_label)
        
        info = {
            'correct': is_correct,
            'true_label': true_label,
            'predicted_label': predicted_label
        }
        return info

    def _extract_label(self, output: str, subset: str):
        """출력에서 라벨을 추출합니다."""
        output_lower = output.lower().strip()
        
        if subset == 'cola':
            # 0: unacceptable, 1: acceptable
            if 'unacceptable' in output_lower or '0' in output_lower:
                return 0
            elif 'acceptable' in output_lower or '1' in output_lower:
                return 1
        elif subset == 'sst2':
            # 0: negative, 1: positive
            if 'negative' in output_lower or '0' in output_lower:
                return 0
            elif 'positive' in output_lower or '1' in output_lower:
                return 1
        elif subset == 'mrpc':
            # 0: not_equivalent, 1: equivalent
            if 'not_equivalent' in output_lower or 'not equivalent' in output_lower or '0' in output_lower:
                return 0
            elif 'equivalent' in output_lower or '1' in output_lower:
                return 1
        elif subset == 'qqp':
            # 0: not_duplicate, 1: duplicate
            if 'not_duplicate' in output_lower or 'not duplicate' in output_lower or '0' in output_lower:
                return 0
            elif 'duplicate' in output_lower or '1' in output_lower:
                return 1
        elif subset == 'rte':
            # 0: entailment, 1: not_entailment
            if 'entailment' in output_lower and 'not' not in output_lower or '0' in output_lower:
                return 0
            elif 'not_entailment' in output_lower or 'not entailment' in output_lower or '1' in output_lower:
                return 1
        elif subset == 'qnli':
            # 0: entailment, 1: not_entailment
            if 'entailment' in output_lower and 'not' not in output_lower or '0' in output_lower:
                return 0
            elif 'not_entailment' in output_lower or 'not entailment' in output_lower or '1' in output_lower:
                return 1
        
        # 기본값: 첫 번째 숫자 또는 라벨을 찾아서 반환
        numbers = re.findall(r'\b[01]\b', output)
        if numbers:
            return int(numbers[0])
        
        return None

    @staticmethod
    def prompt_unwrap(response: str, method: str, **kwargs):
        '''
            response: raw generation from the model
            return:
                - str: the answer
                - bool: whether the answer is successfully parsed from the raw generation
        '''
        if method in ["standard"]:
            return response, True
        
        elif method == "cot":
            # Chain of thought에서 최종 답변 추출
            # 대소문자 구분 없이 "answer" 패턴 찾기
            pattern = r'answer\s*:?\s*(.*)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip(), True
            else:
                return response, False
        
        elif method in ["spp", "bpp"]:
            # SPP/BPP에서 최종 답변 추출
            # 대소문자 구분 없이 "final answer" 패턴 찾기
            pattern = r'final\s+answer\s*:?\s*(.*)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip(), True
            else:
                return response, False

        else:
            raise NotImplementedError(f"method {method} not implemented")
