import os
import re
from tasks.base import Task, DATA_PATH
from prompts.trivia_creative_writing import *
import json
# from models import gpt

class TriviaCreativeWritingTask(Task):
    def __init__(self, file='trivia_creative_writing_100_n_5.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'trivia_creative_writing', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        datapoint = self.data[idx]
        questions = datapoint["questions"]
        topic = datapoint["topic"]
        n = len(questions)
        questions_str = " ".join(questions)
        
        if method == "standard":
            input_prompt = standard_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "cot":
            input_prompt = cot_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "macro_bpp":
            input_prompt = macro_bpp_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "meso_bpp":
            input_prompt = meso_bpp_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "micro_bpp":
            input_prompt = micro_bpp_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "bpp":
            input_prompt = bpp_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "spp":
            input_prompt = spp_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "bpp_w_r_demo":
            input_prompt = bpp_prompt_w_r_demo.format(n=n, questions=questions_str, topic=topic)
        elif method == "bpp_w_k_demo":
            input_prompt = bpp_prompt_w_k_demo.format(n=n, questions=questions_str, topic=topic)
        elif method == "bpp_two_k_demo":
            input_prompt = bpp_prompt_two_k_demo.format(n=n, questions=questions_str, topic=topic)
        elif method == "bpp_two_r_demo":
            input_prompt = bpp_prompt_two_r_demo.format(n=n, questions=questions_str, topic=topic)
        elif method == "self_refine":
            phase = kwargs["phase"]
            if phase == "init":
                input_prompt = standard_prompt.format(n=n, questions=questions_str, topic=topic)
            elif phase == "feedback":
                input_prompt = self_refine_feedback_prompt.format(question_answer=kwargs["question_answer"])
            elif phase == "refine":
                input_prompt = self_refine_refinement_prompt.format(question_answer=kwargs["question_answer"], feedback=kwargs["feedback"])
        else:
            raise NotImplementedError(f"method {method} not implemented")
        
        return input_prompt

    def test_output(self, idx: int, output: str):
        # test whether the output includes all the answers of the trivia questions
        instance = self.data[idx]
        correct_count = 0
        question_count = len(instance["answers"])
        for ans_to_question in instance["answers"]:
            for ans in ans_to_question:
                # compare all to lower
                if ans.lower() in output.lower():
                    correct_count += 1
                    break
        info = {'correct_count': correct_count, 'question_count': question_count}
        return info

    @staticmethod
    def prompt_unwrap(response: str, method: str, **kwargs):
        '''
            response: raw genration from the model
            return:
                - str: the story
                - bool: whether the story is successfully parsed from the raw genration
        '''
        if method in ["standard", "self_refine"]:
            return response, True
        
        elif method == "cot":
            if "Story:" in response:
                return response.split("Story:")[1].strip(), True
            elif "story:" in response:
                return response.split("story:")[1].strip(), True
            else:
                return response, False
        
        elif method in ["macro_bpp", "meso_bpp", "micro_bpp", "bpp", "spp", "bpp_w_r_demo", "bpp_w_k_demo", "bpp_two_k_demo", "bpp_two_r_demo"]:
            # 대소문자 구분 없이 "final answer" 패턴 찾기
            pattern = r'final\s+answer\s*:?\s*(.*)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip(), True
            else:
                return response, False            

        else:
            raise NotImplementedError(f"method {method} not implemented")