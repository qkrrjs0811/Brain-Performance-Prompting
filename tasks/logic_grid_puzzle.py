import os
import re
from tasks.base import Task, DATA_PATH
from prompts.logic_grid_puzzle import *
import json


target_aliases = {
    "1": "first",
    "2": "second",
    "3": "third",
    "4": "fourth",
    "5": "fifth",
    "6": "sixth",
    "7": "seventh",
    "8": "eighth",
    "9": "ninth",
    "10": "tenth"
}

class LogicGridPuzzleTask(Task):
    def __init__(self, file='logic_grid_puzzle_200.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'logic_grid_puzzle', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        datapoint = self.data[idx]
        input_str = datapoint['inputs']
        
        input_str = input_str.replace("\nA:", "")

        if method == "standard":
            input_prompt = standard_prompt.format(input=input_str)
        elif method == "cot":    
            input_prompt = cot_prompt.format(input=input_str)
        elif method == "macro_bpp":
            input_prompt = macro_bpp_prompt.format(input=input_str)
        elif method == "meso_bpp":
            input_prompt = meso_bpp_prompt.format(input=input_str)
        elif method == "micro_bpp":
            input_prompt = micro_bpp_prompt.format(input=input_str)
        elif method == "bpp":
            input_prompt = bpp_prompt.format(input=input_str)
        elif method == "spp":
            input_prompt = spp_prompt.format(input=input_str)
        elif method == "bpp_w_r_demo":
            input_prompt = bpp_prompt_w_r_demo.format(input=input_str)
        elif method == "bpp_w_k_demo":
            input_prompt = bpp_prompt_w_k_demo.format(input=input_str)
        elif method == "bpp_two_k_demo":
            input_prompt = bpp_prompt_two_k_demo.format(input=input_str)
        elif method == "bpp_two_r_demo":
            input_prompt = bpp_prompt_two_r_demo.format(input=input_str)
        elif method == "self_refine":
            phase = kwargs["phase"]
            if phase == "init":
                input_prompt = standard_prompt.format(input=input_str)
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
        target = instance["targets"][0]
        targets = [target]
        if target in target_aliases:
            targets.append(target_aliases[target])
        
        # get all other candidates
        not_targets = []
        for i in range(1, 11):
            if str(i) not in targets:
                not_targets.append(str(i))
                not_targets.append(target_aliases[str(i)])
        # print("targets", targets)
        # print("negatives", not_targets)
        info = {'correct': False}
        for target in targets:
            if target.lower().strip() in output.lower().strip(): # if the target is in the output
                info['correct'] = True
                # and if all the other targets are not in the output
                for not_target in not_targets:
                    if not_target.lower().strip() in output.lower().strip():
                        info['correct'] = False
                        break
                break
        return info

    @staticmethod
    def prompt_unwrap(response: str, method: str, **kwargs):
        '''
            response: raw genration from the model
            return:
                - str: the story
                - bool: whether the story is successfully parsed from the raw genration
        '''
        # take only the first few characters (enough for successfully parsed output) -> aviod unparsed result to have high accuracy when test output
        if method in ["standard", "cot"]:
            # 대소문자 구분 없이 "answer" 패턴 찾기
            pattern = r'answer\s*:?\s*(.*)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip(), True
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
        
        elif method == "self_refine":
            phase = kwargs["phase"]
            if phase == "feedback":
                return response, True
            else:
                # 대소문자 구분 없이 "answer" 패턴 찾기
                pattern = r'answer\s*:?\s*(.*)'
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return match.group(1).strip(), True
                else:
                    return response, False
        else:
            raise NotImplementedError(f"method {method} not implemented")