import os
import re
from tasks.base import Task, DATA_PATH
from prompts.codenames_collaborative import *
import json

class CodenamesCollaborativeTask(Task):
    def __init__(self, file='codenames_50.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'codenames_collaborative', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        datapoint = self.data[idx]
        word_list = datapoint['word_list']
        word_list_str = ", ".join(word_list)
        target_words = datapoint['target_words']
        target_words_str = ", ".join(target_words)

        # for guesser
        assert 'role' in kwargs
        role = kwargs['role']
        if role == 'guesser':
            assert 'hint_word' in kwargs
            hint_word = kwargs['hint_word']
        else:
            hint_word = None

        n = len(target_words)
        if role == 'spymaster':
            if method == "standard":
                input_prompt = standard_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "cot":
                input_prompt = cot_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "macro_bpp":
                input_prompt = macro_bpp_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "meso_bpp":
                input_prompt = meso_bpp_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "micro_bpp":
                input_prompt = micro_bpp_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "bpp":
                input_prompt = bpp_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "spp":
                input_prompt = spp_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "bpp_w_r_demo":
                input_prompt = bpp_prompt_w_r_demo_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "bpp_w_k_demo":
                input_prompt = bpp_prompt_w_k_demo_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)    
            elif method == "bpp_two_k_demo":
                input_prompt = bpp_prompt_two_k_demo_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "bpp_two_r_demo":
                input_prompt = bpp_prompt_two_r_demo_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)       
            elif method == "self_refine":
                phase = kwargs["phase"]
                if phase == "init":
                    input_prompt = standard_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
                elif phase == "feedback":
                    input_prompt = self_refine_feedback_prompt.format(question_answer=kwargs["question_answer"])
                elif phase == "refine":
                    input_prompt = self_refine_refinement_prompt.format(question_answer=kwargs["question_answer"], feedback=kwargs["feedback"])
            else:
                raise NotImplementedError(f"method {method} not implemented for spymaster role")
        elif role == 'guesser':
            if method == "standard":
                input_prompt = standard_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "cot":
                input_prompt = cot_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "macro_bpp":
                input_prompt = macro_bpp_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "meso_bpp":
                input_prompt = meso_bpp_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "micro_bpp":
                input_prompt = micro_bpp_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "bpp":
                input_prompt = bpp_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "spp":
                input_prompt = spp_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "bpp_w_r_demo":
                input_prompt = bpp_prompt_w_r_demo_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "bpp_w_k_demo":
                input_prompt = bpp_prompt_w_k_demo_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "bpp_two_k_demo":
                input_prompt = bpp_prompt_two_k_demo_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "bpp_two_r_demo":
                input_prompt = bpp_prompt_two_r_demo_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "self_refine":
                phase = kwargs["phase"]
                if phase == "init":
                    input_prompt = standard_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
                elif phase == "feedback":
                    input_prompt = self_refine_feedback_prompt.format(question_answer=kwargs["question_answer"])
                elif phase == "refine":
                    input_prompt = self_refine_refinement_prompt.format(question_answer=kwargs["question_answer"], feedback=kwargs["feedback"])
            else:
                raise NotImplementedError(f"method {method} not implemented for guesser role")
        else:
            raise NotImplementedError(f"role {role} not implemented; choose from 'spymaster' or 'guesser'")
        return input_prompt

    def test_output(self, idx: int, output: str):
        # test whether the output includes all the answers of the trivia questions
        datapoint = self.data[idx]
        target_words = datapoint['target_words']
        target_words = [word.strip().lower() for word in target_words]

        predicted_words = output.split(",")
        predicted_words = [word.strip().replace(".","").lower() for word in predicted_words]
        
        # ground truth set
        target_words_set = set(target_words)
        # predicted set
        predicted_words_set = set(predicted_words)
        
        common_words = predicted_words_set.intersection(target_words_set)
        common_words = list(common_words)
        info = {"matched_words":common_words, "matched_count":len(common_words), "target_count":len(target_words_set)}
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
            if "Answer:" in response:
                return response.split("Answer:")[1].strip(), True
            elif "answer:" in response:
                return response.split("answer:")[1].strip(), True
            else:
                return response, True
        
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
                if "Answer:" in response:
                    return response.split("Answer:")[1].strip(), True
                elif "answer:" in response:
                    return response.split("answer:")[1].strip(), True
                else:
                    return response, True
        else:
            raise NotImplementedError(f"method {method} not implemented")