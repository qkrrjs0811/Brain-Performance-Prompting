import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
import logging  

from transformers import AutoTokenizer
import transformers
import torch
import uuid



# Configure logging  
logging.basicConfig(level=logging.INFO)  
  
# Error callback function
def log_retry_error(retry_state):  
    logging.error(f"Retrying due to error: {retry_state.outcome.exception()}")  



DEFAULT_GPT_CONFIG = {
    "model": "gpt-4o-2024-08-06",
    "temperature": 0.0,
    "max_tokens": 15999,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}

class OpenAIWrapper:
    def __init__(self, config = DEFAULT_GPT_CONFIG, system_message=""):
        # TODO: set up your API key with the environment variable OPENAIKEY
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not openai.api_key:
            raise ValueError("OpenAI API key is not set. Please set your API key.")
      

        # if os.environ.get("USE_AZURE")=="True":
        #     print("using azure api")
        #     openai.api_type = "azure"
        # openai.api_base = os.environ.get("API_BASE")
        # openai.api_version = os.environ.get("API_VERSION")

        self.config = config
        print("api config:", config, '\n')

        # count total tokens
        self.completion_tokens = 0
        self.prompt_tokens = 0

        # system message
        self.system_message = system_message # "You are an AI assistant that helps people find information."

    # retry using tenacity
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), retry_error_callback=log_retry_error)
    def completions_with_backoff(self, **kwargs):
        # print("making api call:", kwargs)
        # print("====================================")
        return openai.ChatCompletion.create(**kwargs)

    def run(self, prompt, n=1, system_message=""):
        """
            prompt: str
            n: int, total number of generations specified
        """
        try:
            # overload system message
            if system_message != "":
                sys_m = system_message
            else:
                sys_m = self.system_message
            if sys_m != "":
                # print("adding system message:", sys_m)
                messages = [
                    {"role":"system", "content":sys_m},
                    {"role":"user", "content":prompt}
                ]
            else:
                messages = [
                    {"role":"user","content":prompt}
                ]
            text_outputs = []
            raw_responses = []
            while n > 0:
                cnt = min(n, 10) # number of generations per api call
                n -= cnt
                res = self.completions_with_backoff(messages=messages, n=cnt, **self.config)
                text_outputs.extend([choice["message"]["content"] for choice in res["choices"]])
                # add prompt to log
                res['prompt'] = prompt
                if sys_m != "":
                    res['system_message'] = sys_m
                raw_responses.append(res)
                # log completion tokens
                self.completion_tokens += res["usage"]["completion_tokens"]
                self.prompt_tokens += res["usage"]["prompt_tokens"]

            return text_outputs, raw_responses
        except Exception as e:
            print("an error occurred:", e)
            return [], []

    def compute_gpt_usage(self):
        model = self.config["model"]
        if model == "gpt-4o-2024-08-06":
            cost = self.completion_tokens / 1000000 * 10.00 + self.prompt_tokens / 1000000 * 2.50
        elif model == 'gpt-3.5-turbo':
            cost = self.completion_tokens / 1000000 * 1.500 + self.prompt_tokens / 1000000 * 0.500
        elif model == 'gpt-4o-mini':
            cost = self.completion_tokens / 1000000 * 0.600 + self.prompt_tokens / 1000000 * 0.150
        elif model == 'o1-mini':
            cost = self.completion_tokens / 1000000 * 12.00 + self.prompt_tokens / 1000000 * 3.00
        elif model == 'gpt-4.1':
            cost = self.completion_tokens / 1000000 * 8.00 + self.prompt_tokens / 1000000 * 2.00
        elif model == 'gpt-4.1-mini':
            cost = self.completion_tokens / 1000000 * 1.60 + self.prompt_tokens / 1000000 * 0.40
        else:
            cost = 0 # TODO: add custom cost calculation for other engines
        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}


DEFAULT_OPEN_MODEL_CONFIG = {
    "task": "text-generation",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "torch_dtype": torch.float16,
    "device_map": "auto",
    "do_sample": False,
    "local_model_path": "./models"  # 로컬 모델 경로 추가
}

class OpenModelWrapper:
    def __init__(self, config = DEFAULT_OPEN_MODEL_CONFIG, local_model_path=None):
        # GPU 사용 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 중인 디바이스: {device}")
        
        # 로컬 모델 경로 설정
        if local_model_path:
            model_path = local_model_path
        elif config.get("local_model_path") and os.path.exists(config["local_model_path"]):
            model_path = config["local_model_path"]
        else:
            model_path = config["model"]
        
        print(f"모델 경로: {model_path}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=config.get("local_model_path", None),
            local_files_only=os.path.exists(config.get("local_model_path", ""))
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # CUDA 사용 (에러 해결을 위한 올바른 설정)
        print(f"CUDA 디바이스 사용: {device}")
        
        # pipeline 생성 시 rope_scaling 문제를 피하기 위한 설정
        try:
            self.pipeline = transformers.pipeline(
                task=config["task"],
                model=model_path,
                tokenizer=self.tokenizer,
                device_map=device,
                dtype=config["torch_dtype"] if device == "cuda" else torch.float32,
                do_sample=config["do_sample"],
                trust_remote_code=True,  # Llama-3.1 모델을 위한 설정
                model_kwargs={"attn_implementation": "eager"}  # rope_scaling 문제 해결을 위한 설정
            )
        except Exception as e:
            print(f"Pipeline 생성 중 오류 발생: {e}")
            print("기본 설정으로 재시도 중...")
            # 기본 설정으로 재시도
            self.pipeline = transformers.pipeline(
                task=config["task"],
                model=model_path,
                tokenizer=self.tokenizer,
                device_map=device,
                dtype=config["torch_dtype"] if device == "cuda" else torch.float32,
                do_sample=config["do_sample"]
            )
        self.config = config
        self.model_path = model_path

    def run(self, prompt, n=1, system_message=""):
        # TODO: make this configurable
        sequences = self.pipeline(
            prompt,
            do_sample=self.config["do_sample"],
            num_return_sequences=n,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024  # 더 긴 토큰 제한
        )
        
        # convert generation output into the same format as GPT raw response
        text_outputs = []
        raw_responses = []
        for seq in sequences:
            # remove prompt from the generated text
            gen_text = seq['generated_text'][len(prompt):]
            text_outputs.append(gen_text)
            mock_id = str(uuid.uuid4())
            mock_gpt_response_obj = {
                "id": mock_id,
                "object": "text-generation",
                "created": mock_id,
                "model": self.config["model"],
                "choices": [
                    {
                        "index":0,
                        "finish_reason": "stop",
                        "message":{
                            "role": "assistant",
                            "content":gen_text
                        }
                    }
                ],
                "usage": {},
                "prompt":prompt,
                "system_message":system_message
            }
            raw_responses.append(mock_gpt_response_obj)
        return text_outputs, raw_responses
    
    def compute_gpt_usage(self):
        return {}


if __name__ == "__main__":
    open_model = OpenModelWrapper()
    prompt = '''I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'''
    text_outputs, raw_responses = open_model.run(prompt)
    print(text_outputs)
    print('\n\n')
    print(raw_responses)
