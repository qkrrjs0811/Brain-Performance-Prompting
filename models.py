import os
import time
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import logging

import torch
import uuid

# open_model은 vLLM 전용 (미설치 시 open_model 실행 불가)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM이 설치되어 있지 않습니다. open_model(model_type) 실행 시 오류가 납니다.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


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
    "stop": None,
}


def _chat_completion_to_legacy_dict(response, prompt: str, sys_m: str):
    """openai>=1.0 ChatCompletion 객체를 기존 로그/usage 처리용 dict로 변환."""
    choices = []
    for c in response.choices:
        msg = c.message
        content = msg.content if getattr(msg, "content", None) is not None else ""
        choices.append(
            {
                "index": c.index,
                "finish_reason": c.finish_reason,
                "message": {"role": msg.role, "content": content},
            }
        )
    usage = response.usage
    usage_dict = {
        "prompt_tokens": usage.prompt_tokens if usage is not None else 0,
        "completion_tokens": usage.completion_tokens if usage is not None else 0,
        "total_tokens": usage.total_tokens if usage is not None else 0,
    }
    return {
        "id": response.id,
        "object": getattr(response, "object", None) or "chat.completion",
        "created": response.created,
        "model": response.model,
        "choices": choices,
        "usage": usage_dict,
        "prompt": prompt,
        **({"system_message": sys_m} if sys_m else {}),
    }


class OpenAIWrapper:
    def __init__(self, config=DEFAULT_GPT_CONFIG, system_message=""):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not set. Please set your API key.")
        self._client = OpenAI(api_key=api_key)

        self.config = config
        print("api config:", config, "\n")

        # count total tokens
        self.completion_tokens = 0
        self.prompt_tokens = 0

        # system message
        self.system_message = system_message

    # retry using tenacity
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry_error_callback=log_retry_error,
    )
    def completions_with_backoff(self, prompt_for_log: str, sys_m_for_log: str, **kwargs):
        api_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        response = self._client.chat.completions.create(**api_kwargs)
        return _chat_completion_to_legacy_dict(response, prompt_for_log, sys_m_for_log)

    def run(self, prompt, n=1, system_message=""):
        """
        prompt: str
        n: int, total number of generations specified
        """
        try:
            if system_message != "":
                sys_m = system_message
            else:
                sys_m = self.system_message
            if sys_m != "":
                messages = [
                    {"role": "system", "content": sys_m},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            text_outputs = []
            raw_responses = []
            while n > 0:
                cnt = min(n, 10)
                n -= cnt
                res = self.completions_with_backoff(
                    prompt_for_log=prompt,
                    sys_m_for_log=sys_m,
                    messages=messages,
                    n=cnt,
                    **self.config,
                )
                text_outputs.extend([choice["message"]["content"] for choice in res["choices"]])
                raw_responses.append(res)
                self.completion_tokens += res["usage"]["completion_tokens"]
                self.prompt_tokens += res["usage"]["prompt_tokens"]

            return text_outputs, raw_responses
        except Exception as e:
            print("an error occurred:", e)
            return [], []

    def compute_gpt_usage(self):
        return gpt_usage_totals_cost(
            self.config["model"], self.prompt_tokens, self.completion_tokens
        )


def _anthropic_assistant_text(message) -> str:
    parts = []
    for block in message.content:
        btype = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if btype == "text":
            if hasattr(block, "text"):
                parts.append(block.text or "")
            elif isinstance(block, dict):
                parts.append(block.get("text") or "")
    return "".join(parts)


def _anthropic_message_to_legacy_dict(response, prompt: str, sys_m: str):
    assistant_text = _anthropic_assistant_text(response)
    stop_r = getattr(response, "stop_reason", None)
    choices = [
        {
            "index": 0,
            "finish_reason": str(stop_r) if stop_r is not None else "stop",
            "message": {"role": "assistant", "content": assistant_text},
        }
    ]
    u = response.usage
    usage_dict = {
        "prompt_tokens": u.input_tokens if u is not None else 0,
        "completion_tokens": u.output_tokens if u is not None else 0,
        "total_tokens": (u.input_tokens + u.output_tokens) if u is not None else 0,
    }
    return {
        "id": response.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model,
        "choices": choices,
        "usage": usage_dict,
        "prompt": prompt,
        **({"system_message": sys_m} if sys_m else {}),
    }


class AnthropicWrapper:
    """Anthropic Messages API — OpenAIWrapper와 동일한 run()/raw_response 형태 유지."""

    def __init__(self, config, system_message=""):
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic 패키지가 없습니다. pip install anthropic 후 다시 실행하세요."
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. 환경 변수를 설정해 주세요."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.config = config
        print("api config (anthropic):", config, "\n")
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.system_message = system_message

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry_error_callback=log_retry_error,
    )
    def _messages_with_backoff(self, prompt_for_log: str, sys_m_for_log: str, **api_kwargs):
        api_kwargs = {k: v for k, v in api_kwargs.items() if v is not None}
        response = self._client.messages.create(**api_kwargs)
        return _anthropic_message_to_legacy_dict(response, prompt_for_log, sys_m_for_log)

    def run(self, prompt, n=1, system_message=""):
        try:
            sys_m = system_message if system_message != "" else self.system_message
            max_tokens = int(self.config["max_tokens"])
            base_kw = {"model": self.config["model"], "max_tokens": max_tokens}
            thinking = self.config.get("thinking")
            output_config = self.config.get("output_config")
            has_reasoning_controls = isinstance(thinking, dict) or isinstance(output_config, dict)
            if isinstance(thinking, dict):
                base_kw["thinking"] = dict(thinking)
            if isinstance(output_config, dict):
                base_kw["output_config"] = dict(output_config)
            # Claude에서 thinking/effort 제어를 쓸 때는 temperature를 함께 보내지 않는다.
            if (
                not has_reasoning_controls
                and "temperature" in self.config
                and self.config["temperature"] is not None
            ):
                base_kw["temperature"] = float(self.config["temperature"])
            stops = self.config.get("stop_sequences")
            if stops:
                base_kw["stop_sequences"] = stops

            messages = [{"role": "user", "content": prompt}]
            text_outputs = []
            raw_responses = []
            remaining = int(n)
            while remaining > 0:
                batch = min(remaining, 10)
                remaining -= batch
                for _ in range(batch):
                    api_kw = dict(base_kw)
                    api_kw["messages"] = messages
                    if sys_m:
                        api_kw["system"] = sys_m
                    res = self._messages_with_backoff(prompt, sys_m, **api_kw)
                    text_outputs.append(res["choices"][0]["message"]["content"])
                    raw_responses.append(res)
                    self.completion_tokens += res["usage"]["completion_tokens"]
                    self.prompt_tokens += res["usage"]["prompt_tokens"]
            return text_outputs, raw_responses
        except Exception as e:
            print("an error occurred:", e)
            return [], []

    def compute_gpt_usage(self):
        return gpt_usage_totals_cost(
            self.config["model"], self.prompt_tokens, self.completion_tokens
        )


def gpt_usage_totals_cost(model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    """동기 API·Batch 결과 집계용: 모델별 토큰 합계에 대해 run.py와 동일한 단가로 cost를 추정한다."""
    if model == "gpt-4o-2024-08-06":
        cost = completion_tokens / 1000000 * 10.00 + prompt_tokens / 1000000 * 2.50
    elif model == "gpt-3.5-turbo":
        cost = completion_tokens / 1000000 * 1.500 + prompt_tokens / 1000000 * 0.500
    elif model == "gpt-4o-mini":
        cost = completion_tokens / 1000000 * 0.600 + prompt_tokens / 1000000 * 0.150
    elif model == "o1-mini":
        cost = completion_tokens / 1000000 * 4.40 + prompt_tokens / 1000000 * 1.10
    elif model == "gpt-4.1":
        cost = completion_tokens / 1000000 * 8.00 + prompt_tokens / 1000000 * 2.00
    elif model == "gpt-4.1-mini":
        cost = completion_tokens / 1000000 * 1.60 + prompt_tokens / 1000000 * 0.40
    elif model == "gpt-5.2":
        cost = completion_tokens / 1000000 * 14.00 + prompt_tokens / 1000000 * 1.75
    elif model == "gpt-5.4":
        cost = completion_tokens / 1000000 * 15.00 + prompt_tokens / 1000000 * 2.50
    elif model == "gpt-5.4-nano":
        cost = completion_tokens / 1000000 * 1.25 + prompt_tokens / 1000000 * 0.20
    # Claude 단가(MTok): 사용자 제공 최신 표(Ant Claude Opus/Sonnet/Haiku 비교표) 및 docs 기준
    elif model == "claude-haiku-4-5":
        cost = completion_tokens / 1000000 * 5.00 + prompt_tokens / 1000000 * 1.00
    elif model == "claude-sonnet-4-6":
        cost = completion_tokens / 1000000 * 15.00 + prompt_tokens / 1000000 * 3.00
    elif model == "claude-opus-4-7":
        cost = completion_tokens / 1000000 * 25.00 + prompt_tokens / 1000000 * 5.00
    else:
        cost = 0
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
    }


def claude_message_batch_usage_totals_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> dict:
    """Anthropic Message Batches API 단가(MTok): 표준 Messages 대비 할인 반영 분기 가격."""
    # https://platform.claude.com/docs/ko/build-with-claude/batch-processing ( 배치 표 )
    # 동일 계열은 문서상 배치 입·출력 단가가 같음
    batch_prices = {
        "claude-haiku-4-5": (0.50, 2.50),
        "claude-sonnet-4-6": (1.50, 7.50),
        "claude-sonnet-4-5": (1.50, 7.50),
        "claude-sonnet-4": (1.50, 7.50),
        "claude-opus-4-7": (2.50, 12.50),
        "claude-opus-4-6": (2.50, 12.50),
        "claude-opus-4-5": (2.50, 12.50),
    }
    if model not in batch_prices:
        # 미등록 모델은 표준 Claude 단가의 50%로 근사 (배치 가이드 일반 규칙)
        sync = gpt_usage_totals_cost(model, prompt_tokens, completion_tokens)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": sync["cost"] * 0.5,
        }
    pin, pout = batch_prices[model]
    cost = completion_tokens / 1_000_000 * pout + prompt_tokens / 1_000_000 * pin
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
    }


DEFAULT_OPEN_MODEL_CONFIG = {
    "task": "text-generation",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "torch_dtype": torch.float16,
    "device_map": "auto",
    "temperature": 0.7,
    "top_p": 0.95,
    "max_model_len": 16384,
    "max_new_tokens": 4096,
    "gpu_memory_utilization": 0.9,
}


class OpenModelWrapper:
    """로컬 오픈모델은 vLLM만 사용합니다 (CUDA + vllm 패키지 필수)."""

    def __init__(self, config=DEFAULT_OPEN_MODEL_CONFIG, local_model_path=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 중인 디바이스: {device}")
        if device != "cuda":
            raise RuntimeError(
                "open_model은 vLLM만 지원합니다. CUDA GPU가 필요합니다."
            )
        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "open_model은 vLLM이 필요합니다. pip install vllm 후 다시 실행하세요."
            )

        if local_model_path:
            model_path = local_model_path
        else:
            model_path = config["model"]

        print(f"모델 경로: {model_path}")
        self.config = config
        self.model_path = model_path

        if "VLLM_USE_FLASHINFER" not in os.environ:
            os.environ["VLLM_USE_FLASHINFER"] = "0"
            print("cuDNN 충돌 방지: VLLM_USE_FLASHINFER=0 설정 (flashinfer 비활성화)")

        vllm_kwargs = {
            "model": model_path,
            "trust_remote_code": True,
        }
        if "torch_dtype" in config:
            if config["torch_dtype"] == torch.bfloat16:
                vllm_kwargs["dtype"] = "bfloat16"
            elif config["torch_dtype"] == torch.float16:
                vllm_kwargs["dtype"] = "float16"
        if "max_model_len" in config:
            vllm_kwargs["max_model_len"] = config["max_model_len"]
        if "gpu_memory_utilization" in config:
            vllm_kwargs["gpu_memory_utilization"] = config["gpu_memory_utilization"]
        else:
            vllm_kwargs["gpu_memory_utilization"] = 0.9

        print("vLLM을 사용하여 모델을 로드합니다...")
        print(f"  - 모델: {model_path}")
        print(f"  - max_model_len: {vllm_kwargs.get('max_model_len', 'vLLM 기본')}")
        print("  - tensor_parallel_size, enforce_eager: vLLM 기본값")
        try:
            self.llm = LLM(**vllm_kwargs)
            print(
                f"✓ vLLM 로드 완료 (gpu_memory_utilization={vllm_kwargs['gpu_memory_utilization']})"
            )
        except Exception as e:
            raise RuntimeError(f"vLLM 모델 로드 실패: {e}") from e

    def run(self, prompt, n=1, system_message=""):
        sampling_kwargs = {
            "temperature": float(self.config.get("temperature", 0.7)),
            "top_p": float(self.config.get("top_p", 0.95)),
            "n": n,
        }
        if "max_new_tokens" in self.config:
            sampling_kwargs["max_tokens"] = int(self.config["max_new_tokens"])
        top_k = self.config.get("top_k")
        if top_k is not None:
            sampling_kwargs["top_k"] = int(top_k)
        if self.config.get("seed") is not None:
            sampling_kwargs["seed"] = int(self.config["seed"])
        sampling_params = SamplingParams(**sampling_kwargs)

        prompts = [prompt] * n if n > 1 else [prompt]
        outputs = self.llm.generate(prompts, sampling_params)

        text_outputs = []
        raw_responses = []
        for output in outputs:
            for j, generated_text in enumerate(output.outputs):
                gen_text = generated_text.text
                text_outputs.append(gen_text)
                mock_id = str(uuid.uuid4())
                mock_gpt_response_obj = {
                    "id": mock_id,
                    "object": "text-generation",
                    "created": mock_id,
                    "model": self.config["model"],
                    "choices": [
                        {
                            "index": j,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": gen_text,
                            },
                        }
                    ],
                    "usage": {},
                    "prompt": prompt,
                    "system_message": system_message,
                }
                raw_responses.append(mock_gpt_response_obj)

        return text_outputs, raw_responses

    def compute_gpt_usage(self):
        return {}


if __name__ == "__main__":
    open_model = OpenModelWrapper()
    prompt = (
        'I liked "Breaking Bad" and "Band of Brothers". '
        "Do you have any recommendations of other shows I might like?\n"
    )
    text_outputs, raw_responses = open_model.run(prompt)
    print(text_outputs)
    print("\n\n")
    print(raw_responses)
