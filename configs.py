import torch

# TODO: add your custom model config here:
gpt_configs = {
    "gpt-4.1": {
        "model": "gpt-4.1",
        "temperature": 0.0,
        "max_completion_tokens": 29999,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    },
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "max_completion_tokens": 29999,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    },
    "o1-mini": {
        "model": "o1-mini",
        "temperature": 1.0,
        "max_completion_tokens": 59999,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    "gpt-4o": {
        "model": "gpt-4o-2024-08-06",
        "temperature": 0.0,
        "max_tokens": 15999,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    },
    "gpt-4o-mini": {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 15999,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    },
    "gpt35-turbo": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "max_tokens": 3999,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    },
    "gpt-5.2": {
        "model": "gpt-5.2",
        "temperature": 0.0,
        "max_completion_tokens": 128000,
        "reasoning_effort": "none",
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    },
    "gpt-5.4": {
        "model": "gpt-5.4",
        "temperature": 0.0,
        "max_completion_tokens": 128000,
        "reasoning_effort": "none",     # none, low, medium, high, xhigh ==> none일 때만 temperature가 활성화되고, 나머지 mode에서는 무시됨
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    },
    "gpt-5.4-nano": {
        "model": "gpt-5.4-nano",
        "temperature": 0.0,
        "max_completion_tokens": 128000,
        "reasoning_effort": "none",
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    }
}

# Anthropic Claude (Messages API) — max_tokens 필수 (표기 Max Output 상한 근거)
claude_configs = {
    "claude-haiku-4-5": {
        "model": "claude-haiku-4-5",
        "temperature": 0.0,
        "max_tokens": 64000,
    },
    "claude-sonnet-4-6": {
        "model": "claude-sonnet-4-6",
        "temperature": 0.0,
        "max_tokens": 64000,
        "thinking": {"type": "adaptive"}, # disabled, adaptive
        "output_config": {"effort": "high"}, # low, medium, high, max
    },
    "claude-opus-4-7": {
        "model": "claude-opus-4-7",
        "temperature": 0.0,
        "max_tokens": 128000,
    },
}

default_claude_config = {
    "model": None,
    "temperature": 0.0,
    "max_tokens": 64000,
}

# open_model:
# - max_model_len: 프롬프트+생성 합산 상한 (KV). VRAM 부족 시 낮추면 됨.
# - max_new_tokens: 한 번 생성 시 새 토큰 상한. 모델이 EOS를 안 내면 여기까지 쓰므로
#   너무 크게 두면 같은 문단 무한 반복 + 지연만 커짐 (짧은 스토리·bpp는 보통 수백~수천 토큰이면 충분).
open_model_configs = {
    "llama3.1-8b-inst": {
        "task": "text-generation",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "temperature": 0.0,
        "top_p": 0.95,
        "max_model_len": 16384,
        "max_new_tokens": 4096,
        "gpu_memory_utilization": 0.9,
    },
    "qwen2.5-7b-instruct": {
        "task": "text-generation",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "temperature": 0.0,
        "top_p": 0.95,
        "max_model_len": 16384,
        "max_new_tokens": 4096,
        "gpu_memory_utilization": 0.9,
    },
    "qwen2.5-14b-instruct": {
        "task": "text-generation",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "temperature": 0.0,
        "top_p": 0.95,
        "max_model_len": 16384,
        "max_new_tokens": 4096,
        "gpu_memory_utilization": 0.9,
    }
}

default_open_model_config = {
    "task": "text-generation",
    "model": None,
    "torch_dtype": torch.float16,
    "device_map": "auto",
    "temperature": 0.7,
    "top_p": 0.95,
    "max_model_len": 16384,
    "max_new_tokens": 4096,
    "gpu_memory_utilization": 0.9,
}

default_gpt_config = {
    "model": None,
    "temperature": 0.0,
    "max_tokens": 5000,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}