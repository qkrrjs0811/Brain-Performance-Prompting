#!/usr/bin/env python3
"""
Llama-3.1-8b-instruct 모델 다운로드 및 로컬 로드 스크립트
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from huggingface_hub import snapshot_download
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaModelLoader:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", cache_dir="./models"):
        """
        Llama 모델 로더 초기화
        
        Args:
            model_name (str): Hugging Face 모델 이름
            cache_dir (str): 모델을 저장할 로컬 디렉토리
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        
        # 모델 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)
        
    def download_model(self):
        """모델을 로컬에 다운로드"""
        logger.info(f"모델 다운로드 시작: {self.model_name}")
        
        try:
            # 모델과 토크나이저 다운로드
            model_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            logger.info(f"모델 다운로드 완료: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"모델 다운로드 실패: {e}")
            raise
    
    def load_model(self, use_quantization=False, device_map="auto"):
        """
        다운로드된 모델을 메모리에 로드
        
        Args:
            use_quantization (bool): 4-bit 양자화 사용 여부 (기본값: False)
            device_map (str): 디바이스 매핑 방식
        """
        logger.info("모델 로딩 시작...")
        
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 설정
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "local_files_only": True,
                "dtype": torch.float16,
                "device_map": device_map,
                "trust_remote_code": True
            }
            
            # 양자화 설정 (메모리 절약) - bitsandbytes가 설치된 경우에만
            if use_quantization and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("4-bit 양자화 활성화")
                except ImportError:
                    logger.warning("bitsandbytes가 설치되지 않았습니다. 양자화 없이 로드합니다.")
                    use_quantization = False
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if use_quantization:
                logger.info("모델 로딩 완료 (양자화 적용)")
            else:
                logger.info("모델 로딩 완료 (양자화 없음)")
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise
    
    def generate_text(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """
        텍스트 생성
        
        Args:
            prompt (str): 입력 프롬프트
            max_length (int): 최대 생성 길이
            temperature (float): 생성 온도
            top_p (float): Top-p 샘플링 값
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        # 입력 토큰화
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048  # 최대 입력 길이 제한
        )
        
        # GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 텍스트 생성 - 더 안전한 파라미터 사용
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_length - inputs['input_ids'].shape[1], 200),  # max_new_tokens 사용
                    temperature=max(temperature, 0.1),  # 최소 온도 보장
                    top_p=min(max(top_p, 0.1), 0.95),   # top_p 범위 제한
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # 반복 방지
                    no_repeat_ngram_size=2,  # n-gram 반복 방지
                    early_stopping=True
                )
            except Exception as e:
                logger.warning(f"첫 번째 생성 시도 실패: {e}")
                # 더 보수적인 파라미터로 재시도
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
        
        # 생성된 텍스트 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 원본 프롬프트 제거
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def is_model_downloaded(self):
        """모델이 이미 다운로드되었는지 확인"""
        model_path = os.path.join(self.cache_dir, "models--meta-llama--Llama-3.1-8B-Instruct")
        return os.path.exists(model_path)

def main():
    """메인 실행 함수"""
    # 모델 로더 초기화
    loader = LlamaModelLoader()
    
    # 모델이 다운로드되지 않은 경우 다운로드
    if not loader.is_model_downloaded():
        print("모델을 다운로드합니다...")
        loader.download_model()
    else:
        print("모델이 이미 다운로드되어 있습니다.")
    
    # 모델 로드
    print("모델을 로딩합니다...")
    loader.load_model(use_quantization=False)  # 양자화 비활성화
    
    # 테스트 실행
    test_prompt = "안녕하세요! 오늘 날씨가 어떤가요?"
    print(f"\n테스트 프롬프트: {test_prompt}")
    
    try:
        response = loader.generate_text(test_prompt, max_length=200)
        print(f"모델 응답: {response}")
    except Exception as e:
        print(f"생성 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
