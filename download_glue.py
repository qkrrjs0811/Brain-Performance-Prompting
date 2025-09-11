#!/usr/bin/env python3
"""
GLUE 데이터셋 다운로드 스크립트
Hugging Face datasets 라이브러리를 사용하여 GLUE 데이터셋을 다운로드하고 
/data/glue 경로에 저장합니다.
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
import json

def create_directory(path):
    """디렉토리가 존재하지 않으면 생성합니다."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"디렉토리 생성: {path}")

def download_glue_dataset():
    """GLUE 데이터셋의 validataion split만 다운로드하고 저장합니다."""
    
    # 데이터 저장 경로 설정 (프로젝트 내부 data 디렉토리 사용)
    data_dir = "data/glue"
    create_directory(data_dir)
    
    # GLUE 데이터셋의 선택된 서브셋 목록 (6개)
    glue_subsets = [
        'cola',      # Corpus of Linguistic Acceptability
        'sst2',      # Stanford Sentiment Treebank v2
        'mrpc',      # Microsoft Research Paraphrase Corpus
        'qqp',       # Quora Question Pairs
        'rte',       # Recognizing Textual Entailment
        'qnli',      # Question Natural Language Inference
    ]
    
    print("GLUE 데이터셋 validataion split 샘플링 다운로드를 시작합니다...")
    print("각 데이터셋별로 최대 200개 샘플을 추출합니다.")
    print(f"저장 경로: {data_dir}")
    print("=" * 50)
    
    downloaded_datasets = {}
    
    for subset in glue_subsets:
        try:
            print(f"\n{subset.upper()} validataion 데이터셋 다운로드 중...")
            
            # 데이터셋 로드 (validataion split만)
            dataset = load_dataset('glue', subset, split='validation')
            
            # 200개 샘플만 추출 (데이터셋이 200개 미만이면 전체 사용)
            sample_size = min(200, len(dataset))
            sampled_dataset = dataset.select(range(sample_size))
            
            # validataion split 저장
            subset_dir = os.path.join(data_dir, subset)
            create_directory(subset_dir)
            
            # JSONL 형식으로 저장
            output_file = os.path.join(subset_dir, "validataion.jsonl")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in sampled_dataset:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            print(f"  - validataion: {len(sampled_dataset)}개 샘플 저장됨 (전체 {len(dataset)}개 중)")
            
            # 메타데이터 저장
            metadata = {
                'subset': subset,
                'splits': {'validataion': len(sampled_dataset)},
                'total_examples': len(sampled_dataset),
                'original_validataion_size': len(dataset),
                'sampled_size': len(sampled_dataset)
            }
            
            metadata_file = os.path.join(subset_dir, 'metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            downloaded_datasets[subset] = metadata
            print(f"  ✓ {subset} validataion 완료")
            
        except Exception as e:
            print(f"  ✗ {subset} validataion 다운로드 실패: {str(e)}")
            continue
    
    # 전체 요약 정보 저장
    summary = {
        'total_subsets': len(downloaded_datasets),
        'downloaded_subsets': list(downloaded_datasets.keys()),
        'datasets_info': downloaded_datasets,
        'total_examples': sum(info['total_examples'] for info in downloaded_datasets.values())
    }
    
    summary_file = os.path.join(data_dir, 'glue_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 50)
    print("GLUE 데이터셋 validataion split 샘플링 다운로드 완료!")
    print(f"총 {len(downloaded_datasets)}개 서브셋의 validataion split 샘플링됨")
    print(f"총 {summary['total_examples']}개 validataion 예제 (각 최대 200개)")
    print(f"요약 정보: {summary_file}")
    print(f"데이터 위치: {data_dir}")

def main():
    """메인 함수"""
    print("GLUE 데이터셋 다운로더")
    print("=" * 50)
    
    # 필요한 라이브러리 확인
    try:
        import datasets
        print(f"datasets 라이브러리 버전: {datasets.__version__}")
    except ImportError:
        print("ERROR: datasets 라이브러리가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요: pip install datasets")
        sys.exit(1)
    
    # 데이터셋 다운로드 실행
    download_glue_dataset()

if __name__ == "__main__":
    main()
