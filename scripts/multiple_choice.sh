#!/usr/bin/env bash
set -euo pipefail

# Multiple-choice (hf_multiple_choice) 실험 스크립트

# MODEL="gpt-5.4-nano" # model: gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini, gpt35-turbo, gpt-5.2, gpt-5.4, gpt-5.4-nano, o1-mini, llama3.1-8b-inst, qwen2.5-7b-instruct, qwen2.5-14b-instruct
# 여러 모델 순차 실험: 앞 모델에서 모든 데이터셋×METHOD×RUN이 끝난 뒤 다음 모델. 비우면 MODEL만 사용합니다.
MODELS=(
  "gpt-4o"
)
OPEN_MODEL_CUDA_DEVICES=()

MODEL_TYPE="gpt" # model_type: gpt | claude | open_model (USE_BATCH=1이면 open_model 불가·gpt/claude는 run_batch.py)

TASK="hf_multiple_choice"

HF_DATASETS=("mmlu_pro_2100") # dataset: gpqa-diamond, mmlu-pro(전체), mmlu_pro_140/mmlu-pro-140(140개), mmlu_pro_1400/mmlu-pro-1400(100/카테고리·1400), mmlu_pro_2100/mmlu-pro-2100(150/카테고리·2100, 시드42) — END_IDX를 데이터 크기에 맞게 조정

START_IDX=0
END_IDX=2100

METHODS=("standard" "spp" "bpp") # method: standard, cot, spp, self_refine, bpp3, bpp, bpp_w_r_demo, bpp_w_k_demo, bpp_two_k_demo, bpp_two_r_demo (주의: USE_BATCH=1이면 self_refine 미지원)
SYSTEM_MESSAGE=""

# 반복 실행(시행/replicate) 설정:
RUNS="${RUNS:-1}"
RUN_START="${RUN_START:-1}"
RUN_WIDTH="${RUN_WIDTH:-2}"
SEED_BASE="${SEED_BASE:-42}"
USE_BATCH="${USE_BATCH:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}" # open_model(vLLM)일 때만 사용

RUNNER="run.py"
if [[ "${USE_BATCH}" == "1" ]]; then
  RUNNER="run_batch.py"
fi

# args에서 --additional_output_note만 추출해(run suffix를 붙이기 위해) 제거합니다.
BASE_ADDITIONAL_OUTPUT_NOTE=""
FORWARD_ARGS=()
ARGS=("$@")
idx=0
while [[ $idx -lt ${#ARGS[@]} ]]; do
  if [[ "${ARGS[$idx]}" == "--additional_output_note" ]]; then
    if [[ $((idx + 1)) -lt ${#ARGS[@]} ]]; then
      BASE_ADDITIONAL_OUTPUT_NOTE="${ARGS[$((idx + 1))]}"
      idx=$((idx + 2))
      continue
    fi
  fi
  FORWARD_ARGS+=("${ARGS[$idx]}")
  idx=$((idx + 1))
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS_RESOLVED=("${MODEL}")
else
  MODELS_RESOLVED=("${MODELS[@]}")
fi

CUDA_MATCH=0
if [[ ${#OPEN_MODEL_CUDA_DEVICES[@]} -eq ${#MODELS_RESOLVED[@]} ]] && [[ ${#MODELS_RESOLVED[@]} -gt 0 ]]; then
  CUDA_MATCH=1
fi
if [[ ${#OPEN_MODEL_CUDA_DEVICES[@]} -gt 0 ]] && [[ "${CUDA_MATCH}" -eq 0 ]]; then
  echo "Warning: OPEN_MODEL_CUDA_DEVICES 길이(${#OPEN_MODEL_CUDA_DEVICES[@]})가 모델 수(${#MODELS_RESOLVED[@]})와 달라 GPU 고정은 적용하지 않습니다." >&2
fi

dataset_to_file() {
  # data/hf_multiple_choice/<file>
  case "$1" in
    gpqa_diamond|gpqa-diamond) echo "gpqa_diamond.jsonl" ;;
    mmlu-pro) echo "mmlu_pro.jsonl" ;;
    mmlu_pro_140|mmlu-pro-140) echo "mmlu_pro_140.jsonl" ;;
    mmlu_pro_1400|mmlu-pro-1400) echo "mmlu_pro_1400.jsonl" ;;
    mmlu_pro_2100|mmlu-pro-2100) echo "mmlu_pro_2100.jsonl" ;;
    *) echo "" ;;
  esac
}

for ((mi = 0; mi < ${#MODELS_RESOLVED[@]}; mi++)); do
  SINGLE_MODEL="${MODELS_RESOLVED[$mi]}"
  CUDEV=""
  if [[ "${CUDA_MATCH}" -eq 1 ]]; then
    CUDEV="${OPEN_MODEL_CUDA_DEVICES[$mi]}"
  fi

  LAUNCH=()
  if [[ -n "${CUDEV}" ]]; then
    LAUNCH=(env "CUDA_VISIBLE_DEVICES=${CUDEV}")
  fi

  echo ""
  echo "##########################################"
  echo "Model ($((mi + 1))/${#MODELS_RESOLVED[@]}): ${SINGLE_MODEL}"
  if [[ -n "${CUDEV}" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDEV}"
  fi
  echo "##########################################"

  for DATASET in "${HF_DATASETS[@]}"; do
    DATA_FILE="$(dataset_to_file "$DATASET")"
    if [[ -z "${DATA_FILE}" ]]; then
      echo "Unknown dataset: ${DATASET}"
      exit 1
    fi

    echo "================================================"
    echo "HF Multiple-choice dataset: ${DATASET} | ${DATA_FILE}"
    echo "================================================"

    for METHOD in "${METHODS[@]}"; do
      echo "=========================================="
      echo "Methods batch (one replicate per run): ${METHOD} | dataset=${DATASET}"
      echo "=========================================="

      for RUN in $(seq "${RUN_START}" "$((RUN_START + RUNS - 1))"); do
        RUN_IDX="$(printf "%0${RUN_WIDTH}d" "${RUN}")"
        if [[ -n "${BASE_ADDITIONAL_OUTPUT_NOTE}" ]]; then
          ADDITIONAL_NOTE="${BASE_ADDITIONAL_OUTPUT_NOTE}_run${RUN_IDX}"
        else
          ADDITIONAL_NOTE="_run${RUN_IDX}"
        fi

        SEED_ARGS=()
        GPU_ARGS=()
        if [[ "${MODEL_TYPE}" == "open_model" ]]; then
          RUN_SEED=$((SEED_BASE + RUN))
          SEED_ARGS=(--seed "${RUN_SEED}")
          GPU_ARGS=(--gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}")
        fi

        echo "------------------------------------------"
        echo "Running model: ${SINGLE_MODEL} | method: ${METHOD} | run: ${RUN_IDX} | dataset=${DATASET}"
        echo "Note: ${ADDITIONAL_NOTE}"
        if [[ "${#SEED_ARGS[@]}" -gt 0 ]]; then
          echo "vLLM seed: ${RUN_SEED} (SEED_BASE=${SEED_BASE})"
          echo "vLLM gpu_memory_utilization: ${GPU_MEMORY_UTILIZATION}"
        fi
        echo "------------------------------------------"

        "${LAUNCH[@]}" python "${RUNNER}" \
          --model "${SINGLE_MODEL}" \
          --model_type "${MODEL_TYPE}" \
          --method "${METHOD}" \
          --task "${TASK}" \
          --task_data_file "${DATA_FILE}" \
          --task_start_index "${START_IDX}" \
          --task_end_index "${END_IDX}" \
          --system_message "${SYSTEM_MESSAGE}" \
          --additional_output_note "${ADDITIONAL_NOTE}" \
          "${GPU_ARGS[@]}" \
          "${SEED_ARGS[@]}" \
          "${FORWARD_ARGS[@]}"

        echo "Completed model: ${SINGLE_MODEL} | method: ${METHOD} | run: ${RUN_IDX} | dataset=${DATASET}"
        echo ""
      done
    done
  done
done

echo "All models and multiple-choice experiments completed!"

