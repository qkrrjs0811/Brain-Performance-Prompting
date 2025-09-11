MODEL="llama3.1-8b-inst" # your engine name: gpt-4.1, gpt-4.1-mini, gpt-4o, gpt35-turbo, gpt-4o-mini, o1-mini, llama3.1-8b-inst, qwen2.5-7b-instruct
MODEL_TYPE="open_model" # 'gpt' or 'open_model'

DATA_FILE="validataion.jsonl" # GLUE validation data file

START_IDX=0
END_IDX=200

# choose methods to run (space-separated)
METHODS=("standard" "spp" "bpp") # ["standard", "cot", "spp", "bpp"]

# w/ or w/o system message (BPP works without a system message, but you can add one if you want)
SYSTEM_MESSAGE="" # or "You are an AI assistant that helps people find information."

# GLUE 데이터셋 목록
GLUE_SUBSETS=("cola" "sst2" "mrpc" "qqp" "rte" "qnli")

# 각 GLUE 데이터셋에 대해 실험 실행
for SUBSET in "${GLUE_SUBSETS[@]}"; do
    echo "=========================================="
    echo "Running GLUE subset: $SUBSET"
    echo "=========================================="
    
    # Run each method for current subset
    for METHOD in "${METHODS[@]}"; do
        echo "------------------------------------------"
        echo "Running method: $METHOD for $SUBSET"
        echo "------------------------------------------"
        
        python run.py \
            --model ${MODEL} \
            --model_type ${MODEL_TYPE} \
            --method ${METHOD} \
            --task glue_${SUBSET} \
            --task_data_file ${DATA_FILE} \
            --task_start_index ${START_IDX} \
            --task_end_index ${END_IDX} \
            --system_message "${SYSTEM_MESSAGE}" \
            ${@}
        
        echo "Completed method: $METHOD for $SUBSET"
        echo ""
    done
    
    echo "Completed all methods for $SUBSET"
    echo ""
done

echo "All GLUE experiments completed!"
