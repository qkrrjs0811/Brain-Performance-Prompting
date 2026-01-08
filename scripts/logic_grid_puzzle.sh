MODEL="llama3.1-8b-inst" # your engine name: gpt-4.1, gpt-4.1-mini, gpt-4o, gpt35-turbo, gpt-4o-mini, o1-mini, llama3.1-8b-inst, qwen2.5-7b-instruct
MODEL_TYPE="open_model" # 'gpt' or 'open_model'

DATA_FILE="logic_grid_puzzle_200.jsonl"

START_IDX=0
END_IDX=200

# choose method
METHODS=("spp" "bpp") # ["standard", "cot", "self_refine", "spp", "bpp", "macro_bpp", "meso_bpp", "micro_bpp", "bpp_w_r_demo", "bpp_w_k_demo", "bpp_two_k_demo", "bpp_two_r_demo"]

# w/ or w/o system message (BPP works without a system message, but you can add one if you want)
SYSTEM_MESSAGE="" # or "You are an AI assistant that helps people find information."

# Run each method
for METHOD in "${METHODS[@]}"; do
    echo "=========================================="
    echo "Running method: $METHOD"
    echo "=========================================="
    
    python run.py \
        --model ${MODEL} \
        --model_type ${MODEL_TYPE} \
        --method ${METHOD} \
        --task logic_grid_puzzle \
        --task_data_file ${DATA_FILE} \
        --task_start_index ${START_IDX} \
        --task_end_index ${END_IDX} \
        --system_message "${SYSTEM_MESSAGE}" \
        ${@}
    
    echo "Completed method: $METHOD"
    echo ""
done

echo "All methods completed!"

