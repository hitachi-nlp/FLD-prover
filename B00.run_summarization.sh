#!/bin/bash

MODEL_NAME_OR_PATH=t5-small
OUTPUT_DIR=./outputs/B00.run_summarization.sh
DATASET_JSONL=./res/datasets/summarization/sample.json

if [ ! -e "${OUTPUT_DIR}" ]; then
    mkdir -p ${OUTPUT_DIR}
fi


python ./run_summarization.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_train \
    --do_eval \
    --train_file "${DATASET_JSONL}" \
    --validation_file "${DATASET_JSONL}" \
    --source_prefix "summarize: " \
    --output_dir "${OUTPUT_DIR}" \
    --torch_compile=true \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
