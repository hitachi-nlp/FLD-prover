#!/bin/bash

MODEL_NAME_OR_PATH=t5-small
OUTPUT_DIR=./outputs/01.run_FLD.sh/2023-05-09

TRAIN_JSONL=./res/datasets/FLD.schema_fixed/test.jsonl
VALID_JSONL=./res/datasets/FLD.schema_fixed/test.jsonl
TEST_JSONL=./res/datasets/FLD.schema_fixed/test.jsonl

SOURCE_PREFIX="Let's think step-by-step following the rigid formal logic:"

if [ ! -e "${OUTPUT_DIR}" ]; then
    mkdir -p ${OUTPUT_DIR}
fi


python ./run_FLD.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_train \
    --do_eval \
    --train_file "${TRAIN_JSONL}" \
    --validation_file "${VALID_JSONL}" \
    --test_file "${TEST_JSONL}" \
    --file_type "json" \
    --source_prefix "${SOURCE_PREFIX}: " \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --remove_unused_columns=false \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
