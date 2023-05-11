#!/bin/bash

MODEL_NAME_OR_PATH=t5-base
OUTPUT_DIR=./outputs/01.run_FLD.sh/2023-05-11

TRAIN_JSONL=./res/datasets/FLD.schema_fixed/test.jsonl
VALID_JSONL=./res/datasets/FLD.schema_fixed/test.jsonl
TEST_JSONL=./res/datasets/FLD.schema_fixed/test.jsonl
MAX_SAMPLES=100
EPOCHS=10
BATCH_SIZE=4

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
    --num_train_epochs ${EPOCHS} \
    --max_train_samples ${MAX_SAMPLES} \
    --max_eval_samples ${MAX_SAMPLES} \
    --max_predict_samples ${MAX_SAMPLES} \
    --source_prefix "${SOURCE_PREFIX}: " \
    --max_source_length 1500\
    --max_target_length 100\
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --remove_unused_columns=false \
    --per_device_train_batch_size=${BATCH_SIZE} \
    --per_device_eval_batch_size=${BATCH_SIZE} \
    --predict_with_generate 1>${OUTPUT_DIR}/log.txt 2>&1
