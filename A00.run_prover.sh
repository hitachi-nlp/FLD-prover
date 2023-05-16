#!/bin/bash

MODEL_NAME_OR_PATH=t5-base
OUTPUT_DIR=./outputs/A00.run_prover.sh/2023-05-15

TRAIN_JSONL=./res/datasets/FLD/20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP/test.jsonl
VALID_JSONL=./res/datasets/FLD/20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP/test.jsonl
TEST_JSONL=./res/datasets/FLD/20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP/test.jsonl
MAX_SAMPLES=10
EPOCHS=300
BATCH_SIZE=4
DO_TRAIN="--do_train"
# DO_TRAIN=""

SOURCE_PREFIX="Let's think step-by-step following the rigid formal logic"

if [ ! -e "${OUTPUT_DIR}" ]; then
    mkdir -p ${OUTPUT_DIR}
fi


python ./run_prover.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    ${DO_TRAIN} \
    --do_eval \
    --do_predict \
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
    --logging_dir "${OUTPUT_DIR}/tensorboard_logs" \
    --overwrite_output_dir \
    --remove_unused_columns=false \
    --per_device_train_batch_size=${BATCH_SIZE} \
    --per_device_eval_batch_size=${BATCH_SIZE} \
    --generation_top_k 10\
    --predict_with_generate
