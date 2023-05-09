#!/bin/bash

INPUT_DIR=./res/datasets/FLD/
OUTPUT_DIR=./res/datasets/FLD.schema_fixed/

if [ ! -e "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi

for input_file in ${INPUT_DIR}/*.jsonl; do
    filename=`basename ${input_file}`
    output_file=${OUTPUT_DIR}/${filename}

    python ./fix_FLD_schema.py ${input_file} ${output_file}

done
