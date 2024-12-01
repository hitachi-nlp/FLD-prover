#!/bin/bash


find ./outputs/01.train.py/2024-09-18.fix_negation | grep "checkpoint-390$" | while read output_dir; do
    parent_dir=`dirname ${output_dir}`
    log_file_path="${parent_dir}/log.txt"

    if [ ! -f "${log_file_path}" ]; then
        echo "dummy log file will be created at ${log_file_path}"

        echo "THIS IS A DUMMY LOG FILE CREATED BY fix_unfinished_logs.sh" > "${log_file_path}"
        echo "train metrics" > "${log_file_path}"

        continue
    fi

done
