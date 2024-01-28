#!/bin/bash

STATS_DIR=./outputs/A00.count_tokens.py/20230120.jpn.large

for dir in ${STATS_DIR}/*; do
    dataset_name=`echo ${dir} | sed 's:.*FLD_dtst_unm=\([^/]*\).*:\1:'`
    echo ====================== ${dataset_name} =============================
    
    find ${dir} | grep token_stats.json$ | sort | while read stats; do
        model_name=`echo ${stats} | sed 's:.*mdl_nm=\([^/]*\).*:\1:'`
        echo ""
        echo ${model_name}
        jq . ${stats}
    done
done
