#!/bin/bash


export PYTHONPATH=`pwd -P`:$PROJECTS/qsub-launcher:$PROJECTS/FLD/FLD-task/:$PROJECTS/FLD/rec-adam/:$PROJECTS/line-profiling/:$PROJECTS/script-engine/:$PROJECTS/lab:$PROJECTS:$PROJECTS/python-logger-setup:$PROJECTS/machine-learning-python/:$PROJECTS/FLD/FLD-user-shared-settings/:${PYTHONPATH}

HF_CACHE=`readlink -f ./outputs.lustre/hf_cache`
export HF_DATASETS_CACHE="${HF_CACHE}/hf_datatset_cache"
export HF_HOME="${HF_CACHE}/transformers_cache"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers_cache"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hf_hub_cache"
export SENTENCE_TRANSFORMERS_HOME="${HF_CACHE}/sentence_transformers_cache"




# check if hostname is like "es*.abci.local"
if [[ `hostname` =~ "es.*.abci.local|g[0-9]*" ]]; then

    module load cuda/11.8/11.8.0 cudnn/9.2/9.2.1

    source ${PROJECTS}/spack/share/spack/setup-env.sh  # load spack
    spack load openmpi@4.1.4 ^cuda@11.8.0

elif [[ `hostname` =~ "haicl|haicxh" ]]; then

    source /etc/profile.d/modules.sh
    module load cuda12.1.105_cudnn8.9.7_nccl2.18.3 openmpi-4.1.6

else
    echo "!!!!!!!!!!!! Unknown host"
fi


