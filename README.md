# FLD-Prover
This repository includes the code to train and evaluate large language models on FLD corpora.  

See [the entry-point repository](https://github.com/hitachi-nlp/FLD.git) about the whole FLD project.




## Release Branches (READ CAREFULLY to determine which branch suits you)
* **(New!)** `NeurIPS_2024` branch (2024-12)
    - We released the code for training LLMs.
* `NLP_2024_KOBE_BEEF` branch (2024-01-24) 
    - Release at NLP (言語処理学会) 2024.
    - We made it possible to [Fine-tune LLMs](#fine-tune-llms), including both English and Japanese models.
    - Minor update on the proof generation strategy: For examples with the UNKNOWN label, we now generate only the label. Previously, in addition to the label, we also generated a subproof, which was somewhat unreasonable since this subproof could not be distinguished from the noise proofs yielded by the distractors. This change in strategy might slightly affect performance.
    - This branch might not be compatible with the older branches of relevant repositories.
* `main` branch (2023-08-22)
    - Initial release at ICML 2023.
    - Note that the prover implemented in this repository is slightly different from the one used in the original ICML paper, as follows:
        * The model used in the paper is the step-wise prover of [the previous study](https://github.com/princeton-nlp/NLProofS), which comes with the code for the proof verifier. For simplicity and ease of use, we have implemented a simpler prover.
        * Besides the difference in implementation details, there is a difference in how to predict an answer label. Our re-implemented model predicts a label simply by generating a marker (`__PROVED__`/`__DISPROVED__`/`__UNKNOWN__`) at the end of a proof sequence, while the original model predicts an answer label by using another classifier on top of a generated proof sequence.




## Installation
The code has been tested on Python 3.11.5

```console
# [!] First, prepare CUDA libraries and set correct environmental variables,
# as some of the modules , such as torch, can only be built under the environment.

pip install --upgrade pip

# Install pytorch. This is just an example with CUDA 11.8.,
# and users have to install torch respecting THEIR OWN CUDA versions.
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install -r ./requirements/requirements.txt

# Additional package for data preprocessing.
git clone https://github.com/hitachi-nlp/FLD-task.git
cd FLD-task
git checkout <taget branch>
pip install -e .
cd ..

export PYTHONPATH=`pwd -P`:$PYTHONPATH
```




## How to Train LLMs

### Using Your own Script
[Our training script](#using_our_script) is becoming a bit complicated, so it could be better to use your own script, roughly as follows:
1. Prepare the corpus.
    * You can simply use the released version of [FLDx2 (FLD Diverse) 🤗](https://huggingface.co/datasets/hitachi-nlp/FLDx2).
    * (Optional) Or, you can create your own corpus by [FLD-generator](https://github.com/hitachi-nlp/FLD-generator).
2. Modify your training script as follows:
    * Use `prompt_serial` field of the corpus for LLM's input, and `proof_serial` for the output.
    * DO MASK the LLM's input, meaning that we do not use the input for loss computation, similarly to supervised fine-tuning. This prevents LLMs from memorizing unknown facts included in the corpus.
    * Use [Rcall Adam Optimizer](https://github.com/hitachi-nlp/rec-adam) to, again, prevent LLMs from memorizing unknown facts.

For the other details, please refer to our paper.

### Using Our Script
The training script is `./scripts/run_causal_prover.py`.

To train Llama-3.1-8B on [FLDx2 (FLD Diverse) 🤗](https://huggingface.co/datasets/hitachi-nlp/FLDx2), run the following command:
```console
python ./scripts/run_causal_prover.py \
    --output_dir {output_dir} \
    --logging_dir {log_dir} \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --logic_dataset_name hitachi-nlp/FLDx2 \
    --use_original_serial True \
    --proof_intermediate_steps_prob 0.5 \
    --seed 0 \
    --learning_rate 2e-05 \
    --warmup_steps 200 \
    --max_steps 390 \
    --eval_steps 390 \
    --optimizer rec_adam \
    --rec_adam_fisher_coef 4000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --max_eval_samples 10000 \
    --block_size 2048 \
    --logging_strategy steps \
    --logging_steps 5 \
    --overwrite_output_dir True \
    --log_examples True \
    --logic_dataset_prob 1.0 \
    --logic_eval_max_samples 100 \
    --remove_unused_columns False \
    --streaming False \
    --evaluation_strategy steps \
    --save_model_at_end False \
    --save_only_model True \
    --save_steps 390 \
    --save_strategy steps \
    --save_total_limit 2 \
    --generation_temperature 1.0 \
    --generation_max_prompt_length 1300 \
    --generation_timeout 7200 \
    --evaluation_timeout 36000 \
    --do_train True \
    --do_predict False \
    --fp16 False \
    --bf16 True \
    --preprocessing_num_workers 1 \
    --preprocess_batch_size 500 \
    --ddp_timeout 36000 \
    --use_auth_token True
```

If you use huggingface's deepspeed integration, you can modify the command as something like as follows, depending on your environment:
```console
deepspeed \
    --master_addr {hostname} \
    --master_port {port} \
    --hostfile {hostfile} \
    --no_ssh_check \
    --launcher OpenMPI \
    --launcher_args "-mca coll ^hcoll --oversubscribe" \
    ./scripts/run_causal_prover.py \
    --deepspeed ds_config/ds_config_zero3_wo_optimizer_offload.json \
    {other_options}
```

Additionally, if you have corpora on your local filesystem, swap the `--logic_dataset_name` option to the following:
```console
    --logic_train_file {train_jsonl_path} \
    --logic_validation_file {validation_jsonl_path} \
    --logic_test_file {test_jsonl_path} \
```

After launching the script, you can check the results by tensorboard as:
```console
tensorboard --port <your_port> --logdir ./outputs/tensorboard/
```
