# FLD-Prover
This repository includes the code to train and evaluate language models on FLD corpora.  

See [the entry-point repository](https://github.com/hitachi-nlp/FLD.git) about the whole FLD project.




## Release notes
* (2024-01-24) `NLP_2024_KOBE_BEEF` branch
    - Release at LREC-COLING 2024 and NLP(言語処理学会) 2024．
    - We made it possible to [Fine-tune LLMs](#fine-tune-llms), including both English and Japanese models.
    - Minor update on proof generation strategy: for the example with `UNKNOWN` label, we now generate only the label. Previously, we also generated the subproof in addition to the label, which was a bit unreasonable, as that subproof can not be distinguished from the noise proofs yielded from the distractors. The change of the strategy might affect the performance a litte.
    - **This branch might not be compatible with the older branches of relevant repositories.**
* (2023-08-22) `main` branch.
    - Initial release at ICML 2023.
    - Note the followings:
        * The model used in the paper is the step-wise prover of [the previous study](https://github.com/princeton-nlp/NLProofS), which comes with the code for the proof verifier. For simplicity and ease of use, we have implemented a simpler prover.
        * Besides the difference in implementation details, there is a difference in how to predict an answer label. Our re-implemented model predicts a label simply by generating a marker (`__PROVED__`/`__DISPROVED__`/`__UNKNOWN__`) at the end of a proof sequence, while the original model predicts an answer label by using another classifier on top of a generated proof sequence.




## Other Framework
FLD training is also accessible through a logical reasoning framework called [LogiTorch/logitorch](https://github.com/LogiTorch/logitorch).
Specifically, LogiTorch enables the training of an "all-at-once prover that generates an entire logical proof at once.
This prover differs from the original stepwise prover used in the paper and delivers slightly better performance.




## Installation
The code has been tested on Python 3.11.5
```console
$ pip install -r ./requirements/requirements.txt

$ git clone https://github.com/hitachi-nlp/FLD-task.git
$ cd FLD-task
$ git checkout NLP_2024_KOBE_BEEF
$ pip install -e .
$ cd ..

$ export PYTHONPATH=`pwd -P`:$PYTHONPATH
```




## Fine-tune T5
To train and evaluate the T5-based prover, which was used in the ICML paper, on **FLD** corpus hosted by [🤗 huggingface hub](https://huggingface.co/datasets/hitachi-nlp/FLD.v2):
```console
$ python ./run_prover.py \
    --dataset_name hitachi-nlp/FLD.v2 \
    --dataset_config_name default \
    --model_name_or_path t5-base \
    --output_dir outputs/ \
    --logging_dir outputs/tensorboard/ \
    --file_type json \
    --predict_with_generate True \
    --remove_unused_columns False \
    --do_train True \
    --do_eval True \
    --do_predict False \
    --seed 0 \
    --max_grad_norm 0.5 \
    --max_steps 20000 \
    --gradient_accumulation_steps 16 \
    --max_eval_samples 500 \
    --proof_sampling stepwise \
    --learning_rate 0.0001 \
    --warmup_steps 1000 \
    --source_prefix "Solve FLD task: " \
    --generation_num_beams 10 \
    --generation_top_k 10 \
    --generation_max_proof_steps 20 \
    --max_source_length 1700 \
    --max_target_length 100 \
    --logging_strategy steps \
    --logging_steps 25 \
    --overwrite_output_dir True \
    --log_generation True \
    --sample_negative_proof True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 0     \
    --log_examples True \
    --evaluation_strategy steps \
    --save_strategy steps  \
    --max_predict_samples 1000 \
    --eval_steps 5000 \
    --tokenizer_padding longest
```
or, if you want to use **FLD★**(FLD.4 in the paper), specify `--dataset_config_name star`.


If you have the datasets on your local filesystem, swap the `--dataset_name` option to the following:
```console
    --train_file ./data/FLD.v2/FLD.v2/train.jsonl \
    --validation_file ./data/FLD.v2/FLD.v2/valid.jsonl \
    --test_file ./data/FLD.v2/FLD.v2/test.jsonl \
```

After that, you can check the results by tensorboard as:
```console
$ tensorboard --port 6006 --logdir ./outputs/tensorboard/
```


## Fine-tune LLMs
LLMs are mostly encoder-only models, which can be trained by the other script as follows:
```console
python ./scripts/run_causal_prover.py  \
    --dataset_name hitachi-nlp/FLD.v2 \
    --dataset_config_name default \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output_dir outputs/ \
    --logging_dir outputs/tensorboard/ \
    --seed 0  \
    --max_grad_norm 0.5   \
    --max_steps 70  \
    --gradient_accumulation_steps 4  \
    --max_eval_samples 152  \
    --learning_rate 1e-05  \
    --warmup_steps 21  \
    --max_target_length 2000  \
    --logging_strategy steps  \
    --logging_steps 1  \
    --overwrite_output_dir True  \
    --no_subproof_for_unknown True  \
    --per_device_train_batch_size 2  \
    --per_device_eval_batch_size 4  \
    --dataloader_num_workers 0  \
    --log_examples True  \
    --max_train_samples 5  \
    --FLD_dataset_prob 1.0  \
    --FLD_max_eval_samples 150  \
    --eval_steps 70  \
    --remove_unused_columns False  \
    --instruction False  \
    --streaming False    \
    --evaluation_strategy steps  \
    --save_strategy no  \
    --save_model_at_end False  \
    --gradient_checkpointing True  \
    --block_size 2000  \
    --FLD_proof_eval_padding longest   \
    --generation_do_sample False  \
    --generation_temperature 1.0     \
    --generation_timeout 7200  \
    --evaluation_timeout 36000  \
    --do_train True  \
    --do_eval_in_outerloop False  \
    --do_predict False  \
    --fp16 True  \
    --lr_scheduler_type linear  \
    --weight_decay 0.0  \
    --lora False  \
    --use_auth_token
```

If you have the datasets on your local filesystem, swap the `--FLD_dataset_name` option to the following:
```console
    --FLD_train_file ./data/FLD.v2/FLD.v2/train.jsonl \
    --FLD_validation_file ./data/FLD.v2/FLD.v2/valid.jsonl \
```

After that, you can check the results by tensorboard as:
```console
$ tensorboard --port 6006 --logdir ./outputs/tensorboard/
```




## Performance and the metrics
The T5-based prover trained after 20000 steps on each corpus should perform as follows:

| corpus           | extr_stps.D-all.proof_accuracy | strct.D-all.proof_accuracy | D-all.answer_accuracy |
|------------------|--------------------------------|-----------------------------|-----------------------|
| **FLD** (FLD.3)  | 85.2                           | 75.8                        | 91.6                  |
| **FLD★**(FLD.4)   | 60.6                           |44.4                        | 72.2                  |

As seen above, we have defined the two types of metrics:
* `strict` (shown as `strct.*`. used in the paper.)
    * Do not allow any logical step generated by the model that is extra to the gold proof. Note that such a step can be logically valid because there are distractors that can lead to some valid logical steps irrelevant to the gold proof.
* `extra_steps` (shown as `extr_stps.*`)
    * Allows such extra steps.

The difference in the two metrics is the most noticeable for a dataset instance with an `unknown` label, on which the `strict` metric allows the model to output only `__UNKNOWN__` marker while the `extra_steps` metric allows the model to output some logical steps to investigate whether the hypothesis can be (dis-) proved or not.

We think both metrics have their Pros/Cons and both are OK for use as long as they are not contaminated.
Note that the previous studies have used the metric colse to `extra_steps` regarding the `unknown` labels.
