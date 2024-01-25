# FLD-Prover
This repository includes the code to train and evaluate language models on FLD corpora.  

See [the entry-point repository](https://github.com/hitachi-nlp/FLD.git) about the whole FLD project.




## Releases (READ CAREFULLY to determine which branch suits you)
* **`NLP_2024_KOBE_BEEF`** branch (2024-01-24) 
    - Released at LREC-COLING 2024 and NLP (言語処理学会) 2024.
    - **We made it possible to [Fine-tune LLMs](#fine-tune-llms), including both English and Japanese models.**
    - Minor update on the proof generation strategy: For examples with the UNKNOWN label, we now generate only the label. Previously, in addition to the label, we also generated a subproof, which was somewhat unreasonable since this subproof could not be distinguished from the noise proofs yielded by the distractors. This change in strategy might slightly affect performance.
    - **This branch might not be compatible with the older branches of relevant repositories.**
* **`main`** branch (2023-08-22)
    - Initial release at ICML 2023.
    - Note that the prover implemented in this repository is slightly different from the one used in the original ICML paper, as follows:
        * The model used in the paper is the step-wise prover of [the previous study](https://github.com/princeton-nlp/NLProofS), which comes with the code for the proof verifier. For simplicity and ease of use, we have implemented a simpler prover.
        * Besides the difference in implementation details, there is a difference in how to predict an answer label. Our re-implemented model predicts a label simply by generating a marker (`__PROVED__`/`__DISPROVED__`/`__UNKNOWN__`) at the end of a proof sequence, while the original model predicts an answer label by using another classifier on top of a generated proof sequence.




## Other Framework
FLD training is also accessible through a logical reasoning framework called [LogiTorch/logitorch](https://github.com/LogiTorch/logitorch).
Specifically, LogiTorch enables the training of an "all-at-once prover that generates an entire logical proof at once.
This prover differs from the original stepwise prover used in the paper and delivers slightly better performance.

## Installation
The code has been tested on Python 3.8.5.
```console
$ pip install -r ./requirements.txt
$ git clone https://github.com/hitachi-nlp/FLD-task.git && pip install -e ./FLD-task
$ export PYTHONPATH=`pwd -P`:$PYTHONPATH
```

## How to train a prover

1. Run the training script. We use the **FLD** (FLD.3 in the paper) corpus hosted by [🤗 huggingface hub](https://huggingface.co/datasets/hitachi-nlp/FLD.v2):

    ```console
    $ python\
        ./run_prover.py\
        --dataset_name hitachi-nlp/FLD.v2\
        --dataset_config_name default\
        --output_dir outputs/\
        --logging_dir outputs/tensorboard/\
        --file_type json\
        --predict_with_generate True\
        --remove_unused_columns False\
        --do_train True\
        --do_eval True\
        --do_predict False\
        --seed 0\
        --max_grad_norm 0.5\
        --max_steps 20000\
        --gradient_accumulation_steps 16\
        --max_eval_samples 500\
        --proof_sampling stepwise\
        --learning_rate 0.0001\
        --warmup_steps 1000\
        --model_name_or_path t5-base\
        --source_prefix "Solve FLD task: "\
        --generation_num_beams 10\
        --generation_top_k 10\
        --generation_max_proof_steps 20\
        --max_source_length 1700\
        --max_target_length 100\
        --logging_strategy steps\
        --logging_steps 25\
        --overwrite_output_dir True\
        --log_generation True\
        --sample_negative_proof True\
        --per_device_train_batch_size 1\
        --per_device_eval_batch_size 1\
        --dataloader_num_workers 0    \
        --log_examples True\
        --evaluation_strategy steps\
        --save_strategy steps \
        --max_predict_samples 1000\
        --eval_steps 5000\
        --tokenizer_padding longest
    ```
    or, if you want to use **FLD★**(FLD.4 in the paper), specify `--dataset_config_name star`.

    If you have the datasets on your local filesystem, swap the `--dataset_name` option to the following:
    ```console
        --train_file ./data/FLD.v2/FLD.v2/train.jsonl\
        --validation_file ./data/FLD.v2/FLD.v2/valid.jsonl\
        --test_file ./data/FLD.v2/FLD.v2/test.jsonl\
    ```


1. Check the results using tensorboard:

    ```console
    $ tensorboard --port 6006 --logdir ./outputs/tensorboard/
    ```

## Prover performance and the metrics
A prover trained for 20000 steps on each corpus should perform as follows:

| corpus           | extr_stps.D-all.proof_accuracy | strct.D-all.proof_accuracy | D-all.answer_accuracy |
|------------------|--------------------------------|-----------------------------|-----------------------|
| **FLD** (FLD.3)  | 85.2                           | 75.8                        | 91.6                  |
| **FLD★**(FLD.4)   | 60.6                           |44.4                        | 72.2                  |

As seen above, we have defined the two types of metrics:
* `strict` (shown as `strct.*`. used in the paper.)
    * Do not allow any logical step generated by the model that is extra to the gold proof. Note that such a step can be logically valid because there are distractors that can lead to some valid logical steps irrelevant to the gold proof.
* `extra_steps` (shown as `extr_stps.*`)
    * Allows such extra steps.

The difference in the two metrics is the most noticeable for a dataset instance with an `unknown` label, on which the `strict` metric allows the model to output *only* `__UNKNOWN__` marker while the `extra_steps` metric allows the model to output some logical steps to investigate whether the hypothesis can be (dis-) proved or not.

We think both metrics have their Pros/Cons and both are OK for use as long as they are not contaminated.
Note that the previous studies have used the metric colse to `extra_steps` regarding the `unknown` labels.
