# FLD-Prover
This is one of the official repositories of the paper [Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic](TODO).
This repository includes the code for the deductive prover model.  

See [the entry-point repository](https://github.com/hitachi-nlp/FLD) for the other repositories used in the paper.

## About this release
* The model used in the paper is the step-wise prover of [the previous study](https://github.com/princeton-nlp/NLProofS), which is a little complex due to the code for the proof verifier.
* For simplicity and ease of use, we have re-implemented a prover in`./run_prover.py`, which is a straightforward adaptation from the HuggingFace [run_summarization.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py).
* Besides the difference in implementation details, there is a difference in how to predict an answer label. Our re-implemented model predicts a label by generating a marker (`__PROVED__`/`__DISPROVED__`/`__UNKNOWN__`) at the end of a proof sequence, while the original model predicts an answer label by using another classifier on top of a generated proof sequence.

## Installation
The code has been tested on Python 3.8.5.
```console
# pip install -r ./requirements.txt
# git clone https://github.com/hitachi-nlp/FLD-task.git && pip install -e ./FLD-task
```

## How to run
1. Download the FLD corpus from [FLD-corpus](https://github.com/hitachi-nlp/FLD-corpus), or, create your own using [FLD-generator](https://github.com/hitachi-nlp/FLD-generator.git)
1. Convert json schema of the original corpus to fit the prover script:

    ```console
    $ python ./convert_json_schema.py ./data/FLD/FLD.3/train.jsonl  ./data/FLD.converted/FLD.3/train.jsonl
    $ python ./convert_json_schema.py ./data/FLD/FLD.3/valid.jsonl  ./data/FLD.converted/FLD.3/valid.jsonl
    $ python ./convert_json_schema.py ./data/FLD/FLD.3/test.jsonl   ./data/FLD.converted/FLD.3/test.jsonl
    ```

1. Train the prover:

    ```console
    $ python\
        ./run_prover.py\
        --output_dir outputs/\
        --logging_dir outputs/tensorboard/\
        --train_file ./data/FLD.converted/FLD.3/train.jsonl\
        --validation_file ./data/FLD.converted/FLD.3/valid.jsonl\
        --test_file ./data/FLD.converted/FLD.3/test.jsonl\
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
        --max_eval_samples 1000\
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

3. Check the results using tensorboard:

    ```console
    $ tensorboard --port 6006 --logdir ./outputs/tensorboard/
    ```

The performance will be somthing like:
|             | strct.D-all.answer_accuracy | strct.D-all.proof_accuracy | extr_stps.D-all.answer_accuracy | extr_stps.D-all.proof_accuracy |
|-------------|-----------------------------|----------------------------|---------------------------------|--------------------------------|
| 5000 steps  | TODO                          | TODO                         | TODO                              | TODO                             |
| 20000 steps | TODO                          | TODO                         | TODO                              | TODO                             |

where `strct|extr_stps` denotes the metrics type (see below) and `D-all` means the depth-aggregated results.


## About the metrics
As seen above, we have defined the two types of metrics:
* `strict` (originally used in the paper)
    * Do not allow any logical step generated by the model that is extra to the gold proof, even if the step is logically valid. Note that, due to the distractors, there are valid logical steps irrelevant to the gold proof.
* `extra_steps`
    * Allows such extra steps.

The difference in the two metrics is the most noticeable for a dataset instance with an `unknown` label, on which the `strict` metric allows the model to output only `__UNKNOWN__` marker while the `extra_steps` metric allows the model to output some logical steps to investigate whether the hypothesis can be (dis-) proved or not.

We think both metrics have their Pros/Cons and both are OK for use as long as they are not contaminated.
Note that the previous studies have used the metric colse to `extra_steps` regarding the `unknown` labels.

## Citation
```bibtex
@inproceedings{morishita2023FLD,
  title={Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic},
  author={Morishita, Terufumi and Morio, Gaku and Yamaguchi, Atsuki and Sogawa, Yasuhiro},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```
