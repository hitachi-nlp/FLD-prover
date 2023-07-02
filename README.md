# FLD-Prover
This is one of the official repositories of the paper `Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic`.
This repository includes the code for generating the FLD corpus.  

See [the entry-point repository](https://github.com/hitachi-nlp/FLD) for the other repositories used in the paper.

## About this release
The model used in the paper was the step-wise prover of [the previous study](https://github.com/princeton-nlp/NLProofS), which is a little complex due to the code for the proof verifier.
For simplicity and ease-of-use, we have reimplemented the prover, `./run_prover.py`, which is the minimal adaptation from the HuggingFace [run_summarization.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py).

## Installation
The code has been tested on Python 3.8.5.
1. `pip install -r ./requirements.txt`
1. `git clone https://github.com/hitachi-nlp/FLD-task.git && pip install -e ./FLD-task`

## How to run

* table
    * scoring_disallow_any_proof_for_unknown=True/False


## About the metrics
As seen, we have defined the two types of the metrics,

* なぜ
    * Pros/Consを論じる．
* 備考
    - 論文ではTrue
    - ProofWriter original paper -> False




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
