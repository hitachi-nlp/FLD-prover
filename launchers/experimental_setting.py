import logging
from typing import Any, Dict, Optional, Union, List, Tuple
import glob
from pathlib import Path
import json
import math

from pydantic import BaseModel
from lab import build_dir
from script_engine import QsubEngine, SubprocessEngine
from script_engine.base import EngineBase
# from pytorch_lightning import Trainer


logger = logging.getLogger(__name__)


def maybe_option_value(option: str, value: Any) -> str:
    if value is None:
        return ''
    else:
        if isinstance(value, str) and value.find(' ') >= 0:
            return f'{option} "{value}"'
        else:
            return f'{option} {value}'


def maybe_option_flag(flag: str, value: bool) -> str:
    if not isinstance(value, bool):
        raise ValueError()
    return f'{flag} {str(value)}'


_PROVER_BATCH_SETTINGS = {

    'google/long-t5-tglobal-base': {
        'max_source_length': 1500,
        'max_target_length': 100,

        'per_device_train_batch_size': 10,
        'per_device_eval_batch_size': 10,

        # 'tokenizer_padding': 'max_length',
        'tokenizer_padding': 'longest',
    },

    'google/long-t5-tglobal-large': {
        'max_source_length': 1500,
        'max_target_length': 100,

        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,

        # 'tokenizer_padding': 'max_length',
        'tokenizer_padding': 'longest',
    },

    # XXX: if you change max_source_length or max_target_length,
    # make sure that all the stuf fit into memory with tokenizer_padding='max_len' option.
    # The 'max_len' option guarantee that the model always use the max_len inputs without truncation
    # thus, we can measure the maxmum usage of memory.

    # only for fast experiments on small dataset like depth-3
    't5-small': {
        'max_source_length': 1000,
        'max_target_length': 100,

        'per_device_train_batch_size': 32,
        'per_device_eval_batch_size': 32,

        # 'tokenizer_padding': 'max_length',
        'tokenizer_padding': 'longest',
    },

    't5-base': {
        # 'max_source_length': 1500,
        'max_source_length': 1700,
        'max_target_length': 100,

        # -- V100 --
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,

        # -- A100 --
        # 'per_device_train_batch_size': 4,
        # 'per_device_eval_batch_size': 4,

        # 'tokenizer_padding': 'max_length',
        'tokenizer_padding': 'longest',
    },

    # 't5-large': {
    #     'max_source_length': 1200,
    #     'max_target_length': 100,

    #     'per_device_train_batch_size': 3,
    #     'per_device_eval_batch_size': 3,

    #     # 'tokenizer_padding': 'max_length',
    #     'tokenizer_padding': 'longest',
    # },

    't5-large': {
        'max_source_length': 1500,
        'max_target_length': 100,

        # # XXX: may cause being killed by qsub manager ???
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,

        # 'per_device_train_batch_size': 1,

        # 'tokenizer_padding': 'max_length',
        'tokenizer_padding': 'longest',
    },

    # # TODO must tune the max_source_length and batch size for speed
    # 'allenai/led-base-16384': {
    #     'max_source_length': 1500,
    #     'max_target_length': 100,

    #     'per_device_train_batch_size': 8,
    #     'per_device_eval_batch_size': 8,

    #     # 'tokenizer_padding': 'max_length',
    #     'tokenizer_padding': 'longest',
    # },

    # # TODO must tune the max_source_length and batch size for speed
    # 'allenai/led-large-16384': {
    #     'max_source_length': 1500,
    #     'max_target_length': 100,

    #     'per_device_train_batch_size': 8,
    #     'per_device_eval_batch_size': 8,

    #     # 'tokenizer_padding': 'max_length',
    #     'tokenizer_padding': 'longest',
    # },

}

_VERIFIER_BATCH_SETTINGS = {
    'roberta-base': {
        # 'max_source_length': 400,
        'max_source_length': 300,
        'per_device_train_batch_size': 16,
    },

    'roberta-large': {
        # 'max_source_length': 400,
        'max_source_length': 300,

        'per_device_train_batch_size': 16,

        'tokenizer_padding': 'max_length',
        # 'tokenizer_padding': 'longest',
    },
}


def get_batch_setting(model_name: str,
                      num_gpus: int,
                      train_effective_batch_size=64) -> Dict[str, Any]:
    setting = _PROVER_BATCH_SETTINGS[model_name]
    accum_steps = int(train_effective_batch_size / (setting['per_device_train_batch_size'] * num_gpus))
    if accum_steps < 1:
        raise ValueError()
    setting['gradient_accumulation_steps'] = accum_steps
    return setting


_DATASET_PATHS = {
    '20221120.negative_tree.debug': {
        'train_file': './outputs.FLNL/10.create_FLNL_corpus/20221120.negative_tree/local_dataset_name=20221120.negative_tree__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-1000/rsd_objct_nns_mx_fctr=1.0/smpl_hrd_ngtvs=True/try_ngtd_hypthss_frst=False/us_fxd_trnsltn=False/test/test.jsonl',
        'validation_file': './outputs.FLNL/10.create_FLNL_corpus/20221120.negative_tree/local_dataset_name=20221120.negative_tree__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-1000/rsd_objct_nns_mx_fctr=1.0/smpl_hrd_ngtvs=True/try_ngtd_hypthss_frst=False/us_fxd_trnsltn=False/test/test.jsonl',
        'test_file': './outputs.FLNL/10.create_FLNL_corpus/20221120.negative_tree/local_dataset_name=20221120.negative_tree__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-1000/rsd_objct_nns_mx_fctr=1.0/smpl_hrd_ngtvs=True/try_ngtd_hypthss_frst=False/us_fxd_trnsltn=False/test/test.jsonl',
    },




    'ruletaker': {
        'train_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-3ext/meta-train.jsonl',
        'validation_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-3ext/meta-dev.jsonl',
        'test_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-3ext/meta-test.jsonl',

    },


    'ruletaker.include_all_answers': {
        'train_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext/meta-train.jsonl',
        'validation_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext/meta-dev.jsonl',
        'test_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext/meta-test.jsonl',
    },

    'ruletaker.D5.include_all_answers': {
        'train_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-5/meta-train.jsonl',
        'validation_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-5/meta-dev.jsonl',
        'test_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-5/meta-test.jsonl',
    },

    'ruletaker.NatLang.include_all_answers': {
        'train_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext-NatLang/meta-train.jsonl',
        'validation_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext-NatLang/meta-dev.jsonl',
        'test_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext-NatLang/meta-test.jsonl',
    },

    'ruletaker.birds-electricity.include_all_answers': {
        'train_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/birds-electricity/meta-test.jsonl',
        'validation_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/birds-electricity/meta-test.jsonl',
        'test_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/birds-electricity/meta-test.jsonl',
    },


    'ruletaker.include_all_answers.unknown_with_collapsed_proof': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.include_all_answers/sd=0/unk_rt=0.333/train.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.include_all_answers/sd=0/unk_rt=0.333/val.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.include_all_answers/sd=0/unk_rt=0.333/test.jsonl',
    },

    'ruletaker.NatLang.include_all_answers.unknown_with_collapsed_proof': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.NatLang.include_all_answers/sd=0/unk_rt=0.333/train.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.NatLang.include_all_answers/sd=0/unk_rt=0.333/val.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.NatLang.include_all_answers/sd=0/unk_rt=0.333/test.jsonl',
    },

    'ruletaker.birds-electricity.include_all_answers.unknown_with_collapsed_proof': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.birds-electricity.include_all_answers/sd=0/unk_rt=0.333/train.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.birds-electricity.include_all_answers/sd=0/unk_rt=0.333/val.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.birds-electricity.include_all_answers/sd=0/unk_rt=0.333/test.jsonl',
    },


    'EB-task1': {
        'train_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl',
        'validation_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_1/dev.jsonl',
        'test_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_1/test.jsonl',
    },

    'EB-task2': {
        'train_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl',
        'validation_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl',
        'test_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_2/test.jsonl',
    },

    'EB-task3': {
        'train_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.jsonl',  # The NLProofS paper uses task2's data for task3.
        'validation_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_3/dev.jsonl',
        'test_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_3/test.jsonl',
    },






    'ruletaker.shuffled': {
        'train_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-3ext/meta-train.shuffled.jsonl',
        'validation_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-3ext/meta-dev.shuffled.jsonl',
        'test_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA/depth-3ext/meta-test.shuffled.jsonl',

    },


    'ruletaker.include_all_answers.shuffled': {
        'train_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext/meta-train.shuffled.jsonl',
        'validation_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext/meta-dev.shuffled.jsonl',
        'test_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext/meta-test.shuffled.jsonl',
    },

    'ruletaker.NatLang.include_all_answers.shuffled': {
        'train_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext-NatLang/meta-train.shuffled.jsonl',
        'validation_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext-NatLang/meta-dev.shuffled.jsonl',
        'test_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/depth-3ext-NatLang/meta-test.shuffled.jsonl',
    },

    'ruletaker.birds-electricity.include_all_answers.shuffled': {
        'train_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/birds-electricity/meta-test.shuffled.jsonl',
        'validation_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/birds-electricity/meta-test.shuffled.jsonl',
        'test_file': './data/proofwriter-dataset-V2020.12.3/preprocessed_OWA.include_all_answers/birds-electricity/meta-test.shuffled.jsonl',
    },


    'ruletaker.include_all_answers.unknown_with_collapsed_proof.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.include_all_answers/sd=0/unk_rt=0.333/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.include_all_answers/sd=0/unk_rt=0.333/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.include_all_answers/sd=0/unk_rt=0.333/test.shuffled.jsonl',
    },

    'ruletaker.NatLang.include_all_answers.unknown_with_collapsed_proof.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.NatLang.include_all_answers/sd=0/unk_rt=0.333/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.NatLang.include_all_answers/sd=0/unk_rt=0.333/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.NatLang.include_all_answers/sd=0/unk_rt=0.333/test.shuffled.jsonl',
    },

    'ruletaker.birds-electricity.include_all_answers.unknown_with_collapsed_proof.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.birds-electricity.include_all_answers/sd=0/unk_rt=0.333/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.birds-electricity.include_all_answers/sd=0/unk_rt=0.333/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221129.wo_hypothesis/local_dataset_name=ruletaker.birds-electricity.include_all_answers/sd=0/unk_rt=0.333/test.shuffled.jsonl',
    },



    'ruletaker.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/test.shuffled.jsonl',
    },

    'ruletaker.D5.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.D5.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=None/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.D5.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=None/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.D5.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=None/test.shuffled.jsonl',
    },

    'ruletaker.NatLang.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.NatLang.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.NatLang.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.NatLang.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/test.shuffled.jsonl',
    },

    'ruletaker.birds-electricity.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.birds-electricity.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.birds-electricity.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.birds-electricity.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/test.shuffled.jsonl',
    },



    'ruletaker.ours.20221202': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/test.shuffled.jsonl',
    },

    'ruletaker.D5.ours.20221202': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.D5.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=None/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.D5.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=None/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.D5.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=None/test.shuffled.jsonl',
    },

    'ruletaker.NL.ours.20221202': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.NatLang.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.NatLang.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.NatLang.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/test.shuffled.jsonl',
    },

    'ruletaker.BE.ours.20221202': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.birds-electricity.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.birds-electricity.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.birds-electricity.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.0/mx_dpth=3/test.shuffled.jsonl',
    },




    'ruletaker.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.5.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.5/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.5/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.5/test.shuffled.jsonl',
    },

    'ruletaker.NatLang.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.5.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.NatLang.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.5/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.NatLang.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.5/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.NatLang.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.5/test.shuffled.jsonl',
    },

    'ruletaker.birds-electricity.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.5.shuffled': {
        'train_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.birds-electricity.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.5/train.shuffled.jsonl',
        'validation_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.birds-electricity.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.5/val.shuffled.jsonl',
        'test_file': './outputs/00.create_unknown_with_collapsed_proof_corpus.py/20221202.negative_proof/local_dataset_name=ruletaker.birds-electricity.include_all_answers/unk_ratio=0.333/reference_unknown_proof_ratio=0.3/negative_proof_prob=0.5/test.shuffled.jsonl',
    },




    # XXX do not shuffle the test and validation split since the scorer referes to the unshuffled versions
    'EB-task1.shuffled': {
        'train_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.shuffled.jsonl',
        'validation_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_1/dev.jsonl',
        'test_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_1/test.jsonl',
    },

    'EB-task2.shuffled': {
        'train_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.shuffled.jsonl',
        'validation_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_2/dev.jsonl',
        'test_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_2/test.jsonl',
    },

    'EB-task3.shuffled': {
        'train_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_2/train.shuffled.jsonl',  # The NLProofS paper uses task2's data for task3.
        'validation_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_3/dev.jsonl',
        'test_file': './data/entailment_trees_emnlp2021_data_v3/dataset/task_3/test.jsonl',
    },


    'FLD.debug.2023-05-13': {
        'train_file': './outputs/00.fix_FLD_schema.py/2023-05-15/20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP/test.jsonl',
        'validation_file': './outputs/00.fix_FLD_schema.py/2023-05-15/20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP/test.jsonl',
        'test_file': './outputs/00.fix_FLD_schema.py/2023-05-15/20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP/test.jsonl',
    },

}


def get_dataset_paths(name: str,
                      top_dirs: List[str],
                      use_test_as_train=False,
                      use_test_as_val=False) -> Dict[str, Optional[str]]:
    if name in _DATASET_PATHS:
        paths = _DATASET_PATHS[name].copy()

        if use_test_as_train:
            paths['train_file'] = paths['test_file']

        if use_test_as_val:
            paths['validation_file'] = paths['test_file']

        return paths

    else:

        def get_split_jsonl(top_dir: Path, split: str) -> Optional[Path]:
            all_paths = glob.glob(str(top_dir) + '/**/*', recursive=True)
            paths = [path for path in all_paths
                     if path.endswith(f'{split}.jsonl') and str(path).find('job-') < 0]
            if len(paths) == 0:
                return None
            elif len(paths) == 1:
                return paths[0]
            else:
                raise ValueError(f'multiple dataset file found under {str(top_dir)} as:\n'
                                 '\n'.join([str(path) for path in paths]))

        def get_split_jsonl_with_warning(top_dir: Path, local_dataset_name: str, split: str) -> Optional[Path]:
            path = get_split_jsonl(top_dir, split)
            if path is None:
                logger.warning('dataset split="%s" name="%s" not found under "%s"',
                               split,
                               local_dataset_name,
                               str(top_dir))
            return path

        # We must use glob to follow symbolic links
        found_dataset_paths = None
        found_lab_path = None
        for top_dir in top_dirs:

            for lab_path in glob.glob(top_dir + '/**/*', recursive=True):

                if not lab_path.endswith('lab.params.json'):
                    continue
                if lab_path.find('job-') >= 0:
                    continue

                setting = json.load(open(lab_path))

                if setting.get('dataset_name', None) != name:
                    continue

                if found_lab_path is not None:
                    raise Exception(f'Multiple files for dataset "{name}" are found:\n1. "{str(found_lab_path)}"\n2. "{str(lab_path)}"')

                lab_path = Path(lab_path)
                found_lab_path = lab_path

                train_path = get_split_jsonl_with_warning(lab_path.parent, name, 'train')
                valid_path = get_split_jsonl_with_warning(lab_path.parent, name, 'valid')
                test_path = get_split_jsonl_with_warning(lab_path.parent, name, 'test')

                if use_test_as_train:
                    train_path = test_path
                if use_test_as_val:
                    valid_path = test_path

                found_dataset_paths = {
                    'train_file': str(train_path) if train_path is not None else None,
                    'validation_file': str(valid_path) if valid_path is not None else None,
                    'test_file': str(test_path) if test_path is not None else None,
                }

            if found_dataset_paths is not None:
                return found_dataset_paths

    raise ValueError(f'Dataset {name} not found under {top_dirs}.')


def get_dataset_setting(name: str) -> Dict[str, Any]:
    if name.startswith('FLD'):
        return {
            'file_type': 'json',
            'predict_with_generate': True,
            'remove_unused_columns': False,
        }
    elif name.startswith('20221203.first_exp'):
        return {
            'file_type': 'json',
            'predict_with_generate': True,
            'remove_unused_columns': False,
        }
    elif name.startswith('2023'):
        return {
            'file_type': 'json',
            'predict_with_generate': True,
            'remove_unused_columns': False,
        }
    else:
        raise NotImplementedError()


# _PROVER_CONFIGS = {
#     # -- official --
#     # 'ruletaker'   : './configs/prover/cli_ruletaker_stepwise_t5-large.yaml',
#     # 'EB-task1'   : './configs/prover/cli_task1_stepwise_t5-large.yaml',
#     # 'EB-task2': './configs/prover/cli_task2_stepwise_t5-large.yaml',
# 
#     # -- (old) aacorpus --
#     # 'aacorpus_20220702.trial': './configs/prover/cli_aacorpus_20220702.trial_stepwise_t5-large.yaml',
#     # 'aacorpus_20220707.small': './configs/prover/cli_aacorpus_20220707.small_stepwise_t5-large.yaml',
# 
#     # -- our setting --
#     # 'ruletaker.20220922.base': './configs/prover/ruletaker.20220922.base.yaml',
# 
#     'FLNLcorpus.20220827.base': './configs/prover/FLNLcorpus.20220827.base.yaml',
# 
#     'ruletaker.include_all_answers.20220922.base': './configs/prover/ruletaker.include_all_answers.20220922.base.yaml',
#     'ruletaker.D5.include_all_answers.20220922.base': './configs/prover/ruletaker.include_all_answers.20220922.base.yaml',
#     'ruletaker.NatLang.include_all_answers.20220922.base': './configs/prover/ruletaker.NatLang.include_all_answers.20220922.base.yaml',
#     'ruletaker.birds-electricity.include_all_answers.20220922.base': './configs/prover/ruletaker.birds-electricity.include_all_answers.20220922.base.yaml',
# 
#     'EB-task1.20220922.base': './configs/prover/EB-task1.20220922.base.yaml',
#     'EB-task2.20220922.base': './configs/prover/EB-task2.20220922.base.yaml',
#     'EB-task3.20220922.base': './configs/prover/EB-task3.20220922.base.yaml',
# }


_PROVER_CONFIGS = {
    # -- official --
    # 'ruletaker'   : './configs/prover/cli_ruletaker_stepwise_t5-large.yaml',
    # 'EB-task1'   : './configs/prover/cli_task1_stepwise_t5-large.yaml',
    # 'EB-task2': './configs/prover/cli_task2_stepwise_t5-large.yaml',

    # -- (old) aacorpus --
    # 'aacorpus_20220702.trial': './configs/prover/cli_aacorpus_20220702.trial_stepwise_t5-large.yaml',
    # 'aacorpus_20220707.small': './configs/prover/cli_aacorpus_20220707.small_stepwise_t5-large.yaml',

    # -- our setting --
    # 'ruletaker.20220922.base': './configs/prover/ruletaker.20220922.base.yaml',

    'FLNLcorpus.20220827.base': {
        'seed': 42,
        # 'n_gpu': 1,
        'max_grad_norm': 0.5,
        # 'num_train_epochs': 10,
        'max_steps': 10000,
        'gradient_accumulation_steps': 11,
        # 'check_val_every_n_epoch': 1,
        'max_eval_samples': 2000,
        'proof_sampling': 'stepwise',
        'max_proof_steps': 30,
        'learning_rate': 1e-4,
        'warmup_steps': 1000,
        'model_name_or_path': 't5-large',
        # 'fp16': True,

        'source_prefix': 'Solve FLD task: ',
        'generation_num_beams': 10,
        'generation_top_k': 10,
        'generation_max_proof_steps': 20,
        # 'verifier_ckpt': "",
        # 'verifier_weight': 0,
        # 'proof_search': False,
        # 'oracle_prover': False,
        # 'oracle_verifier': False,
        'max_source_length': 512,
        # 'tokenizer_padding': longest,
        'max_target_length': 64,

        'logging_strategy': 'steps',
        'logging_steps': 25,
        'overwrite_output_dir': True,

        'log_generation': True,

        # 'dataset': FLNL,
        # 'path_train': specify_me,
        # 'path_val': specify_me,
        # 'path_test': specify_me,

        # 'exclude_unknown': False,
        # 'add_final_reference_to_proofs': False,
        # 'sample_goal': intermediates # hypothesis | intermediates,
        # 'subtree_proved_prob': 0.75,
        # 'subtree_proved_all_or_none': False,
        'sample_negative_proof': False,
        'per_device_train_batch_size': 3,
        'per_device_eval_batch_size': 3,
        'dataloader_num_workers': 2,
        # 'tokenizer_padding': longest,
        # 'shuffle_train': True,

        'dataset_1': None,
        'path_train_1': None,
        'path_val_1': None,
        'path_test_1': None,

        'log_examples': False,
        # 'no_lower': True,

        'evaluation_strategy': 'steps',
        'save_strategy': 'steps',

        'logging_strategy': 'steps',

    },

    # 'ruletaker.include_all_answers.20220922.base': './configs/prover/ruletaker.include_all_answers.20220922.base.yaml',
    # 'ruletaker.D5.include_all_answers.20220922.base': './configs/prover/ruletaker.include_all_answers.20220922.base.yaml',
    # 'ruletaker.NatLang.include_all_answers.20220922.base': './configs/prover/ruletaker.NatLang.include_all_answers.20220922.base.yaml',
    # 'ruletaker.birds-electricity.include_all_answers.20220922.base': './configs/prover/ruletaker.birds-electricity.include_all_answers.20220922.base.yaml',

    # 'EB-task1.20220922.base': './configs/prover/EB-task1.20220922.base.yaml',
    # 'EB-task2.20220922.base': './configs/prover/EB-task2.20220922.base.yaml',
    # 'EB-task3.20220922.base': './configs/prover/EB-task3.20220922.base.yaml',
}


_VERIFIER_CONFIGS = {
    # -- official --
    # 'ruletaker': './configs/verifier/cli_ruletaker.yaml',
    # 'EB-task1': './configs/verifier/cli_entailmentbank_task1.yaml',
    # 'EB-task2': './configs/verifier/cli_entailmentbank_task2.yaml',

    # -- our setting --
    # 'ruletaker.20220922.base': './configs/verifier/ruletaker.20220922.base.yaml',

    'FLNLcorpus.20220827.base': './configs/verifier/FLNLcorpus.20220827.base.yaml',
    'ruletaker.include_all_answers.20220922.base': './configs/verifier/ruletaker.include_all_answers.20220922.base.yaml',
    'EB-task1.20220922.base': './configs/verifier/EB-task1.20220922.base.yaml',
    'EB-task2.20220922.base': './configs/verifier/EB-task2.20220922.base.yaml',
    'EB-task3.20220922.base': './configs/verifier/EB-task3.20220922.base.yaml',
}


def get_default_config_name(local_dataset_name: str) -> str:
    if local_dataset_name.find('ruletaker.include_all_answers') >= 0:
        return 'ruletaker.include_all_answers.20220922.base'
    elif local_dataset_name.find('ruletaker.D5.include_all_answers') >= 0:
        return 'ruletaker.D5.include_all_answers.20220922.base'
    elif local_dataset_name.find('ruletaker.NatLang.include_all_answers') >= 0:
        return 'ruletaker.NatLang.include_all_answers.20220922.base'
    elif local_dataset_name.find('ruletaker.birds-electricity.include_all_answers') >= 0:
        return 'ruletaker.birds-electricity.include_all_answers.20220922.base'
    elif local_dataset_name.find('ruletaker') >= 0:
        return 'ruletaker.20220922.base'
    elif local_dataset_name.find('EB-task1') >= 0:
        return 'EB-task1.20220922.base'
    elif local_dataset_name.find('EB-task2') >= 0:
        return 'EB-task2.20220922.base'
    elif local_dataset_name.find('EB-task3') >= 0:
        return 'EB-task3.20220922.base'
    else:
        return 'FLNLcorpus.20220827.base'


def get_config(name: str) -> Dict[str, Any]:
    return _PROVER_CONFIGS[name]


_PROVER_CHECKPOINTS = {
    # official
    'ruletaker.NLProofS'   : './checkpoints/RuleTaker/NLProofS/prover/epoch=19-step=16940.ckpt',

    'aacorpus_20220702.trial': './outputs/10.train.py/20220707.aacorpus/task=aacorpus_20220702.trial/component=prover/checkpoint_type=None/btch_sz_pr_gp=None/lr=None/mx_epchs=3/mdl_nm=None/lightning_logs/version_0/checkpoints/epoch=2-step=3.ckpt',
    'aacorpus_20220707.small': './outputs/10.train.py/aacorpus_20220707.small/task=aacorpus_20220707.small/component=prover/checkpoint_type=None/btch_sz_pr_gp=None/lr=None/mx_epchs=10/mdl_nm=None/lightning_logs/version_0/checkpoints/epoch=3-step=956.ckpt',

    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-P.arg-basic.dpth-1.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=prover/local_dataset_name=20220916.atmf-P.arg-basic.dpth-1.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/gps=1/lrt=0.0001/mx_epchs=5/mx_inpt_ln=1500/mx_outpt_ln=100/smpl_gl=intermediates/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=4-step=60000.ckpt',
    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-PA.arg-basic.dpth-1.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=prover/local_dataset_name=20220916.atmf-PA.arg-basic.dpth-1.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/gps=1/lrt=0.0001/mx_epchs=5/mx_inpt_ln=1500/mx_outpt_ln=100/smpl_gl=intermediates/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=4-step=60000.ckpt',
    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-PA.arg-compl.dpth-1.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=prover/local_dataset_name=20220916.atmf-PA.arg-compl.dpth-1.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/gps=1/lrt=0.0001/mx_epchs=5/mx_inpt_ln=1500/mx_outpt_ln=100/smpl_gl=intermediates/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=4-step=60000.ckpt',
    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-PA.arg-compl.dpth-3.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=prover/local_dataset_name=20220916.atmf-PA.arg-compl.dpth-3.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/gps=1/lrt=0.0001/mx_epchs=5/mx_inpt_ln=1500/mx_outpt_ln=100/smpl_gl=intermediates/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=4-step=59996.ckpt',
    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-PA.arg-compl.dpth-5.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=prover/local_dataset_name=20220916.atmf-PA.arg-compl.dpth-5.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/gps=1/lrt=0.0001/mx_epchs=5/mx_inpt_ln=1500/mx_outpt_ln=100/smpl_gl=intermediates/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=4-step=59992.ckpt',

    '20220925.pre_train.large_epochs.large_accum__local_dataset_name=local_dataset_name=ruletaker.include_all_answers': './outputs/10.train.py/20220925.pre_train.large_epochs.large_accum/component=prover/local_dataset_name=ruletaker.include_all_answers/base_config_name=ruletaker.include_all_answers.20220922.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/gps=1/lrt=None/mx_btchs=None/mx_epchs=None/mx_inpt_ln=1000/mx_outpt_ln=100/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=19-step=30120.ckpt',
    '20220925.pre_train.large_epochs.large_accum__local_dataset_name=20220916.atmf-PA.arg-compl.dpth-3.UNKNOWN': './outputs/10.train.py/20220925.pre_train.large_epochs.large_accum/component=prover/local_dataset_name=20220916.atmf-PA.arg-compl.dpth-3.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/gps=1/lrt=None/mx_btchs=None/mx_epchs=None/mx_inpt_ln=1000/mx_outpt_ln=100/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=19-step=31260.ckpt',
    '20220925.pre_train.large_epochs.large_accum__local_dataset_name=20220916.atmf-PA.arg-compl.dpth-5.UNKNOWN': './outputs/10.train.py/20220925.pre_train.large_epochs.large_accum/component=prover/local_dataset_name=20220916.atmf-PA.arg-compl.dpth-5.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/gps=1/lrt=None/mx_btchs=None/mx_epchs=None/mx_inpt_ln=1000/mx_outpt_ln=100/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=19-step=31260.ckpt',

    '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-1e-05': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=1e-05/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-05': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=3e-05/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-06': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=3e-06/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000.lrt-1e-05': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=1e-05/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000.lrt-3e-05': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=3e-05/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000.lrt-3e-06': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=3e-06/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000.lrt-1e-05': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=1e-05/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000.lrt-3e-05': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=3e-05/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000.lrt-3e-06': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=3e-06/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-1e-05': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=1e-05/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-05': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=3e-05/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',
    '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-06': './outputs/10.train.py/20221112.various_negatives.epoch-1.lrate-small.sweep/system_name=None/component=prover/local_dataset_name=20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000/local_dataset_1_name=None/EB_task=None/split=None/base_config_name=FLNLcorpus.20220827.base/model_name=google@long-t5-tglobal-base/checkpoint_name=None/dtst_1_nm=None/gps=1/lg_exmpls=True/lrt=3e-06/mx_epchs=1/mx_evl_btchs=1/mx_inpt_ln=2000/mx_outpt_ln=100/mx_trn_btchs=None/n_smlrty_thrshld=False/sd=1/sht=FT/shffl_trn=True/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/wrmp_stps=1000/lightning_logs/version_0/checkpoints/epoch=0-step=1515.ckpt',


}

_VERIFIER_CHECKPOINTS = {
    'ruletaker.NLProofS': './checkpoints/RuleTaker/NLProofS/verifier/epoch=49-step=93000.ckpt',

    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-P.arg-basic.dpth-1.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=verifier/local_dataset_name=20220916.atmf-P.arg-basic.dpth-1.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=roberta-large/checkpoint_name=None/gps=1/lrt=None/mx_epchs=5/mx_inpt_ln=400/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=1-step=19538.ckpt',
    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-PA.arg-basic.dpth-1.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=verifier/local_dataset_name=20220916.atmf-PA.arg-basic.dpth-1.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=roberta-large/checkpoint_name=None/gps=1/lrt=None/mx_epchs=5/mx_inpt_ln=400/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=4-step=68116.ckpt',
    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-PA.arg-compl.dpth-1.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=verifier/local_dataset_name=20220916.atmf-PA.arg-compl.dpth-1.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=roberta-large/checkpoint_name=None/gps=1/lrt=None/mx_epchs=5/mx_inpt_ln=400/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=4-step=67152.ckpt',
    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-PA.arg-compl.dpth-3.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=verifier/local_dataset_name=20220916.atmf-PA.arg-compl.dpth-3.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=roberta-large/checkpoint_name=None/gps=1/lrt=None/mx_epchs=5/mx_inpt_ln=400/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=0-step=45000.ckpt',
    '20220919.UNKNOWN.fix_translation.local_dataset_name=20220916.atmf-PA.arg-compl.dpth-5.UNKNOWN': './outputs/10.train.py/20220919.UNKNOWN.fix_translation/component=verifier/local_dataset_name=20220916.atmf-PA.arg-compl.dpth-5.UNKNOWN/base_config_name=FLNLcorpus.20220827.base/model_name=roberta-large/checkpoint_name=None/gps=1/lrt=None/mx_epchs=5/mx_inpt_ln=400/stnc_indctn_mthd=STANCE_MARKER_IN_PROOF/lightning_logs/version_0/checkpoints/epoch=0-step=45000.ckpt',

}



ICML_2023_NL_TRANSFER_MAJOR_DATASETS = [
    # '20221101__arg-basic__dpth-3__bx-3__dist-var__dist_size-0__reuse-0.0__fixed_transl-True__voc_limit-100__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-3__bx-3__dist-var__dist_size-0__reuse-0.0__fixed_transl-True__voc_limit-100__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-3__bx-3__dist-var__dist_size-0__reuse-0.0__fixed_transl-False__voc_limit-None__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-0__reuse-0.0__fixed_transl-False__voc_limit-None__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-10__reuse-1.0__fixed_transl-False__voc_limit-None__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-10__reuse-1.0__fixed_transl-False__voc_limit-None__dataset_size-300000',

    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-10__reuse-1.0__fixed_transl-False__voc_limit-None__dataset_size-100000.local_dataset_1_name--ruletaker.include_all_answers',
    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-10__reuse-1.0__fixed_transl-False__voc_limit-None__dataset_size-100000.local_dataset_1_name--cc100.20221103.small',

    # '20221107__arg-base__dpth-03__dist-00__transl-nrrw__size-100000',
    # '20221107__arg-cmpl__dpth-03__dist-00__transl-nrrw__size-100000',
    # '20221107__arg-cmpl__dpth-03__dist-00__transl-wide__size-100000',
    # '20221107__arg-cmpl__dpth-03__dist-10__transl-wide__size-100000',
    # '20221107__arg-cmpl__dpth-10__dist-10__transl-wide__size-100000',

    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000',
    # '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000',

    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-1e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-06',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000.lrt-1e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000.lrt-3e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000.lrt-3e-06',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000.lrt-1e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000.lrt-3e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000.lrt-3e-06',
    # '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-1e-05',
    # '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-05',
    # '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-06',

    # '20221117__arg-RT__frml-cmpl__tree-tiny__dist-0__transl_dist--20__transl-wide__size-100000',

    # '20221120.negative_tree__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-100000',

    # '20221123.and__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-10000',

    # '20221124.and__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-10000',

    # '20221125.full__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-10000',

    # '20221126.transl__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-30000',

    # '20221203.first_exp__arg-RT__frml-smpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-RT__frml-cmpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-AA__frml-cmpl__dist-20__transl-nrrw__tree-1__dataset_size-30000',
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000',

    # ---------------------------------- 20221216 additional experiments ------------------------------------
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-FLNL__frml-smpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-5__dataset_size-30000',

    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',
    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-8__dataset_size-30000.G_MP',

    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT',

    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-wide__tree-5__dataset_size-30000.G_MP',
    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.G_MP',

    # ---------------------------------- multitask ------------------------------------
    # XXX: missing!
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.local_dataset_1_name--ruletaker.ours.20221202',
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.local_dataset_1_name--cc100.20221103.small',

    # ---------------------------------- 20221217.back_to_the_past ------------------------------------
    # '20221217.back_to_the_past__arg-FLNL__frml-cmpl__dist-10__transl-wide__tree-10__dataset_size-100000',

    # ---------------------------------- baselines ------------------------------------
    # 'ruletaker.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',
    'ruletaker.D5.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',
    # 'ruletaker.NatLang.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',

    # -- for multitask --
    # 'ruletaker.ours.20221202',
    # 'ruletaker.NL.ours.20221202',

    # 'EB-task1.shuffled',
    # 'EB-task2.shuffled',

    # None,
]


ICML_2023_NL_TRANSFER_MAJOR_DATASETS_LARGE_DEPTH = [
    # '20221101__arg-basic__dpth-3__bx-3__dist-var__dist_size-0__reuse-0.0__fixed_transl-True__voc_limit-100__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-3__bx-3__dist-var__dist_size-0__reuse-0.0__fixed_transl-True__voc_limit-100__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-3__bx-3__dist-var__dist_size-0__reuse-0.0__fixed_transl-False__voc_limit-None__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-0__reuse-0.0__fixed_transl-False__voc_limit-None__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-10__reuse-1.0__fixed_transl-False__voc_limit-None__dataset_size-100000',
    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-10__reuse-1.0__fixed_transl-False__voc_limit-None__dataset_size-300000',

    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-10__reuse-1.0__fixed_transl-False__voc_limit-None__dataset_size-100000.local_dataset_1_name--ruletaker.include_all_answers',
    # '20221101__arg-cmpl__dpth-10__bx-5__dist-var__dist_size-10__reuse-1.0__fixed_transl-False__voc_limit-None__dataset_size-100000.local_dataset_1_name--cc100.20221103.small',

    # '20221107__arg-base__dpth-03__dist-00__transl-nrrw__size-100000',
    # '20221107__arg-cmpl__dpth-03__dist-00__transl-nrrw__size-100000',
    # '20221107__arg-cmpl__dpth-03__dist-00__transl-wide__size-100000',
    # '20221107__arg-cmpl__dpth-03__dist-10__transl-wide__size-100000',
    # '20221107__arg-cmpl__dpth-10__dist-10__transl-wide__size-100000',

    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000',
    # '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000',

    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-1e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-06',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000.lrt-1e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000.lrt-3e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--0__transl-wide__unk-0.65__size-100000.lrt-3e-06',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000.lrt-1e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000.lrt-3e-05',
    # '20221112__arg-cmpl__dpth-10__dist-5__transl_dist--10__transl-wide__unk-0.33__size-100000.lrt-3e-06',
    # '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-1e-05',
    # '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-05',
    # '20221112__arg-cmpl__dpth-3__dist-5__transl_dist--0__transl-wide__unk-0.33__size-100000.lrt-3e-06',

    # '20221117__arg-RT__frml-cmpl__tree-tiny__dist-0__transl_dist--20__transl-wide__size-100000',

    # '20221120.negative_tree__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-100000',

    # '20221123.and__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-10000',

    # '20221124.and__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-10000',

    # '20221125.full__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-10000',

    # '20221126.transl__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-30000',

    # '20221203.first_exp__arg-RT__frml-smpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-RT__frml-cmpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-AA__frml-cmpl__dist-20__transl-nrrw__tree-1__dataset_size-30000',
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-30000',
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000',

    # ---------------------------------- 20221216 additional experiments ------------------------------------
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    # '20221203.first_exp__arg-FLNL__frml-smpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-5__dataset_size-30000',

    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',
    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-8__dataset_size-30000.G_MP',

    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP',
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT',

    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-wide__tree-5__dataset_size-30000.G_MP',
    # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.G_MP',

    # ---------------------------------- multitask ------------------------------------
    # XXX: missing!
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.local_dataset_1_name--ruletaker.ours.20221202',
    # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.local_dataset_1_name--cc100.20221103.small',

    # ---------------------------------- 20221217.back_to_the_past ------------------------------------
    # '20221217.back_to_the_past__arg-FLNL__frml-cmpl__dist-10__transl-wide__tree-10__dataset_size-100000',

    # ---------------------------------- baselines ------------------------------------
    # 'ruletaker.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',
    'ruletaker.D5.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',
    # 'ruletaker.NatLang.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',

    # -- for multitask --
    # 'ruletaker.ours.20221202',
    # 'ruletaker.NL.ours.20221202',

    # 'EB-task1.shuffled',
    # 'EB-task2.shuffled',

    # None,
]


# SHOT_SETTINGS = {
#     'zero-shot': {
#         'limit_train_batches': 1,
#         'shuffle_train': False,
#         'limit_eval_batches': 500,
# 
#         'num_train_epochs': None,
#         'warmup_steps': 0,
# 
#         'max_steps': 1,
# 
#         'num_val_stage_throught_training': None,
#         'val_check_interval_in_batch': None,
#         # 'check_val_every_n_epoch': 1,
#     },
# 
#     'few-shot.batch-3': {
#         'limit_train_batches': 3,
#         'shuffle_train': False,
#         'limit_eval_batches': 500,
# 
#         'num_train_epochs': None,
#         'warmup_steps': 500,
# 
#         'max_steps': 2000,
# 
#         'num_val_stage_throught_training': 5,
#         'val_check_interval_in_batch': None,
#         # 'check_val_every_n_epoch': None,
#     },
# 
#     'few-shot.batch-25': {
#         'limit_train_batches': 25,
#         'shuffle_train': False,
#         'limit_eval_batches': 500,
# 
#         'num_train_epochs': None,
#         'warmup_steps': 500,
# 
#         'max_steps': 2000,
# 
#         'num_val_stage_throught_training': 5,
#         'val_check_interval_in_batch': None,
#         # 'check_val_every_n_epoch': None,
#     },
# 
#     'few-shot.batch-75': {
#         'limit_train_batches': 75,
#         'shuffle_train': False,
#         'limit_eval_batches': 500,
# 
#         'num_train_epochs': None,
#         'warmup_steps': 500,
# 
#         'max_steps': 2000,
# 
#         'num_val_stage_throught_training': 5,
#         'val_check_interval_in_batch': None,
#         # 'check_val_every_n_epoch': None,
#     },
# 
#     'few-shot.batch-250': {
#         'limit_train_batches': 1000,
#         'shuffle_train': False,
#         'limit_eval_batches': 500,
# 
#         'num_train_epochs': None,
# 
#         'warmup_steps': 500,
# 
#         'max_steps': 2000,
# 
#         'num_val_stage_throught_training': 5,
#         'val_check_interval_in_batch': None,
#         # 'check_val_every_n_epoch': None,
#     },
# 
#     'FT': {
#         # -- NLProofS_hypara--
#         # 'max_steps': 12500,
#         # 'max_source_length': 1024,
#         # 'max_target_length': 64,
# 
# 
#         'limit_train_batches': None,
#         'shuffle_train': True,
#         'limit_eval_batches': 500,
# 
#         'num_train_epochs': None,
# 
#         'num_val_stage_throught_training': 10,
#         'val_check_interval_in_batch': None,
# 
#         # ----------------------- pre-trianing --------------------------
# 
#         # -- pre-training: FLNL (arg-RT/arg-AA) / RuleTaker / EB
#         # 'max_steps': 5000,
# 
#         # -- pre-training (RT_large_steps): FLNL (arg-RT/arg-AA) / RuleTaker / EB
#         # 'max_steps': 20000,
# 
#         # -- pre-training: FLNL (arg-FLNL)
#         # 'max_steps': 20000,
# 
#         # -- pre-training: large models
#         # 'max_steps': 10000,
# 
#         # ----------------------- EB fine-tuning --------------------------
#         'max_steps': 10000,
#         # 'max_steps': 5000,
# 
#         'warmup_steps': None,
#         # 'warmup_steps': 1000,
#         # 'warmup_steps': 3000,
#     },
# 
# 
#     # ----------- XXX: this is for debug!!! --------------
#     # 'FT': {
#     #     'limit_train_batches': None,
#     #     'shuffle_train': True,
#     #     'limit_eval_batches': 100,
# 
#     #     'num_train_epochs': None,
#     #     'max_steps': 100,
#     #     'warmup_steps': 10,
# 
#     #     'num_val_stage_throught_training': 2,
#     #     'val_check_interval_in_batch': None,
#     #     # 'check_val_every_n_epoch': None,
#     # },
# 
# }


SHOT_SETTINGS = {
    'FS.shot-0': {
        'max_train_samples': 0,
        'max_eval_samples': 2000,
        'max_predict_samples': 2000,

        'num_train_epochs': None,
        'warmup_steps': 0,

        'max_steps': 1,
        'eval_steps': 1,
    },

    'FS.shot-10': {
        'max_train_samples': 10,
        'max_eval_samples': 2000,
        'max_predict_samples': 2000,

        'num_train_epochs': None,
        'warmup_steps': 500,

        'max_steps': 2000,
        'eval_steps': 100,
    },

    'FS.shot-100': {
        'max_train_samples': 100,
        'max_eval_samples': 2000,
        'max_predict_samples': 2000,

        'num_train_epochs': None,
        'warmup_steps': 500,

        'max_steps': 2000,
        'eval_steps': 100,
    },

    # -- pre-training: FLNL (arg-RT/arg-AA) / RuleTaker / EB
    'FT.step-5000': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 1000,
        'max_predict_samples': 1000,

        'max_steps': 5000,
        'eval_steps': 1000,
        'warmup_steps': 1000,
    },

    'FT.step-8100': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 1000,
        'max_predict_samples': 1000,

        'max_steps': 8100,
        'eval_steps': 4000,
        'warmup_steps': 1000,
    },

    # -- pre-training (RT_large_steps): FLNL (arg-RT/arg-AA) / RuleTaker / EB
    # -- pre-training: FLNL (arg-FLNL)
    'FT.step-20000': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 1000,
        'max_predict_samples': 1000,

        'max_steps': 20000,
        'eval_steps': 5000,
        'warmup_steps': 1000,
    },

    'debug.tiny': {
        'max_train_samples': 10,
        'max_eval_samples': 10,
        'max_predict_samples': 10,

        'num_train_epochs': None,

        'max_steps': 300,
        'eval_steps': 300,
        'warmup_steps': 0,
    },

}


class CheckpointSpec(BaseModel):

    name_or_local_dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    lrate: Optional[float] = None
    # add_final_reference_to_proofs: Optional[bool] = None


def get_checkpoints(spec: CheckpointSpec,
                    check_point_dirs: Optional[List[Union[str, Path]]] = None) -> List[Tuple[str, CheckpointSpec]]:
    if spec.name_or_local_dataset_name.startswith('t5-'):
        return [(spec.name_or_local_dataset_name, spec)]
    else:
        raise NotImplementedError()
    if spec.name_or_local_dataset_name is None:
        return [(None, CheckpointSpec())]

    name_or_local_dataset_name = spec.name_or_local_dataset_name

    if name_or_local_dataset_name in _PROVER_CHECKPOINTS:
        return [_PROVER_CHECKPOINTS[name_or_local_dataset_name], CheckpointSpec(name_or_local_dataset_name=name_or_local_dataset_name)]
    else:
        if check_point_dirs is None:
            raise ValueError()

        checkpoints: List[Tuple[str, CheckpointSpec]] = []
        for check_point_dir in check_point_dirs:
            for checkpoint in Path(check_point_dir).glob('**/*ckpt'):
                lab_setting = json.load(open(checkpoint.parent.parent.parent.parent / 'lab.params.json'))

                def check_spec(name: str) -> bool:
                    lab_value = lab_setting.get(name, '<<default>>')
                    if getattr(spec, name) is not None and lab_value != getattr(spec, name):
                        return False
                    else:
                        return True

                not_met = False
                for name in spec.dict().keys():

                    if name == 'name_or_local_dataset_name':
                        continue

                    if not check_spec(name):
                        not_met = True
                        break

                if not_met:
                    continue

                if name_or_local_dataset_name.find('.local_dataset_1_name--') >= 0:
                    local_dataset_name, local_dataset_1_name = name_or_local_dataset_name.split('.local_dataset_1_name--')
                else:
                    local_dataset_name = name_or_local_dataset_name
                    local_dataset_1_name = None

                found_local_dataset_name = lab_setting['local_dataset_name']
                found_local_dataset_1_name = lab_setting.get('local_dataset_1_name', None)

                if found_local_dataset_name == local_dataset_name and found_local_dataset_1_name == local_dataset_1_name:
                    found_spec = CheckpointSpec(**lab_setting)
                    found_spec.name_or_local_dataset_name = name_or_local_dataset_name
                    checkpoints.append((str(checkpoint), found_spec))

        return checkpoints


def get_logging_step_setting(max_steps: Optional[int] = None,
                             eval_steps: Optional[int] = None) -> Dict[str, Any]:
    setting = {}
    if max_steps is not None:
        setting['max_steps'] = max_steps
        if eval_steps is not None:
            setting['eval_steps'] = eval_steps
        if setting['eval_steps'] > max_steps:
            setting['eval_steps'] = max_steps
    setting['save_steps'] = setting.get('eval_steps', None)
    return setting



def make_val_interval_setting(all_setting: Dict[str, Any], train_file: str) -> Dict[str, Any]:

    if all_setting.get('num_train_epochs', None) not in [None, -1]:
        num_val_stage_throught_training = all_setting['num_val_stage_throught_training']
        return {
            'val_check_interval_in_batch': None,
            # 'check_val_every_n_epoch': max(1, int(all_setting['num_train_epochs'] / num_val_stage_throught_training)),
        }
    else:
        if all_setting.get('max_steps', None) is None\
                or all_setting.get('gradient_accumulation_steps', None) is None:
            raise Exception('Could not specify validation interval')
        else:
            max_steps = all_setting['max_steps']
            num_val_stage_throught_training = all_setting['num_val_stage_throught_training']
            num_batches_per_grad_step = all_setting['gradient_accumulation_steps']

            num_batches_per_val = int(max_steps * num_batches_per_grad_step / num_val_stage_throught_training)

            if all_setting['limit_train_batches'] is not None:
                num_batches_per_epoch = all_setting['limit_train_batches']
            else:
                num_train_examples = len(open(train_file).readlines())
                num_batches_per_epoch = max(1, int(num_train_examples / all_setting['per_device_train_batch_size']))

            if num_batches_per_val >= num_batches_per_epoch:
                grad_steps_per_epoch = max(math.ceil(num_batches_per_epoch / num_batches_per_grad_step), 1)
                train_epochs = max_steps / grad_steps_per_epoch

                val_check_interval_in_batch = None
                check_val_every_n_epoch = max(int(train_epochs / num_val_stage_throught_training), 1)
            else:
                val_check_interval_in_batch = max(int(max_steps * num_batches_per_grad_step / num_val_stage_throught_training), 1)
                check_val_every_n_epoch = None

            return {
                'val_check_interval_in_batch': val_check_interval_in_batch,
                # 'check_val_every_n_epoch': check_val_every_n_epoch,
            }


def make_command(output_dir: Union[str, Path],
                 setting: Dict,
                 run_mode: str,
                 n_gpus: Optional[int] = None) -> str:
    unused_option_names = [
        'base_config_name',
        'checkpoint_name',
        'checkpoint_path',
        'shot',
        'max_proof_steps',
        'local_dataset_name',
        'n_proc_per_node',
    ]

    commands: List[str] = []

    commands.append('source ./set-envs.sh &&')

    if run_mode == 'debug':
        commands.append('python ./run_prover.py')
    elif run_mode == 'profile':
        commands.append('kernprof -lv ./run_prover.py')
    elif run_mode == 'torchrun':
        if n_gpus is None:
            raise ValueError()
        commands.append(f'torchrun --nproc_per_node {n_gpus} ./run_prover.py')
    else:
        ValueError()

    commands.extend([
        f'--output_dir {str(output_dir)}',
        f'--logging_dir {str(Path(output_dir) / "tensorboard_log")}',
    ])

    for name, value in setting.items():
        if name in unused_option_names:
            continue

        if isinstance(value, bool):
            commands.append(maybe_option_flag(f'--{name}', setting.get(name, False)))
        else:
            commands.append(maybe_option_value(f'--{name}', setting.get(name, None)))

    return ' '.join(commands)


def make_output_dir(setting: Dict, top_dir: Union[str, Path]) -> str:
    dataset_setting_names = [key for key in setting if
                             key.startswith('dataset_setting.')]
    return build_dir(
        setting,
        top_dir=str(
            Path(top_dir)
            # / f'sstm_nm={setting.get("system_name", str(None))}'
            / f'dtst_nm={setting.get("local_dataset_name", None)}'
            / f'dtst_1_nm={setting.get("local_dataset_1_name", None)}'
            # / f'excld_unknwn={setting.get("exclude_unknown", None)}'
            # / f'add_fnl_rfrc_t_prfs={setting.get("add_final_reference_to_proofs", None)}'
            # / f'EB_tsk={setting.get("EB_task", None)}'
            # / f'splt={setting.get("split", str(None))}'
            / f'bs_cnfg_nm={setting.get("base_config_name", None)}'
            # / f'mdl_nm={setting["model_name"].replace("/", "@") if setting["model_name"] is not None else "None"}'
            / f'chckpnt_nm={setting.get("checkpoint_name", None)}'
        ),
        short=True,

        dirname_ignore_params=[
            # 'system_name',
            # 'EB_task',

            'local_dataset_name',
            'local_dataset_1_name',
            'dataset_1',
            'train_file_1',

            'exclude_unknown',
            # 'add_final_reference_to_proofs',
            # 'split',

            'prover_ckpt',
            'verifier_ckpt',

            'base_config_name',
            # 'base_config_path',

            'checkpoint_name',
            'checkpoint_path',

            'trainer_ckpt_for_resume_training',

            # 'model_name',

            'train_file',
            'validation_file',
            'test_file',

            'train_file_1',
            'validation_file_1',
            'test_file_1',

            'do_train',
            'do_eval',
            'do_predict',

            'estimated_batches_per_epoch',
            'val_check_interval_in_batch',
            # 'check_val_every_n_epoch',
            'per_device_train_batch_size',
            'gradient_accumulation_steps',
            'per_device_eval_batch_size',
            'tokenizer_padding',
            'limit_eval_batches',
            'num_val_stage_throught_training',

            "argument_configs",
            "translation_configs",

            'max_source_length',
            'max_target_length',
            'unknown_ratio',
            'log_examples',
            'dataloader_num_workers',

            'max_grad_norm',
            'max_eval_samples',
            'max_proof_steps',
            'fp16',
            'generation_max_proof_steps',
            'source_prefix',
            'logging_strategy',

            'overwrite_output_dir',
            'log_generation',
            'path_train_1',
            'path_val_1',
            'path_test_1',
            'evaluation_strategy',
            'save_strategy',
            'logging_strategy',
            'logging_steps',
            'predict_with_generate',
            'checkpoint_lrate',
            'checkpoint_model_name_or_path',
            'checkpoint_model_name',
            'eval_steps',
            'file_type',
            'log_examples',
            'max_predict_samples',
            'max_train_samples',
            'max_eval_samples',
            'remove_unused_columns',
            'save_steps',

        ] + dataset_setting_names,
        save_params=True
    )


def make_system_name(lab_setting: Dict) -> str:
    local_dataset_name = lab_setting['local_dataset_name']
    # local_dataset_1_name = lab_setting.get('local_dataset_1_name', None)
    checkpoint_name = lab_setting['checkpoint_name']

    return f'checkpoint_name--{checkpoint_name}__local_dataset_name--{local_dataset_name}'


def run_by_engine(engine: EngineBase,
                  command: str,
                  output_dir: Union[Path, str],
                  hours=24,
                  dry_run=False):
    output_dir = Path(output_dir)
    log_path = output_dir / 'log.txt'
    if isinstance(engine, SubprocessEngine):
        stdout = None
        stderr = None
        wait_until_finish = True
    else:
        command += f' 1>{str(log_path)} 2>&1'
        stdout = output_dir / 'stdout.txt'
        stderr = output_dir / 'stderr.txt'
        wait_until_finish = False

    engine.run(
        command,
        stdout=stdout,
        stderr=stderr,
        options={'l_opts': [f'h_rt={hours}:00:00']},
        dry_run=dry_run,
        wait_until_finish=wait_until_finish,
    )
