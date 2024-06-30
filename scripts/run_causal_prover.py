#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Dict, List, Any, Union, Tuple, Any
import readline
import warnings
import datetime

import numpy as np
import deepspeed
import datasets
import evaluate
import torch
from torch.utils.data import Dataset
from datasets.download.download_config import DownloadConfig
from datasets import (
    get_dataset_config_names,
    load_dataset,
    concatenate_datasets,
    DatasetDict,
    IterableDataset,
    interleave_datasets,
)
from trl import SFTConfig, SFTTrainer
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_utils import get_last_checkpoint
from transformers.testing_utils import CaptureLogger
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model

from logger_setup import setup as setup_logger
from FLD_prover.data_processors import (
    FLDProcessor,
    RuleTakerProcessor,
    PararulePlusProcessor,
    RobustLRProcessor,
    ProofWriterProcessor,
)
from FLD_prover.trainer import ForceCallMetricsSeq2SeqTrainer, RecAdamTrainer
from FLD_prover.tokenizers import load as load_tokenizer
from FLD_prover.lm_types import LMType
from FLD_prover.collators import RemoveUnusedColumnsCollator, RemoveUnusedColumnsCollatorForCompletionOnlyLM
from FLD_prover.generation import generation_handled
from FLD_prover.interactive import launch
from FLD_prover.mixtral_deepspeed_monkey_patch import replace_mixtral_moe_with_dense_impl
from FLD_task import load_deduction


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    from_scratch: bool = field(
        default=False,
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    lora: Optional[bool] = field(
        default=False,
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    is_sft_dataset: Optional[bool] = field(
        default=False,
    )
    sft_lang: str = field(
        default='eng',
    )
    sft_neftune_noise_alpha: Optional[float] = field(
        default=5,
    )
    dataset_names: Optional[str] = field(
        default=None, metadata={"help": "Dataset names separated by ::"}
    )
    dataset_config_names: Optional[str] = field(
        default=None, metadata={"help": "Dataset config names separated by ::"}
    )
    dataset_take_n_s: Optional[str] = field(
        default=None, metadata={"help": "Dataset take 'n' separated by ::. XXX: Should be 2 x #samples you want, as we additionally filter the dataset, which typically only take aboud a half of that datsaet."}
    )
    dataset_probs: Optional[str] = field(
        default=None, metadata={"help": "Dataset probabilities separated by ::"}
    )
    train_files: Optional[str] = field(default=None, metadata={"help": "Training files separated by ::"})
    validation_files: Optional[str] = field(
        default=None,
        metadata={"help": "Validation files separated by ::"},
    )
    file_types: Optional[str] = field(
        default=None, metadata={"help": "File types separated by ::"}
    )
    do_eval_in_outerloop: bool = field(
        default=False,
    )
    save_model_at_end: bool = field(
        default=False,
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column of text field to be use from dataset"}
    )

    logic_dataset_type: str = 'FLD'
    logic_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    logic_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    logic_dataset_concatenate_all_configs: bool = field(
        default=False,
    )
    logic_dataset_concatenate_all_splits_into_train: bool = field(
        default=False,
    )
    logic_train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    logic_validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    logic_dataset_prob: Optional[float] = field(
        default=1.0,
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    train_random_sampling: bool = field(
        default=False,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    num_train_examples_skip: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Skip the first n training examples."
            )
        },
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    eval_random_sampling: bool = field(
        default=False,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    logic_eval_max_samples: Optional[int] = field(
        default=None,
    )
    logic_eval_random_sampling: bool = field(
        default=False,
    )

    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    preprocess_batch_size: Optional[int] = field(
        default=1000,
        metadata={
            "help": (
                "Batch size for preprocessing."
                "XXX: dataset preprocessing with multiple workers and large batch size may hang without any error message."
                "See: https://discuss.huggingface.co/t/datasets-mapper-hanging-issue/32995"
                "Note that the fix introduced in the link did not work for me"
            )
        }
    )
    preprocess_keep_in_memory: bool = field(
        default=False,
    )
    train_sampling_sequential: bool = field(
        default=True,
    )



    # logic_eval_padding: Optional[str] = field(
    #     default="longest",
    # )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    include_prompt_for_causal_lm_loss: bool = field(
        default=False,
        metadata={},
    )
    instruction: bool = field(
        default=False,
        metadata={},
    )

    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    surface_is_formula: bool = field(
        default=False,
        metadata={},
    )

    proof_intermediate_steps: str = field(
        default='include',
    )

    no_subproof_for_unknown: bool = field(
        default=False,
    )

    generation_top_k: int = field(
        default=None,
    )

    generation_num_beams: int = field(
        default=1,
    )

    generation_num_return_sequences: int = field(
        default=1,
    )

    generation_do_sample: bool = field(
        default=False,
    )

    generation_temperature: float = field(
        default=1.0,
    )

    generation_repetition_penalty: float = field(
        default=None,
    )

    generation_max_length: int = field(
        default=2000,
    )

    generation_max_prompt_length: int = field(
        default=1000,
    )

    generation_max_new_tokens: int = field(
        default=None,
    )

    generation_timeout: int = field(
        # default=60,
        default=None,
    )

    evaluation_timeout: int = field(
        # default=60,
        default=None,
    )

    optimizer: str = field(
        default=None,
    )

    rec_adam_target_task_weight: float = field(
        default=1.0,
    )

    rec_adam_fisher_coef: float = field(
        default=300.0,
    )

    update_parameters: str = field(
        default='all',
    )

    interactive_mode: str = field(
        default=None,
    )

    gradio_port: int = 8010

    log_examples: bool = field(
        default=False,
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")


def take(dataset, max_samples: int, random_sampling: bool):
    if isinstance(dataset, IterableDataset):
        dataset = dataset.take(max_samples)
    else:
        _max_samples = min(len(dataset), max_samples)
        if random_sampling:
            indexes = np.random.choice(len(dataset), _max_samples, replace=False)
        else:
            indexes = np.arange(_max_samples)
        dataset = dataset.select(indexes)
    return dataset


def FLD_map_schema(examples: Dict[str, List[Any]]):
    orig_keys = list(examples.keys())
    batch_size = len(examples[orig_keys[0]])

    examples_list = [
        {key: values[i] for key, values in examples.items()}
        for i in range(batch_size)
    ]
    examples_list = [
        load_deduction(example).dict()
        for example in examples_list
    ]

    keys = list(examples_list[0].keys())
    return {key: [examples_list[i][key] for i in range(batch_size)] for key in keys}


def load_raw_dataset_by_name(data_args,
                             model_args,
                             dataset_name: str,
                             dataset_config_name: str,
                             dataset_take_n: int = None,
                             concatenate_all_configs=False,
                             concatenate_all_splits_into_train=False,
                             slim_pajama_take_ratio='10%'):
    load_dataset_kwargs = {
        # 'on_bad_lines': 'skip',
        # 'error_bad_lines': False,
        'cache_dir': model_args.cache_dir,
        'streaming': data_args.streaming,
        'use_auth_token': True if model_args.use_auth_token else None,
        'num_proc': data_args.preprocessing_num_workers,
    }

    if slim_pajama_take_ratio is not None and dataset_name == 'cerebras/SlimPajama-627B':
        load_dataset_kwargs.update({
            'split': [f'train[:{slim_pajama_take_ratio}]', 'validation', 'test'],
        })
        train_ds, valid_ds, test_ds = load_dataset(
            dataset_name,
            dataset_config_name,
            **load_dataset_kwargs,
        )
        raw_datasets = DatasetDict(train=train_ds, validation=valid_ds, test=test_ds)

    else:

        if concatenate_all_configs:
            configs = get_dataset_config_names(dataset_name)
            # if dataset_name.find('proofwriter') >= 0:
            #     configs = [config for config in configs
            #                if not (config.find('birds-electricity') >= 0 or config.find('NatLang') >= 0)]
            logger.info('We will concatenate all configs of %s: %s', dataset_name, str(configs))

            raw_datasets_list = {}
            for _dataset_config_name in configs:
                _raw_datasets = load_dataset(
                    dataset_name,
                    _dataset_config_name,
                    **load_dataset_kwargs,
                )
                raw_datasets_list[_dataset_config_name] = _raw_datasets

            major_datasets = raw_datasets_list[configs[0]]
            raw_datasets = major_datasets
            split_names = set(major_datasets.keys())
            for split_name in split_names:
                split_datasets = [data[split_name] for config, data in raw_datasets_list.items() if split_name in data]
                raw_datasets[split_name] = concatenate_datasets(split_datasets)

        else:
            raw_datasets = load_dataset(
                dataset_name,
                dataset_config_name,
                **load_dataset_kwargs,
            )

    if concatenate_all_splits_into_train:
        logger.info('We will concatenate all splits of %s into the training set', dataset_name)
        split_names = set(raw_datasets.keys())
        split_datasets = [raw_datasets[split_name] for split_name in split_names]
        raw_datasets['train'] = concatenate_datasets(split_datasets)

    if dataset_take_n is not None:
        for split_name in list(raw_datasets.keys()):
            raw_datasets[split_name] = take(raw_datasets[split_name], dataset_take_n, False)

    if "validation" not in raw_datasets.keys():
        if "dev" in raw_datasets.keys():
            raw_datasets["validation"] = raw_datasets["dev"]
        else:
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                **load_dataset_kwargs,
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                **load_dataset_kwargs,
            )

    return raw_datasets


def load_raw_dataset_by_files(data_args,
                              model_args,
                              train_file: Optional[str],
                              validation_file: Optional[str],
                              file_type: str,
                              keep_linebreaks: bool,
                              streaming: bool,
                              concatenate_all_configs=False,
                              concatenate_all_splits_into_train=False):

    if concatenate_all_configs or concatenate_all_splits_into_train:
        raise NotImplementedError()

    data_files = {}
    dataset_args = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file

    extension = file_type
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = keep_linebreaks

    dataset_args.update({
        'data_files': data_files,
        'streaming': data_args.streaming,
        'use_auth_token': True if model_args.use_auth_token else None,
        'cache_dir': model_args.cache_dir,
        # 'on_bad_lines': 'skip',
        # 'error_bad_lines': False,
    })

    if len(data_files) > 0:
        raw_datasets = load_dataset(
            extension,
            **dataset_args,
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                split=f"train[:{data_args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                split=f"train[{data_args.validation_split_percentage}%:]",
                **dataset_args,
            )
    else:
        raw_datasets = DatasetDict()

    return raw_datasets


def parse_listed_option(option: str) -> Optional[List[str]]:
    return [name or None for name in option.split('::')] if option is not None else []


def load_raw_datasets(data_args, model_args):
    dataset_names = parse_listed_option(data_args.dataset_names)
    dataset_config_names = parse_listed_option(data_args.dataset_config_names)
    dataset_take_n_s = parse_listed_option(data_args.dataset_take_n_s)
    train_files = parse_listed_option(data_args.train_files)
    validation_files = parse_listed_option(data_args.validation_files)
    file_types = parse_listed_option(data_args.file_types)

    raw_datasets_list = []
    if len(dataset_names) > 0:
        for i in range(len(dataset_names)):
            raw_datasets_list.append(
                load_raw_dataset_by_name(
                    data_args,
                    model_args,
                    dataset_names[i],
                    dataset_config_names[i] if dataset_config_names[i] != 'None' else None,
                    dataset_take_n=int(dataset_take_n_s[i]) if dataset_take_n_s[i] != 'None' else None,
                )
            )
    else:
        for i in range(len(train_files)):
            raw_datasets_list.append(
                load_raw_dataset_by_files(
                    data_args,
                    model_args,
                    train_files[i],
                    validation_files[i] if validation_files[i] != 'None' else None,
                    file_types[i] if file_types[i] != 'None' else 'json',
                    data_args.keep_linebreaks,
                    data_args.streaming,
                )
            )
    return raw_datasets_list


def tokenize_datasets(training_args,
                      data_args,
                      raw_datasets_list,
                      tokenizer,
                      block_size):
    if len(raw_datasets_list) == 0:
        tokenized_datasets_list = []
    else:
        tokenized_datasets_list = []

        for raw_datasets in raw_datasets_list:
            if training_args.do_train and raw_datasets["train"].features is not None:
                column_names = list(raw_datasets["train"].features)
            elif (training_args.do_eval or data_args.do_eval_in_outerloop) and raw_datasets["validation"].features is not None:
                column_names = list(raw_datasets["validation"].features)
            else:
                column_names = ['text']
            text_column_name = data_args.text_column_name or ("text" if "text" in column_names else column_names[0])

            # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
            tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

            def tokenize_function(examples):
                with CaptureLogger(tok_logger) as cl:
                    output = tokenizer(examples[text_column_name])
                # clm input could be much much longer than block_size
                if "Token indices sequence length is longer than the" in cl.out:
                    tok_logger.warning(
                        "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                        " before being passed to the model."
                    )
                return output

            desc = "[non-logic dataset] tokenize_function()"
            dataset_map_kwargs = {
                'batched': True,
                'batch_size': data_args.preprocess_batch_size,
            }
            if not data_args.streaming:
                dataset_map_kwargs.update({
                    'num_proc': data_args.preprocessing_num_workers,
                    'load_from_cache_file': None if data_args.streaming else not data_args.overwrite_cache,
                    'keep_in_memory': data_args.preprocess_keep_in_memory,
                    'desc': desc,
                })

            with training_args.main_process_first(desc=desc):
                # avoid long text, which make tokenizer too slow
                raw_datasets = raw_datasets.filter(lambda x: len(x[text_column_name]) < 100_000)
                tokenized_datasets = raw_datasets.map(tokenize_function, remove_columns=column_names, **dataset_map_kwargs)

            def group_texts(examples):
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
                # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
                total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            desc = f"[non-logic dataset] group_texts()"
            with training_args.main_process_first(desc=desc):
                tokenized_datasets = tokenized_datasets.map(group_texts, **dataset_map_kwargs)

            tokenized_datasets_list.append(tokenized_datasets)

    return tokenized_datasets_list


def make_logic_data_processor(data_args, tokenizer, max_length, max_prompt_length):
    preprocessor_args = [
        LMType.CAUSAL,
        tokenizer,
    ]

    preprocessor_kwargs = {
        'prompt_prefix': data_args.source_prefix,
        'surface_is_formula': data_args.surface_is_formula,
        # 'padding': logic_padding,
        'max_length': max_length,
        'max_prompt_length': max_prompt_length,
        'proof_intermediate_steps': data_args.proof_intermediate_steps,
        'proof_sampling': False,
        'sample_negative_proof': False,
        'no_subproof_for_unknown': data_args.no_subproof_for_unknown,
        'include_prompt_for_causal_lm_loss': data_args.include_prompt_for_causal_lm_loss,
        'instruction': data_args.instruction,
        # 'log_examples': data_args.log_examples,
    }

    if data_args.logic_dataset_type == 'FLD':
        processor_cls = FLDProcessor
    elif data_args.logic_dataset_type == 'rule_taker':
        processor_cls = RuleTakerProcessor
    elif data_args.logic_dataset_type == 'proof_writer':
        processor_cls = ProofWriterProcessor
    elif data_args.logic_dataset_type == 'pararule_plus':
        processor_cls = PararulePlusProcessor
    elif data_args.logic_dataset_type == 'robust_lr':
        processor_cls = RobustLRProcessor
    else:
        raise ValueError()

    return processor_cls(*preprocessor_args, **preprocessor_kwargs)


def _maybe_logic_preprocess(data_args,
                            logic_data_processor,
                            examples,
                            mode):
    if data_args.logic_dataset_type == 'FLD':
        logic_key = 'hypothesis'
    elif data_args.logic_dataset_type == 'rule_taker':
        logic_key = 'context'
    elif data_args.logic_dataset_type == 'proof_writer':
        logic_key = 'theory'
    elif data_args.logic_dataset_type == 'pararule_plus':
        logic_key = 'context'
    elif data_args.logic_dataset_type == 'robust_lr':
        logic_key = 'context'
    else:
        raise ValueError()

    if logic_key not in examples:
        return examples

    logic_indexes = [i for i in range(len(examples[logic_key]))
                     if examples[logic_key][i] is not None]
    non_logic_indexes = [i for i in range(len(examples[logic_key]))
                         if i not in logic_indexes]
    num_logic_examples = len(logic_indexes)
    num_non_logic_examples = len(non_logic_indexes)

    logic_examples = {
        key: [values[i] for i in logic_indexes]
        for key, values in examples.items()
    }
    non_logic_examples = {
        key: [values[i] for i in non_logic_indexes]
        for key, values in examples.items()
    }

    if data_args.log_non_logic_examples and len(non_logic_examples) > 0:
        i_example = 0
        logger.info(
            '------------------------------ preprocess_function [non-logic example=%d] ------------------------------', i_example)
        for key, values in non_logic_examples.items():
            if len(values) > 0:
                logger.info('%s: "%s"', key, values[i_example])
        data_args.log_non_logic_examples = False  # only log once, as too much logs break the stream, leading to sigkill

    if mode == "auto_regression":
        logic_preproc_mode = "auto_regression"
        logic_padding = "max_length"   # 'longest' leads to error as the shape of tensors will be diferrent in a batch
        feature_names = ['input_ids', 'attention_mask', 'labels']

    elif mode == "generation":
        logic_preproc_mode = "generation"
        # logic_padding = data_args.logic_eval_padding
        logic_padding = "max_length"  # must be 'max_length', otherwise ends with error saying tensor shape differs in a batch
        feature_names = list(logic_examples.keys())
    else:
        raise ValueError()

    if num_logic_examples > 0:
        logic_processed = logic_data_processor.preprocess(
            logic_examples,
            logic_preproc_mode,
            padding=logic_padding,
        )
        logic_data_processor.log_examples = False  # only log once, as too much logs break the stream, leading to sigkill
    else:
        logic_processed = {}

    if mode in "auto_regression":
        if num_logic_examples > 0 and num_non_logic_examples > 0:
            processed = {
                key: torch.concat((logic_processed[key], torch.tensor(
                    non_logic_examples[key], dtype=logic_processed[key].dtype)))
                for key in feature_names
            }
        elif num_logic_examples > 0:
            processed = {key: vals for key, vals in logic_processed.items() if key in feature_names}
        else:
            processed = {key: vals for key, vals in non_logic_examples.items() if key in feature_names}

        return processed
    elif mode == "generation":
        return logic_processed
    else:
        raise ValueError()


def load_logic_raw_datasets(data_args, model_args):
    if data_args.logic_dataset_name is not None:
        logic_raw_datasets = load_raw_dataset_by_name(
            data_args,
            model_args,
            data_args.logic_dataset_name,
            data_args.logic_dataset_config_name,
            concatenate_all_configs=data_args.logic_dataset_concatenate_all_configs,
            concatenate_all_splits_into_train=data_args.logic_dataset_concatenate_all_splits_into_train,
        )
    else:
        logic_raw_datasets = load_raw_dataset_by_files(
            data_args,
            model_args,
            data_args.logic_train_file,
            data_args.logic_validation_file,
            'json',
            data_args.keep_linebreaks,
            False,
            concatenate_all_configs=data_args.logic_dataset_concatenate_all_configs,
            concatenate_all_splits_into_train=data_args.logic_dataset_concatenate_all_splits_into_train,
        )

    if data_args.logic_dataset_type == 'FLD':
        # load and dump once to normalize the schema from different versions of datasets.
        dataset_map_kwargs = {}
        if not data_args.streaming:
            dataset_map_kwargs.update({
                'num_proc': data_args.preprocessing_num_workers,
                'load_from_cache_file': None if data_args.streaming else not data_args.overwrite_cache,
                'keep_in_memory': data_args.preprocess_keep_in_memory,
                'desc': '[logic dataset] mapping schema',
            })

        logic_raw_datasets = logic_raw_datasets.map(
            FLD_map_schema,
            batched=True,
            batch_size=data_args.preprocess_batch_size,
            **dataset_map_kwargs,
        )

    logic_raw_datasets = logic_raw_datasets.filter(lambda x: x is not None)
    return logic_raw_datasets


def make_interleave_datasets(data_args,
                             datasets: List[Dataset],
                             logic_dataset: Optional[Dataset],
                             dataset_probs: List[float]):
    logger.info('making interleave datasets ...')
    if len(datasets) == 0 and logic_dataset is None:
        raise ValueError()

    if len(datasets) == 0:
        dataset_prob_tot = 0.0
        logic_dataset_prob = 1.0
    elif logic_dataset is None:
        dataset_prob_tot = 1.0
        logic_dataset_prob = 0.0

    else:
        logic_dataset_prob = data_args.logic_dataset_prob
        dataset_prob_tot = 1 - data_args.logic_dataset_prob

    all_datasets = datasets.copy()
    all_probs = [dataset_prob_tot * dataset_probs[i] / sum(dataset_probs) for i in range(len(datasets))]
    if logic_dataset is not None:
        all_datasets.append(logic_dataset)
        all_probs.append(logic_dataset_prob)

    all_datasets = [dataset for dataset, prob in zip(all_datasets, all_probs) if prob > 0.0]
    all_probs = [prob for prob in all_probs if prob > 0.0]

    logger.info('dataset probabilities: %s', all_probs)
    if len(all_datasets) == 1:
        return all_datasets[0]
    else:
        return interleave_datasets(
            all_datasets,
            probabilities=all_probs,
            seed=0,
            # stopping_strategy="all_exhausted",
            stopping_strategy="first_exhausted",  # "all_exhausted" will yield dataset that does not respect probs
        )


def make_generation_settings(data_args, tokenizer, model, config):
    generation_config = GenerationConfig.from_model_config(config)
    generation_config.max_time = data_args.generation_timeout
    generation_handle_args = [
        LMType.CAUSAL,
        tokenizer,
        model,
    ]

    if hasattr(model.config, "max_position_embeddings"):
        max_position_embeddings = model.config.max_position_embeddings
        max_length = min(data_args.generation_max_length + 1, max_position_embeddings)
    else:
        max_length = data_args.generation_max_length

    generation_handled_kwargs = {
        'eos_token_id': tokenizer.eos_token_id,
        # 'top_k': data_args.generation_top_k,
        'num_beams': data_args.generation_num_beams,
        'num_return_sequences': data_args.generation_num_return_sequences,
        'do_sample': data_args.generation_do_sample,
        'temperature': data_args.generation_temperature,
        'repetition_penalty': data_args.generation_repetition_penalty,
        'max_length': max_length,
        'max_new_tokens': data_args.generation_max_new_tokens,
    }
    # set top k if not None
    if data_args.generation_top_k is not None:
        generation_handled_kwargs['top_k'] = data_args.generation_top_k
    return generation_config, generation_handle_args, generation_handled_kwargs


def setup_seq2seq_trainer_class(klass,
                                data_args,
                                generation_handle_args,
                                generation_handled_kwargs):
    klass.evaluate = generation_handled(
        klass.evaluate,
        *generation_handle_args,
        timeout_from_call=data_args.evaluation_timeout,
        timeout_msg_title='generation aborted because "evaluate()" took too long',
        **generation_handled_kwargs,
    )
    klass.predict = generation_handled(
        klass.predict,
        *generation_handle_args,
        timeout_from_call=data_args.evaluation_timeout,
        timeout_msg_title='generation aborted because "evaluate()" took too long',
        **generation_handled_kwargs,
    )


def main():
    logging.getLogger().handlers.clear()
    setup_logger(do_stderr=True, level=logging.INFO, clear_other_handlers=True)
    logging.getLogger('absl').setLevel(logging.WARNING)
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    warnings.filterwarnings("ignore", message="is incompatible with gradient checkpointing. Setting")
    replace_mixtral_moe_with_dense_impl()

    # Is this OK? without this magic code, the preprocessing of logic dataset with multiprocess will hang up,
    # possibly because of the torch.where operation used in the processing.
    # https://github.com/pytorch/pytorch/issues/82843#issuecomment-1215281193
    torch.set_num_threads(1)

    # MUST be placed at top (here) !!
    if any(arg.find('--deepspeed') >= 0 for arg in sys.argv):
        # https://github.com/huggingface/accelerate/issues/223
        timeout = datetime.timedelta(seconds=3600 * 10)  # large for preprocessing on large dataset
        deepspeed_port = os.environ.get('RUN_CAUSAL_PROVER_DEEPSPEED_PORT', None)
        logger.info('initialize deepspeed with timeout=%s, port=%s', timeout, deepspeed_port)
        deepspeed.init_distributed(
            timeout=timeout,
            distributed_port=deepspeed_port,
        )

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):  # json file specifiying the arguments
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.dataloader_num_workers > 1:
        raise ValueError('dataloader_num_workers > 0 leads to sigkill during evaluation (generation of proofs) (but I don\'t know why)')

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    if training_args.remove_unused_columns:
        raise ValueError(
            'remove_unused_columns=True is not allowed because we transform dataset instances on-the-fly for augmentation.')

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logging.getLogger().handlers.clear()
    setup_logger(do_stderr=True, level=logging.INFO, clear_other_handlers=True)


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}" +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
        "use_cache": False if training_args.gradient_checkpointing else True,
    }
    config_name = model_args.config_name or model_args.model_name_or_path
    if config_name:
        config = AutoConfig.from_pretrained(config_name, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer = load_tokenizer(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_auth_token=model_args.use_auth_token,
        use_fast_tokenizer=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    if model_args.from_scratch:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            trust_remote_code=True,
        )

    update_parameter_names = []
    if data_args.update_parameters == 'all':
        update_parameter_names = [name for name, params in model.named_parameters()]
    else:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        if data_args.update_parameters == 'attention':
            # model.norm.weiht is somwhow needed, otherwise exception
            update_parameter_names = [name for name, params in model.named_parameters()
                                      if 'attn' in name or 'model.norm.weight' in name]  
        elif data_args.update_parameters == 'mlp':
            update_parameter_names = [name for name, params in model.named_parameters()
                                      if 'mlp' in name or 'model.norm.weight' in name]
        else:
            raise ValueError(data_args.update_parameters)
    freeze_parameter_names = [name for name, params in model.named_parameters()
                              if name not in update_parameter_names]

    logger.info('-- [update_parameters="%s"] will update the following parameters --', data_args.update_parameters)
    for name, param in model.named_parameters():
        if name in update_parameter_names:
            logger.info(name)
            param.requires_grad = True

    logger.info('-- [update_parameters="%s"] will freeze the following parameters --', data_args.update_parameters)
    for name, param in model.named_parameters():
        if name in freeze_parameter_names:
            logger.info(name)
            param.requires_grad = False

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    if model_args.lora:
        # taken from [Quicktour](https://huggingface.co/docs/peft/quicktour)
        peft_config = LoraConfig(task_type=PeftTaskType.CAUSAL_LM,
                                 inference_mode=False,
                                 r=8,
                                 lora_alpha=32,
                                 lora_dropout=0.1)

        # [RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn](https://github.com/huggingface/peft/issues/137)
        model.enable_input_require_grads()

        try:
            model = get_peft_model(model, peft_config)
        except ValueError as e:
            if str(e).find('Please specify `target_modules` in `peft_config`') >= 0:
                peft_config.target_modules = ['query_key_value']
                model = get_peft_model(model, peft_config)
            else:
                raise
        logger.info('train LoRA model with the following parameters:')
        model.print_trainable_parameters()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        logger.info("block_size is set as %d, which is the model's max length")
    else:
        block_size = data_args.block_size
        if data_args.block_size > tokenizer.model_max_length:
            msg = (
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
            raise ValueError(msg)

    if data_args.train_sampling_sequential:
        # [Slower training time per batch for increasing dataset size ](https://github.com/huggingface/transformers/issues/8818#issuecomment-1474785374)
        import transformers.trainer as trainer
        from transformers.trainer import SequentialSampler
        def sampler_monkey_patch(dataset):
            return SequentialSampler(dataset)
        trainer.RandomSampler = sampler_monkey_patch

    raw_datasets_list = load_raw_datasets(data_args, model_args)
    if data_args.is_sft_dataset:
        # SFTTrainer will do the tokenization
        tokenized_datasets_list = raw_datasets_list
    else:
        tokenized_datasets_list = tokenize_datasets(training_args,
                                                    data_args,
                                                    raw_datasets_list,
                                                    tokenizer,
                                                    block_size)

    logic_dataset_processor = make_logic_data_processor(data_args, tokenizer, block_size, block_size)
    data_args.log_non_logic_examples = True

    logic_raw_datasets = load_logic_raw_datasets(data_args, model_args)

    desc = "[logic dataset] _maybe_logic_preprocess()"
    maybe_logic_preprocess_map_kwargs = {
        'batched': True,
        'batch_size': data_args.preprocess_batch_size,
    }
    if not data_args.streaming:
        maybe_logic_preprocess_map_kwargs.update({
            'num_proc': data_args.preprocessing_num_workers,
            'load_from_cache_file': None if data_args.streaming else not data_args.overwrite_cache,
            'keep_in_memory': data_args.preprocess_keep_in_memory,
            'desc': desc,
        })


    logic_processed_dataset = logic_raw_datasets
    for split, dataset in list(logic_raw_datasets.items()):
        _desc = desc + f' on {split} split'
        data_args.log_non_logic_examples = data_args.log_examples
        logic_dataset_processor.log_examples = data_args.log_examples
        with training_args.main_process_first(desc=_desc):
            logic_processed_dataset[split] = dataset.map(
                lambda examples: _maybe_logic_preprocess(data_args, logic_dataset_processor, examples, 'auto_regression'),
                **maybe_logic_preprocess_map_kwargs,

            )
    else:
        logic_processed_dataset = logic_raw_datasets

    dataset_probs = [float(opt) for opt in parse_listed_option(data_args.dataset_probs)]

    if training_args.do_train:
        train_dataset = make_interleave_datasets(
            data_args,
            [tokenized_datasets["train"] for tokenized_datasets in tokenized_datasets_list],
            logic_processed_dataset.get("train", None),
            dataset_probs,
        )

        if data_args.num_train_examples_skip > 0:
            logger.info('skip %d examples from the training dataset', data_args.num_train_examples_skip)
            train_dataset = train_dataset.skip(data_args.num_train_examples_skip)
        if data_args.max_train_samples is not None:
            train_dataset = take(train_dataset,
                                 data_args.max_train_samples,
                                 data_args.train_random_sampling)
    else:
        train_dataset = None

    if training_args.do_eval or data_args.do_eval_in_outerloop:
        eval_dataset = make_interleave_datasets(
            data_args,
            [tokenized_datasets["validation"] for tokenized_datasets in tokenized_datasets_list],
            logic_processed_dataset.get("validation", None),
            dataset_probs,
        )
        eval_dataset = take(eval_dataset,
                            data_args.max_eval_samples,
                            data_args.eval_random_sampling)

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    else:
        eval_dataset = None

    # Wr do the FLD preprocessing here after making interleaved datasets,
    # as the current implementation of interleave_datasets() ignores the processing specified on each dataset.

    desc = "[logic + non-logic interleaved dataset] _maybe_logic_preprocess()"

    generation_config, generation_handle_args, generation_handled_kwargs = make_generation_settings(
        data_args, tokenizer, model, config
    )
    training_args.generation_config = generation_config
    training_args.predict_with_generate = True

    logic_eval_raw_datasets = load_logic_raw_datasets(data_args, model_args)
    logic_eval_dataset_processor = make_logic_data_processor(data_args,
                                                             tokenizer,
                                                             data_args.generation_max_length,
                                                             data_args.generation_max_prompt_length)
    if "validation" in logic_eval_raw_datasets:
        logic_eval_dataset = logic_eval_raw_datasets["validation"]

        if data_args.logic_eval_max_samples is not None:
            logic_eval_dataset = take(logic_eval_dataset,
                                      data_args.logic_eval_max_samples,
                                      data_args.logic_eval_random_sampling)

        data_args.log_non_logic_examples = data_args.log_examples
        logic_eval_dataset_processor.log_examples = data_args.log_examples

        generation_handled_map = generation_handled(
            logic_eval_dataset.map,
            *generation_handle_args,
            **generation_handled_kwargs,
            is_generate_func=False,
        )

        _desc = desc + ' on logic_eval split'
        with training_args.main_process_first(desc=_desc):
            logic_eval_dataset = generation_handled_map(
                lambda examples: _maybe_logic_preprocess(data_args,
                                                         logic_eval_dataset_processor,
                                                         examples,
                                                         'generation'),
                **maybe_logic_preprocess_map_kwargs,
            )
    else:
        logic_eval_dataset = None

    logic_eval_dataset_processor.eval_dataset = logic_eval_dataset
    logic_compute_metrics = logic_eval_dataset_processor.compute_metrics

    setup_seq2seq_trainer_class(ForceCallMetricsSeq2SeqTrainer,
                                data_args,
                                generation_handle_args,
                                generation_handled_kwargs)

    def _build_logic_seq2seq_trainer(other_trainer: Optional[Trainer] = None,
                                     do_compute_metrics=True):
        return ForceCallMetricsSeq2SeqTrainer(
            model,
            other=other_trainer,
            args=training_args,
            data_collator=RemoveUnusedColumnsCollator(return_tensors='pt'),
            train_dataset=None,
            eval_dataset=logic_eval_dataset,
            tokenizer=tokenizer,
            compute_metrics = logic_compute_metrics if do_compute_metrics else None,
        )

    class LogicEvaluationCallback(TrainerCallback):

        def __init__(self, other_trainer: Trainer):
            self._other_trainer = other_trainer
            self._logic_seq2seq_trainer = _build_logic_seq2seq_trainer(other_trainer=self._other_trainer)

        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            self._logic_seq2seq_trainer.state = state
            self._logic_seq2seq_trainer.evaluate(
                metric_key_prefix="logic_eval"
            )

    if data_args.optimizer is None:

        if data_args.is_sft_dataset:
            if len(dataset_probs) != 1:
                raise NotImplementedError()
            if dataset_probs[0] < 1.0:
                raise NotImplementedError('For sft, we currently only support non-logic dataset only training.')

            if data_args.sft_lang == 'eng':
                intro = 'Please answer the question based on the given context.'
                context_template = '### context'
                instruction_template = '### question'
                response_template = '### answer'

            elif data_args.sft_lang == 'jpn':
                intro = '文脈に基づいて、質問に答えてください｡'
                context_template = '### 文脈'
                instruction_template = '### 質問'
                response_template = '### 回答'

            else:
                raise ValueError(data_args.sft_lang)

            def formatting_prompts_func(examples):
                output_texts = []

                def guess_field(candidate_fields: List[str], not_found='raise') -> str:
                    field = None
                    for candidate in candidate_fields:
                        if candidate in examples:
                            field = candidate
                            break
                    if field is None:
                        msg = f'candidate fields {str(candidate_fields)} not found in the examples'
                        if not_found == 'raise':
                            raise ValueError(msg)
                        elif not_found == 'warning':
                            logger.warning(msg)
                        else:
                            raise ValueError()
                    return field

                instruction_field = guess_field(['instruction', 'question'])
                context_field = guess_field(['context'], not_found='warning')
                response_field = guess_field(['response', 'answer'])

                for i in range(len(examples[instruction_field])):
                    instruction = examples[instruction_field][i]
                    context = examples[context_field][i] if context_field is not None else None
                    response = examples[response_field][i]
                    if context is not None:
                        text = '\n'.join([intro, instruction_template, instruction, context_template, context, response_template, response]) + '</s>'
                    else:
                        text = '\n'.join([intro, instruction_template, instruction, response_template, response]) + '</s>'
                    output_texts.append(text)
                return output_texts

            trainer_cls = SFTTrainer
            # sft_config = SFTConfig(
            #     output_dir=training_args.output_dir,
            #     packing=False,
            #     max_seq_length=block_size,
            #     dataset_num_proc=data_args.preprocessing_num_workers,
            #     dataset_batch_size=data_args.preprocess_batch_size,
            #     neftune_noise_alpha: Optional[float] = None
            #     model_init_kwargs: Optional[Dict] = None
            #     dataset_kwargs: Optional[Dict] = None
            #     eval_packing: Optional[bool] = None
            #     num_of_sequences: Optional[int] = 1024
            #     chars_per_token: Optional[float] = 3.6
            # )
            trainer_kwargs = {
                'formatting_func': formatting_prompts_func,
                'max_seq_length': block_size,
                'neftune_noise_alpha': data_args.sft_neftune_noise_alpha,
            }
            collator = RemoveUnusedColumnsCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, return_tensors='pt')

        else:
            trainer_cls = Trainer
            trainer_kwargs = {}
            collator = RemoveUnusedColumnsCollator(return_tensors='pt')

    elif data_args.optimizer == 'rec_adam':
        if data_args.is_sft_dataset:
            raise NotImplementedError()
        else:
            trainer_cls = RecAdamTrainer
            trainer_kwargs = {
                'rec_adam_target_task_weight': data_args.rec_adam_target_task_weight,
                'rec_adam_fisher_coef': data_args.rec_adam_fisher_coef,
            }
            collator = RemoveUnusedColumnsCollator(return_tensors='pt')

    else:
        raise ValueError(f'Unknown optimizer: {model_args.optimizer}')

    # Initialize our Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics = compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available() else None,
        **trainer_kwargs,
    )
    callbacks = trainer.callback_handler.callbacks + [LogicEvaluationCallback(trainer)]
    trainer.callback_handler = CallbackHandler(
        callbacks, trainer.model, trainer.tokenizer, trainer.optimizer, trainer.lr_scheduler
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if data_args.save_model_at_end:
            logger.info("*** Save Model ***")
            trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if data_args.do_eval_in_outerloop:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if data_args.interactive_mode is not None:
        if data_args.logic_dataset_type != 'FLD':
            raise ValueError(f'interactive_mode is not supported for {data_args.logic_dataset_type}')
        data_args.log_non_logic_examples = data_args.log_examples
        logic_eval_dataset_processor.log_examples = data_args.log_examples
        launch(
            _build_logic_seq2seq_trainer(other_trainer=trainer, do_compute_metrics=False),
            tokenizer,
            lambda examples: _maybe_logic_preprocess(data_args, logic_eval_dataset_processor, examples, 'generation'),
            data_args.interactive_mode,
            gradio_port=data_args.gradio_port,
        )


if __name__ == "__main__":
    main()
