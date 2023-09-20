#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import line_profiling
import deepspeed
import time
from FLD_prover import (
    StepWiseGenerationTrainer,
    preprocess_examples_train,
    preprocess_examples_eval,
)
from FLD_prover.utils import tokenize_with_log
from FLD_task.proof import get_stance_markers
from FLD_task import (
    load_deduction,
    serialize,
    build_metrics,
    prettify_context_text,
    prettify_proof_text,
    log_example,
)
from logger_setup import setup as setup_logger
from torch.distributed.elastic.multiprocessing.errors import record
from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model
import huggingface_hub
from transformers.utils.versions import require_version
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.generation.configuration_utils import GenerationConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    default_data_collator,
    is_torch_tpu_available,
)
from transformers.generation.stopping_criteria import StoppingCriteria
import transformers
import gradio as gr
from filelock import FileLock
from datasets import Features, Value
from datasets import load_dataset
import numpy as np
import nltk  # Here to have a nice missing dependency error message early on
import evaluate
import datasets
import torch
from enum import Enum
import readline
from collections import defaultdict
from typing import Optional, Dict, List, Any, Union, Tuple, Any
from dataclasses import dataclass, field
import sys
import logging
import os
import re
import tempfile
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.29.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    lora: Optional[bool] = field(
        default=False,
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    do_eval_in_outerloop: bool = field(
        default=False,
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    dataset_push_to_hub_repo_name: str = field(
        default=None,
    )
    file_type: Optional[str] = field(
        default=None, metadata={"help": "The input file type such as 'json' or 'csv'"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    # pad_to_max_length: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "Whether to pad all samples to model maximum sentence length. "
    #             "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
    #             "efficient on GPU but very bad for TPU."
    #         )
    #     },
    # )
    tokenizer_padding: Optional[str] = field(
        default=False,
    )
    no_prompt_mask: Optional[bool] = field(
        default=False,
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
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    ignore_prompt_for_causal_lm_loss: bool = field(
        default=True,
        metadata={},
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    proof_sampling: str = field(
        default="stepwise",
        metadata={"help": "[stepwise|all_at_once]"},
    )

    sample_negative_proof: bool = field(
        default=False,
    )

    no_subproof_for_unknown: bool = field(
        default=False,
    )

    generation_max_proof_steps: int = field(
        default=20,
    )

    generation_top_k: int = field(
        default=None,
    )

    generation_num_return_sequences: int = field(
        default=1,
    )

    generation_timeout: int = field(
        default=60,
    )

    interactive_mode: str = field(
        default=None,
    )

    gradio_port: int = 8010

    log_examples: bool = field(
        default=False,
    )

    log_generation: bool = field(
        default=False,
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
            and self.interactive_mode is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, test file, or self.interactive_mode.")
        else:
            # if self.train_file is not None:
            #     extension = self.train_file.split(".")[-1]
            #     assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            # if self.validation_file is not None:
            #     extension = self.validation_file.split(".")[-1]
            #     assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            # if self.test_file is not None:
            #     extension = self.test_file.split(".")[-1]
            #     assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
            pass
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


class LMType(Enum):
    SEQ_2_SEQ = 'seq2seq'
    CAUSAL = 'causal'


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}


class RemoveUnusedColumnsCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        for feature in features:
            if "depth" in feature:
                feature.pop("depth", None)
        return super().__call__(features, return_tensors=return_tensors)


class RemoveUnusedColumnsCollator:

    def __init__(self, return_tensors: Optional[str] = None):
        if return_tensors is None:
            raise ValueError()
        self.return_tensors = return_tensors

    def __call__(self, features, return_tensors=None):
        for feature in features:
            if "depth" in feature:
                feature.pop("depth", None)
        return default_data_collator(features,
                                     return_tensors=return_tensors or self.return_tensors)


class MaxTimeCriteriaWithWarning(StoppingCriteria):
    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        elapsed = time.time() - self.initial_timestamp
        do_timeout = elapsed > self.max_time
        if do_timeout:
            logger.warning('generation timeout with %d sec', self.max_time)
            return True
        else:
            return False

@record
def main():
    # TODO we placed here to avoid error but want to place later 
    logging.getLogger().handlers.clear()  # remove handler automatically added
    setup_logger(do_stderr=True, level=logging.INFO)
    logging.getLogger('absl').setLevel(logging.WARNING)
    # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    if any(arg == '--lm_type' for arg in sys.argv):
        arg_idx = sys.argv.index('--lm_type')
        lm_type = LMType(sys.argv[arg_idx + 1])
        # XXX: too hacky
        sys.argv = sys.argv[:arg_idx] + sys.argv[arg_idx + 2:]
    else:
        lm_type = 'seq2seq'

    # must be placed at top, so we extract string from sys.argv directry
    if any(arg.find('deepspeed') >= 0 for arg in sys.argv):
        deepspeed.init_distributed()

    # Seq2SeqTrainingArguments is a sub-class of TrainingArguments, therefore simply loading as Seq2SeqTrainingArguments is enough
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if lm_type == LMType.CAUSAL and data_args.proof_sampling == 'stepwise':
        raise ValueError('causal does not support stepwise prover because it is actually equivalent to all_at_once prover.')

    if training_args.dataloader_num_workers > 0:
        raise Exception(f'dataloader_num_workers({training_args.dataloader_num_workers}) > 0 is extemely slow in our program. We strongly recommend to ppecify it as 0.')

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    if training_args.fp16 and model_args.model_name_or_path.startswith('t5'):
        raise ValueError('Do not use fp16 with for T5 models. Loss will explode or go to nan.')
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

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

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        extension = data_args.file_type
        if extension is None or extension not in ['json', 'csv']:
            raise ValueError()

        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            # extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            # extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            # extension = data_args.test_file.split(".")[-1]

        if len(data_files) > 0:
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            raw_datasets = {}

    if data_args.dataset_push_to_hub_repo_name is not None:
        raw_datasets.push_to_hub(data_args.dataset_push_to_hub_repo_name)
        return

    # load and dump once to normalize the schema from different versions of datasets.
    raw_datasets = raw_datasets.map(
        lambda example: load_deduction(example).dict(),
        batched=False,
        load_from_cache_file=False,  # to always reflect the modification of the preprocessing
    )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_name = model_args.config_name or model_args.model_name_or_path
    config = AutoConfig.from_pretrained(
        config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )

    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    if tokenizer_name.startswith('stabilityai'):
        tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1",
                                                   additional_special_tokens=['▁▁'],
                                                   use_auth_token=True if model_args.use_auth_token else None) 
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if lm_type == LMType.SEQ_2_SEQ:
        auto_model_class = AutoModelForSeq2SeqLM
    elif lm_type == LMType.CAUSAL:
        auto_model_class = AutoModelForCausalLM
    else:
        raise NotImplementedError()
    model = auto_model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
        # torch_dtype=torch.float16 if training_args.fp16 else None,
    )
    # torch._C._dynamo.disable
    # model = torch.compile(model)

    if model_args.lora:
        # taken from [Quicktour](https://huggingface.co/docs/peft/quicktour)
        if lm_type == LMType.SEQ_2_SEQ:
            task_type = PeftTaskType.SEQ_2_SEQ_LM
        elif lm_type == LMType.CAUSAL:
            task_type = PeftTaskType.CAUSAL_LM
        else:
            raise NotImplementedError()
        peft_config = LoraConfig(task_type=task_type,
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


    # Override the decoding parameters of Seq2SeqTrainer
    # + 1 to be compatible with beam search
    training_args.generation_max_length = (
        training_args.generation_max_length + 1
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length + 1
    )

    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if lm_type == LMType.SEQ_2_SEQ and model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    context_column = 'context'
    next_step_column = 'next_step'
    gold_proof_column = 'gold_proof'
    ignore_index = -100

    # Temporarily set max_target_length for training.
    # max_target_length = data_args.max_target_length
    # padding = "max_length" if data_args.pad_to_max_length else False
    padding = data_args.tokenizer_padding or False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def extract_serials(examples: Dict[str, List[Any]]) -> Tuple[List[str], List[str], List[str]]:
        if data_args.log_examples:
            logger.info('')
            logger.info('============================= extract_inputs_targets() =============================')

        inputs: List[str] = []
        targets: List[str] = []
        gold_proofs: List[str] = []
        for i_example in range(len(examples[context_column])):
            context = examples[context_column][i_example]
            next_step = examples[next_step_column][i_example]
            gold_proof = examples[gold_proof_column][i_example]

            inputs.append(context)
            targets.append(next_step)
            gold_proofs.append(gold_proof)
            if data_args.log_examples:
                logger.info('context    [%d] : "%s"', i_example, context)
                logger.info('next_step [%d] : "%s"', i_example, next_step)
                logger.info('gold_proof [%d] : "%s"', i_example, gold_proof)

        inputs = [prefix + inp for inp in inputs]

        return inputs, targets, gold_proofs

    def prepare_tokenized_inputs(inputs: List[str],
                                 max_length: int,
                                 **kwargs) -> Dict[str, torch.Tensor]:
        _padding = kwargs.pop('padding', padding)
        tokenized = tokenize_with_log(tokenizer,
                                      text=inputs,
                                      max_length=max_length,
                                      padding=_padding,
                                      truncation=True,
                                      return_tensors='pt',
                                      add_special_tokens=False,
                                      **kwargs)
        return tokenized

    def prepare_tokenized_targets(targets: List[str],
                                  max_length: int,
                                  **kwargs) -> Dict[str, torch.Tensor]:
        tokenized = tokenize_with_log(tokenizer,
                                      text_target=targets,
                                      max_length=max_length,
                                      padding=padding,
                                      truncation=True,
                                      return_tensors='pt',
                                      add_special_tokens=False,
                                      **kwargs)
        return tokenized

    def mask_labels_by_ignore_index(labels, mask_lengths: Optional[List[int]] = None):
        """
        [OpenCALM-7BをLoRAでinstruction tuningするための実装解説](https://qiita.com/m__k/items/173ade78990b7d6a4be4)
        """

        if mask_lengths is not None:
            if data_args.ignore_prompt_for_causal_lm_loss:
                for i_label, mask_length in enumerate(mask_lengths):
                    non_pad_first_token = 0  # for the case of "left" padding
                    for i_token, token_id in enumerate(labels[i_label].numpy().tolist()):
                        if token_id != tokenizer.pad_token_id:
                            non_pad_first_token = i_token
                            break
                    labels[i_label][non_pad_first_token : non_pad_first_token + mask_length] = ignore_index

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels = torch.where(labels != tokenizer.pad_token_id, labels, ignore_index)

        return labels

    def unmask_by_pad_token(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if not isinstance(tensor, (np.ndarray, torch.Tensor)):
            raise ValueError()
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        return np.where(tensor != ignore_index, tensor, tokenizer.pad_token_id)

    whole_proof_max_length = (
        data_args.max_target_length * 200 if data_args.proof_sampling == 'stepwise' and lm_type == LMType.SEQ_2_SEQ
        else data_args.max_target_length
    )
    proof_col = 'gold_proofs'

    @profile
    def preprocess_function(examples: Dict[str, List[Any]], split: str) -> Dict[str, List[Any]]:
        batch_size = len(list(examples.values())[0])
        unbatched_examples = [{key: examples[key][i] for key in examples.keys()}
                              for i in range(batch_size)]

        prompts_w_partial_proof: List[str] = []
        proof_steps: List[str] = []
        gold_proofs: List[str] = []
        for i_example, example in enumerate(unbatched_examples):
            deduction = load_deduction(example)
            serial = serialize(
                deduction,
                stepwise = (data_args.proof_sampling == 'stepwise'),
                sample_negative_proof = data_args.sample_negative_proof if split == 'train' else False,
                include_max_subproof_for_unknown = not data_args.no_subproof_for_unknown,
            )

            prompt_with_partial_proof = prefix + serial.prompt + (serial.partial_proof or '')

            gold_proof = serial.proofs[0]
            # check whther the tokenizer can recognize stance markers
            gold_proof_dec = tokenizer.decode(prepare_tokenized_targets([gold_proof],
                                                                        whole_proof_max_length)["input_ids"][0])
            if len(get_stance_markers(gold_proof_dec)) == 0:
                logger.warning(
                    '\n'.join([
                        'The tokenizer could not recognized the stance markers.',
                        f'The original proof: "{gold_proof}"',
                        f'The tokenized proof: "{gold_proof_dec}"',
                    ])
                )

            prompts_w_partial_proof.append(prompt_with_partial_proof)
            proof_steps.append(serial.next_proof_step)
            gold_proofs.append(gold_proof)


            if data_args.log_examples:
                logger.info('------------------------------ preprocess_function [example=%d] ------------------------------', i_example)
                logger.info('prompt             : "%s"', prompt_with_partial_proof)
                logger.info('next proof step    : "%s"', serial.next_proof_step)
                logger.info('gold proof         : "%s"', gold_proof)

        _proof_steps_w_eos = [step + f' {tokenizer.eos_token}' for step in proof_steps]

        # without this additional token, we can not accurately calculate the prompt length
        causal_lm_sep_token = '::'
        forward_inputs: Dict[str, Any] = {}
        if split == 'train':
            if any(_targets is None for _targets in proof_steps):
                raise ValueError()

            if lm_type == LMType.SEQ_2_SEQ:
                forward_inputs.update(prepare_tokenized_inputs(prompts_w_partial_proof, data_args.max_source_length))
                forward_inputs["labels"] = prepare_tokenized_targets(_proof_steps_w_eos,
                                                                     data_args.max_target_length)["input_ids"]
                forward_inputs["labels"] = mask_labels_by_ignore_index(forward_inputs["labels"])

            elif lm_type == LMType.CAUSAL:
                # just for getting length
                _prompts = [prompt + causal_lm_sep_token for prompt in prompts_w_partial_proof]

                if data_args.no_prompt_mask:
                    prompt_lengths = None
                else:
                    prompt_ids = [
                        prepare_tokenized_inputs(
                            prompt,
                            data_args.max_source_length,
                            padding='longest',
                            return_length=True,
                            # add_special_tokens=False,
                        )
                        for prompt in _prompts
                    ]
                    prompt_lengths = [_promt_ids['length'][0] for _promt_ids in prompt_ids]

                inputs_with_targets = [f'{prompt}{proof_step}'
                                       for prompt, proof_step in zip(_prompts, _proof_steps_w_eos)]
                forward_inputs.update(prepare_tokenized_inputs(inputs_with_targets, data_args.max_source_length))
                forward_inputs["labels"] = forward_inputs['input_ids'].detach().clone()

                forward_inputs["labels"] = mask_labels_by_ignore_index(forward_inputs["labels"],
                                                                       mask_lengths=prompt_lengths)
            else:
                raise NotImplementedError()
        else:
            if any(_gold_proofs is None for _gold_proofs in gold_proofs):
                raise Exception('Why pass here? might be a bug?')

            if lm_type == LMType.SEQ_2_SEQ:
                forward_inputs.update(prepare_tokenized_inputs(prompts_w_partial_proof, data_args.max_source_length))
                forward_inputs[proof_col] = prepare_tokenized_targets(gold_proofs,
                                                                      whole_proof_max_length)["input_ids"]
                forward_inputs[proof_col] = mask_labels_by_ignore_index(forward_inputs[proof_col])

            elif lm_type == LMType.CAUSAL:
                _prompts = [prompt + causal_lm_sep_token for prompt in prompts_w_partial_proof]

                forward_inputs.update(
                    prepare_tokenized_inputs(
                        _prompts,
                        data_args.max_source_length,
                        # add_special_tokens=False
                    ))

                # the padding is arbitrary
                forward_inputs[proof_col] = prepare_tokenized_targets(gold_proofs,
                                                                      whole_proof_max_length)["input_ids"]
                forward_inputs[proof_col] = mask_labels_by_ignore_index(forward_inputs[proof_col])
            else:
                raise NotImplementedError()

        # some models do not accept 'token_type_ids' as inputs
        if 'token_type_ids' in forward_inputs:
            forward_inputs.pop('token_type_ids', None)
        if "depth" in examples:
            forward_inputs["depth"] = examples["depth"]

        inputs_decoded = tokenizer.batch_decode(unmask_by_pad_token(forward_inputs['input_ids']))
        if 'labels' in forward_inputs:
            labels_decoded = tokenizer.batch_decode(unmask_by_pad_token(forward_inputs['labels']))
        else:
            labels_decoded = [None] * len(inputs_decoded)
        for i_example, (input_decoded, label_decoded) in enumerate(zip(inputs_decoded, labels_decoded)):
            logger.info('------------ [example=%d] tokenized inputs ----------------', i_example)
            logger.info(input_decoded)
            if label_decoded is not None:
                logger.info('------------ [example=%d] tokenized labels ----------------', i_example)
                logger.info(label_decoded)

        return forward_inputs

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset.set_transform(
            lambda examples: preprocess_function(examples, 'train'))
    else:
        train_dataset = None

    if training_args.do_eval or data_args.do_eval_in_outerloop:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_dataset.set_transform(
            lambda examples: preprocess_function(examples, 'eval'))
    else:
        eval_dataset = None

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        predict_dataset.set_transform(
            lambda examples: preprocess_function(examples, 'eval'))
    else:
         predict_dataset = None

    # Data collator
    label_pad_token_id = ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if lm_type == LMType.SEQ_2_SEQ:
        data_collator = RemoveUnusedColumnsCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            return_tensors='pt',
        )
    elif lm_type == LMType.CAUSAL:
        data_collator = RemoveUnusedColumnsCollator(
            return_tensors='pt',
        )
    else:
        raise NotImplementedError()

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_rouges(decoded_preds: List[str], decoded_labels: List[str]) -> Dict[str, Any]:
        # Some simple post-processing
        _decoded_preds, _decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=_decoded_preds, references=_decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        return result

    metric_funcs = {
        'strct': build_metrics('strict'),
        'extr_stps': build_metrics('allow_extra_steps'),
    }

    def compute_metrics(eval_preds, dataloader=None) -> Dict[str, Any]:
        if lm_type == LMType.CAUSAL:
            if padding == 'max_length':
                logger.warning('The generated sequence would have only 1 token, as padding="max_length" is specified.')

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        if dataloader is not None:
            examples = [example for example in dataloader.dataset]  # before collating
        else:
            raise Exception('unexpected')
            examples = [None] * len(preds)

        # Replace ignore_indexs used for padding as we can't decode them
        preds = unmask_by_pad_token(preds)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        if labels is not None:
            labels = unmask_by_pad_token(labels)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        else:
            decoded_labels = None

        decoded_prompts_from_examples = [
            tokenizer.decode(unmask_by_pad_token(examples[i]["input_ids"]), skip_special_tokens=True)
            for i in range(len(examples))
        ]
        decoded_labels_from_examples = [
            tokenizer.decode(unmask_by_pad_token(examples[i][proof_col]), skip_special_tokens=True)
            for i in range(len(examples))
        ]
        if decoded_labels is not None and tuple(decoded_labels) != tuple(decoded_labels_from_examples):
            raise Exception('Unexcected, may be a bug')

        results = {}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        results["gen_len"] = np.mean(prediction_lens)

        metrics: Dict[str, List[Any]] = defaultdict(list)
        for i_example, (prompt, proof_gt, proof_pred, example) in enumerate(zip(
                decoded_prompts_from_examples,
                decoded_labels_from_examples,
                decoded_preds,
                examples)):

            logger.info('')
            logger.info('')
            logger.info('================ compute_metrics() example=[%d] ================\n', i_example)

            if lm_type == LMType.CAUSAL and prompt in proof_pred:
                # the results from model generation include also the prompt
                proof_pred = proof_pred[len(prompt):]

            if example is not None:
                input_ids = example['input_ids']
                # input_ids = np.where(np.array(input_ids) != ignore_index, input_ids, tokenizer.pad_token_id)
                input_ids = unmask_by_pad_token(input_ids)
                decoded_input_ids = tokenizer.decode(input_ids, skip_special_tokens=True)

                context = re.sub(r'.*\$context\$ = (.*) ; \$proof\$.*', '\g<1>', decoded_input_ids)
                hypothesis = re.sub(r'.*\$hypothesis\$ = (.*) ; \$context\$.*', '\g<1>', decoded_input_ids)

                log_example(
                    context=context,
                    hypothesis=hypothesis,
                    gold_proofs=[proof_gt],
                    pred_proof=proof_pred,
                    logger=logger,
                )

                for metric_type, calc_metrics in metric_funcs.items():
                    try:
                        _metrics = calc_metrics(
                            [proof_gt],
                            proof_pred,
                            context=context,
                        )
                    except Exception as e:
                        logger.warning('calc_metrics() failed due to the following error. this sample will be skipped from metrics: %s', str(e))
                        _metrics = {}
                    depths = (['all', str(example['depth'])] if example.get('depth', None) is not None
                              else ['all', 'None'])
                    for depth in depths:
                        for metric_name, metric_val in _metrics.items():
                            metrics[f"{metric_type}.D-{depth}.{metric_name}"].append(metric_val)

                    log_texts, log_args = [], []
                    for metric_name, metric_val in sorted(_metrics.items()):
                        log_texts.append('%-20s: %5.2f')
                        log_args.extend([f"{metric_type}.{metric_name}", metric_val])
                    logger.info('------------   metrics  ------------\n' + '\n'.join(log_texts), *log_args)

            else:
                # XXX: why pass here?
                context = None
                hypothesis = None

        for metric_name, metric_vals in metrics.items():
            results[f"{metric_name}"] = np.mean(metric_vals)

        return results

    # Initialize our Trainer
    if training_args.remove_unused_columns:
        raise ValueError('remove_unused_columns=True is not allowed because we transform dataset instances on-the-fly for augumentation.')
    if lm_type == LMType.SEQ_2_SEQ:
        preprocess_logits_for_metrics = None
    elif lm_type == LMType.CAUSAL:
        #def preprocess_logits_for_metrics(logits, labels):
        #    if isinstance(logits, tuple):
        #        # Depending on the model and config, logits may contain extra tensors,
        #        # like past_key_values, but logits always come first
        #        logits = logits[0]
        #    return logits.argmax(dim=-1)

        preprocess_logits_for_metrics = None
        if data_args.proof_sampling == 'stepwise':
            raise ValueError('proof_sampling = "stepwise" is not suitable for LMType.CAUSAL')
    else:
        raise NotImplementedError()

    if lm_type == LMType.CAUSAL:
        """
            model.generate() with batch size >= 2 require hacks on special tokens.
            this preparation must be exactly when entering/exiting evaluate()/predict(),
            as other functions such as train() should not respect this hack.
            See [here](https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2)
        """

        padding_side_org = tokenizer.padding_side
        pad_token_org = tokenizer.pad_token
        pad_token_id_org = model.config.eos_token_id

        def generation_init_special_tokens():
            logger.info('generation_init_special_tokens() called!')
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        def generation_exit_special_tokens():
            logger.info('generation_exit_special_tokens() called!')
            tokenizer.padding_side = padding_side_org
            tokenizer.pad_token = pad_token_org
            model.config.pad_token_id = pad_token_id_org
    else:
        def generation_init_special_tokens():
            pass

        def generation_exit_special_tokens():
            pass

    def make_gen_kwargs():
        """
            As generation_init_special_tokens()/generation_exit_special_tokens() dynamically change
            tokenizer special tokens, we also have to generate gen_kwargs dynamically.
        """
        stopping_criteria = MaxTimeCriteriaWithWarning(data_args.generation_timeout)
        return {
            'top_k': data_args.generation_top_k,
            'stopping_criteria': [stopping_criteria],
            'num_return_sequences': data_args.generation_num_return_sequences,
            'pad_token_id': tokenizer.pad_token_id,
        }

    def generation_handled(func):
        def inner(self, *args, **kwargs):
            generation_init_special_tokens()
            results = func(self, *args, **kwargs, **make_gen_kwargs())
            generation_exit_special_tokens()
            return results
        return inner

    StepWiseGenerationTrainer.evaluate = generation_handled(StepWiseGenerationTrainer.evaluate)
    StepWiseGenerationTrainer.predict = generation_handled(StepWiseGenerationTrainer.predict)

    trainer = StepWiseGenerationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_steps=data_args.generation_max_proof_steps if data_args.proof_sampling == 'stepwise' else 1,
        compute_metrics=compute_metrics if training_args.predict_with_generate and not is_torch_tpu_available() else None,
        texts_to_inputs_func=lambda texts: prepare_tokenized_inputs(texts, data_args.max_source_length),
        is_finished_func=lambda text: len(get_stance_markers(text)) > 0,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        log_generation=data_args.log_generation,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if data_args.do_eval_in_outerloop:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(metric_key_prefix="eval")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        if data_args.num_return_sequences > 1:
            raise NotImplementedError()

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")

        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = unmask_by_pad_token(predictions)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    if data_args.interactive_mode is not None:

        def get_prediction(context: str, hypothesis: str) -> str:
            context = re.sub(r'\s+', ' ', re.sub(r'\n', ' ', context))
            instance = {
                'context': context,
                'hypothesis': hypothesis,
            }

            tmp = tempfile.mktemp()
            with open(tmp, 'w') as f_out:
                f_out.write(json.dumps(instance))

            extension = data_args.file_type
            if extension is None or extension not in ['json', 'csv']:
                raise ValueError()

            user_input_dataset = load_dataset(
                extension,
                data_files={'tmp': tmp},
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )['tmp']
            user_input_dataset.set_transform(
                lambda examples: preprocess_function(examples, 'eval'))
            results = trainer.predict(user_input_dataset,
                                      metric_key_prefix="predict")

            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    predictions = results.predictions
                    predictions = unmask_by_pad_token(predictions)
                    predictions = tokenizer.batch_decode(
                        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.strip() for pred in predictions]
                    return predictions[0]

            # TODO: implement the logic here
            raise NotImplementedError()

        if data_args.interactive_mode == 'console':
            while True:
                print('\n\n======================= interactive mode ========================')
                context = input('\ncontext:\n\n')
                hypothesis = input('\nhypothesis:\n\n')

                proof = get_prediction(context, hypothesis)
                proof = prettify_proof_text(proof)

                log_example(
                    context=context,
                    hypothesis=hypothesis,
                    pred_proof=proof,
                )

        elif data_args.interactive_mode == 'gradio':

            def predict(context: str, hypothesis: str):
                proof = get_prediction(context, hypothesis)
                proof = prettify_proof_text(proof)
                return proof

            demo = gr.Interface(
                fn=predict,
                inputs=[gr.Textbox(lines=10, placeholder='sent1: Allen is red\nsent2: Allen is blue'),
                        gr.Textbox(lines=1, placeholder='Allen is red')],
                outputs=['text'],
            )
            demo.launch(share=True, server_name='0.0.0.0', server_port=data_args.gradio_port)
        else:
            raise ValueError()

        return

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
