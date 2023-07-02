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

import logging
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union, Tuple, Any
from collections import defaultdict

import torch
import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from datasets import Features, Value
from filelock import FileLock

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from logger_setup import setup as setup_logger

from FLD_task import build_metrics, prettify_context_text, prettify_proof_text
from FLD_task.proof import get_stance_markers
from FLD_prover.utils import tokenize_with_log
from FLD_prover import (
    StepWiseGenerationTrainer,
    preprocess_examples_train,
    preprocess_examples_eval,
)
import line_profiling

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.29.0.dev0")

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
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
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
        metadata={"help": "[stepwise|single_shot]"},
    )

    sample_negative_proof: bool = field(
        default=False,
    )

    generation_max_proof_steps: int = field(
        default=20,
    )

    generation_top_k: int = field(
        default=30,
    )

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
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
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


class RemoveUnusedColumnsCollator(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        for feature in features:
            if "depth" in feature:
                feature.pop("depth", None)
        return super().__call__(features, return_tensors=return_tensors)


def main():
    logging.getLogger().handlers.clear()  # remove handler automatically added
    setup_logger(do_stderr=True, level=logging.INFO)

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger.info('reloading GenerationConfig to reflect the specified parameters: top_k= %s', data_args.generation_top_k)
    training_args.generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path,
                                                                       top_k=data_args.generation_top_k)

    if training_args.dataloader_num_workers > 0:
        raise Exception(f'dataloader_num_workers({training_args.dataloader_num_workers}) > 0 is extemely slow in our program. We strongly recommend to ppecify it as 0.')

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

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

        extension = data_args.file_type
        if extension is None or extension not in ['json', 'csv']:
            raise ValueError()

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # torch._C._dynamo.disable
    # model = torch.compile(model)

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

    if model.config.decoder_start_token_id is None:
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

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

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

    # Get the column names for input/target.
    # dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    # if data_args.text_column is None:
    #     text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    # else:
    #     text_column = data_args.text_column
    #     if text_column not in column_names:
    #         raise ValueError(
    #             f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
    #         )
    # if data_args.summary_column is None:
    #     summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    # else:
    #     summary_column = data_args.summary_column
    #     if summary_column not in column_names:
    #         raise ValueError(
    #             f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
    #         )
    context_column = 'context'
    next_step_column = 'next_step'
    gold_proof_column = 'gold_proof'

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
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
            if context and next_step:
                inputs.append(context)
                targets.append(next_step)
                gold_proofs.append(gold_proof)
                if data_args.log_examples:
                    logger.info('context    [%d] : "%s"', i_example, context)
                    logger.info('next_step [%d] : "%s"', i_example, next_step)
                    logger.info('gold_proof [%d] : "%s"', i_example, gold_proof)
        return inputs, targets, gold_proofs

    def prepare_model_inputs(inputs: List[str], max_length: int) -> Dict[str, List[Any]]:
        return tokenize_with_log(tokenizer, text=inputs, max_length=max_length, padding=padding, truncation=True)

    def prepare_model_targets(targets: List[str], max_length: int) -> Dict[str, List[Any]]:
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenize_with_log(tokenizer, text_target=targets, max_length=max_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        return labels

    @profile
    def preprocess_function(examples: Dict[str, List[Any]],
                            split: str,
                            max_source_length: int,
                            max_target_length: int) -> Dict[str, List[Any]]:
        if split == 'train':
            examples = preprocess_examples_train(
                examples,
                stepwise=data_args.proof_sampling == 'stepwise',
                sample_negative_proof=data_args.sample_negative_proof,
            )
        elif split == 'eval':
            examples = preprocess_examples_eval(examples)
        else:
            raise ValueError()

        inputs, targets, gold_proofs = extract_serials(examples)
        inputs = [prefix + inp for inp in inputs]

        model_inputs = prepare_model_inputs(inputs, max_source_length)

        if split == 'train':
            model_inputs["labels"] = prepare_model_targets(targets, max_target_length)["input_ids"]
        else:
            model_inputs["labels"] = prepare_model_targets(gold_proofs, max_target_length)["input_ids"]

        model_inputs["depth"] = examples["depth"]

        return model_inputs

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # with training_args.main_process_first(desc="train dataset map pre-processing"):
        #     train_dataset = train_dataset.map(
        #         preprocess_function,
        #         batched=True,
        #         num_proc=data_args.preprocessing_num_workers,
        #         remove_columns=column_names,
        #         load_from_cache_file=not data_args.overwrite_cache,
        #         desc="Running tokenizer on train dataset",
        #     )
        train_dataset.set_transform(
            lambda examples: preprocess_function(examples, 'train',
                                                 max_source_length=data_args.max_source_length,
                                                 max_target_length=data_args.max_target_length))

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        # with training_args.main_process_first(desc="validation dataset map pre-processing"):
        #     eval_dataset = eval_dataset.map(
        #         preprocess_function,
        #         batched=True,
        #         num_proc=data_args.preprocessing_num_workers,
        #         remove_columns=column_names,
        #         load_from_cache_file=not data_args.overwrite_cache,
        #         desc="Running tokenizer on validation dataset",
        #     )
        eval_dataset.set_transform(
            lambda examples: preprocess_function(examples, 'eval',
                                                 max_source_length=data_args.max_source_length,
                                                 max_target_length=data_args.max_target_length * 20))

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        # with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        #     predict_dataset = predict_dataset.map(
        #         preprocess_function,
        #         batched=True,
        #         num_proc=data_args.preprocessing_num_workers,
        #         remove_columns=column_names,
        #         load_from_cache_file=not data_args.overwrite_cache,
        #         desc="Running tokenizer on prediction dataset",
        #     )
        predict_dataset.set_transform(
            lambda examples: preprocess_function(examples, 'eval',
                                                 max_source_length=data_args.max_source_length,
                                                 max_target_length=data_args.max_target_length * 20))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = RemoveUnusedColumnsCollator(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

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
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # result = compute_rouges(decoded_preds, decoded_labels)
        results = {}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        results["gen_len"] = np.mean(prediction_lens)

        if dataloader is not None:
            examples = [example for example in dataloader.dataset]  # before collating
        else:
            examples = [None] * len(decoded_labels)

        metrics: Dict[str, List[Any]] = defaultdict(list)
        for i_example, (proof_gt, proof_pred, example) in enumerate(zip(decoded_labels, decoded_preds, examples)):
            logger.info('')
            logger.info('')
            logger.info('================ compute_metrics() example=[%d] ================\n', i_example)

            if example is not None:
                input_ids = example['input_ids']
                input_ids = np.where(np.array(input_ids) != -100, input_ids, tokenizer.pad_token_id)
                decoded_input_ids = tokenizer.decode(input_ids, skip_special_tokens=True)

                context = re.sub(r'.*\$context\$ = (.*) ; \$proof\$.*', '\g<1>', decoded_input_ids)
                hypothesis = re.sub(r'.*\$hypothesis\$ = (.*) ; \$context\$.*', '\g<1>', decoded_input_ids)
            else:
                context = None
                hypothesis = None

            if context is not None:
                try:
                    logger.info('------------ context ------------\n\n%s\n', prettify_context_text(context, indent=4))
                except:
                    logger.fatal('prettify_context failed for the following context. This is unexpected:%s', context)

            if hypothesis is not None:
                logger.info('------------ hypothesis ------------\n\n    %s\n', hypothesis)

            logger.info('------------ proof_gold ------------\n\n%s\n', prettify_proof_text(proof_gt, indent=4))

            logger.info('------------ proof_pred ------------\n\n%s\n', prettify_proof_text(proof_pred, indent=4))

            for metric_type, calc_metrics in metric_funcs.items():
                _metrics = calc_metrics(
                    [proof_gt],
                    proof_pred,
                )
                depths = ['all'] if example is None else ['all', str(example['depth'])]
                for depth in depths:
                    for metric_name, metric_val in _metrics.items():
                        metrics[f"{metric_type}.D-{depth}.{metric_name}"].append(metric_val)

                log_texts, log_args = [], []
                for metric_name, metric_val in sorted(_metrics.items()):
                    log_texts.append('%-20s: %5.2f')
                    log_args.extend([f"{metric_type}.{metric_name}", metric_val])
                logger.info('------------   metrics  ------------\n' + '\n'.join(log_texts), *log_args)

        for metric_name, metric_vals in metrics.items():
            results[f"{metric_name}"] = np.mean(metric_vals)

        return results

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # Initialize our Trainer
    if training_args.remove_unused_columns:
        raise ValueError('remove_unused_columns=True is not allowed because we transform dataset instances on-the-fly for augumentation.')
    # trainer = Seq2SeqTrainer(
    trainer = StepWiseGenerationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        max_steps=data_args.generation_max_proof_steps,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        texts_to_inputs_func=lambda texts: prepare_model_inputs(texts, data_args.max_source_length),
        is_finished_func=lambda text: len(get_stance_markers(text)) > 0,
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
        trainer.save_model()  # Saves the tokenizer too for easy upload

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
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

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
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

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
