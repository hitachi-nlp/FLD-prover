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
from datasets.download.download_config import DownloadConfig
from datasets import interleave_datasets, DatasetDict
import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from datasets import IterableDataset


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_callback import CallbackHandler
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
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
from FLD_prover.trainer import ForceCallMetricsSeq2SeqTrainer
from FLD_prover.tokenizers import load as load_tokenizer
from FLD_prover.lm_types import LMType
from FLD_prover.collators import RemoveUnusedColumnsCollator
from FLD_prover.generation import generation_handled
from FLD_prover.interactive import launch
from FLD_task import load_deduction


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

MAP = True  # temporary to keep old code


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

    dataset_names: Optional[str] = field(
        default=None, metadata={"help": "Dataset names separated by ::"}
    )
    dataset_config_names: Optional[str] = field(
        default=None, metadata={"help": "Dataset config names separated by ::"}
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
    logic_train_fileg: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
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
    random_sample_max_train_samples: bool = field(
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
    random_sample_max_eval_samples: bool = field(
        default=False,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    logic_max_eval_samples: Optional[int] = field(
        default=None,
    )
    random_sample_logic_max_eval_samples: bool = field(
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
        default=10,
    )

    logic_proof_eval_padding: Optional[str] = field(
        default="longest",
    )

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

    interactive_mode: str = field(
        default=None,
    )

    gradio_port: int = 8010

    log_examples: bool = field(
        default=False,
    )

    nccl_timeout: int = field(
        default=1800,
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    logging.getLogger().handlers.clear()  # remove handler automatically added
    setup_logger(do_stderr=True, level=logging.INFO)
    logging.getLogger('absl').setLevel(logging.WARNING)
    # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    warnings.filterwarnings("ignore", message="is incompatible with gradient checkpointing. Setting")


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # must be placed at top, so we extract string from sys.argv directly
    if any(arg.find('--deepspeed') >= 0 for arg in sys.argv):
        deepspeed.init_distributed(timeout=datetime.timedelta(seconds=training_args.ddp_timeout))
    # https://github.com/huggingface/accelerate/issues/223
    # torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=data_args.nccl_timeout))

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    # Initialize our Trainer
    if training_args.remove_unused_columns:
        raise ValueError(
            'remove_unused_columns=True is not allowed because we transform dataset instances on-the-fly for augmentation.')

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    def load_raw_dataset_by_name(dataset_name: str,
                                 dataset_config_name: str,
                                 streaming: bool):
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=streaming,
            # download_config=DownloadConfig(resume_download=True),
        )
        if "validation" not in raw_datasets.keys():
            if "dev" in raw_datasets.keys():
                raw_datasets["validation"] = raw_datasets["dev"]
            else:
                raw_datasets["validation"] = load_dataset(
                    dataset_name,
                    dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    # split=f"train",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=streaming,
                    # download_config=DownloadConfig(resume_download=True),
                )
                raw_datasets["train"] = load_dataset(
                    dataset_name,
                    dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    # split=f"train",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=streaming,
                    # download_config=DownloadConfig(resume_download=True),
                )
        return raw_datasets

    def load_raw_dataset_by_files(train_file: Optional[str],
                                  validation_file: Optional[str],
                                  file_type: str,
                                  keep_linebreaks: bool,
                                  streaming: bool):
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

        if len(data_files) > 0:
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=streaming,
                **dataset_args,
                # download_config=DownloadConfig(resume_download=True),
            )

            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=streaming,
                    **dataset_args,
                    # download_config=DownloadConfig(resume_download=True),
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=streaming,
                    **dataset_args,
                    # download_config=DownloadConfig(resume_download=True),
                )
        else:
            raw_datasets = DatasetDict()

        return raw_datasets

    dataset_names = [name or None for name in data_args.dataset_names.split(
        '::')] if data_args.dataset_names is not None else []
    dataset_config_names = [name or None for name in data_args.dataset_config_names.split(
        '::')] if data_args.dataset_config_names is not None else []
    train_files = [name or None for name in data_args.train_files.split('::')] if data_args.train_files is not None else []
    validation_files = [name or None for name in data_args.validation_files.split(
        '::')] if data_args.validation_files is not None else []
    file_types = [name or None for name in data_args.file_types.split('::')] if data_args.file_types is not None else []
    dataset_probs = [float(prob) for prob in data_args.dataset_probs.split('::')
                     ] if data_args.dataset_probs is not None else []
    raw_datasets_list = []
    if len(dataset_names) > 0:
        for i in range(len(dataset_names)):
            raw_datasets_list.append(load_raw_dataset_by_name(dataset_names[i],
                                                              dataset_config_names[i] if dataset_config_names[i] != 'None' else None,
                                                              data_args.streaming))
    else:
        for i in range(len(train_files)):
            raw_datasets_list.append(load_raw_dataset_by_files(train_files[i],
                                                               validation_files[i] if validation_files[i] != 'None' else None,
                                                               file_types[i] if file_types[i] != 'None' else 'json',
                                                               data_args.keep_linebreaks,
                                                               data_args.streaming))

    logic_dataset_streaming = data_args.streaming
    if data_args.logic_dataset_name is not None:
        logic_raw_datasets = load_raw_dataset_by_name(data_args.logic_dataset_name,
                                                      data_args.logic_dataset_config_name,
                                                      logic_dataset_streaming)
    else:
        logic_raw_datasets = load_raw_dataset_by_files(data_args.logic_train_fileg,
                                                       data_args.logic_validation_file,
                                                       'json',
                                                       False,
                                                       logic_dataset_streaming)

    if dataloader_num_worker > 1\
            and data_args.preprocess_batch_size > 10:
    if data_args.logic_dataset_type == 'FLD':

        # load and dump once to normalize the schema from different versions of datasets.
        # to always reflect the modification of the preprocessing
        # load_from_cache_file=False to ensure that the change of source code is immediately reflect on.

        def FLD_unify_schema(examples: Dict[str, List[Any]]):
            keys = list(examples.keys())
            batch_size = len(examples[keys[0]])
            examples_list = [
                {key: values[i] for key, values in examples.items()}
                for i in range(batch_size)
            ]
            examples_list = [
                load_deduction(example).dict()
                for example in examples_list
            ]
            return {key: [examples_list[i][key] for i in range(batch_size)] for key in keys}
        logic_raw_datasets = logic_raw_datasets.map(
            # lambda example: load_deduction(example).dict(),
            FLD_unify_schema,
            batched=True,
            batch_size=data_args.preprocess_batch_size,
            **({} if logic_dataset_streaming else {'load_from_cache_file': False}),
        )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
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

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
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
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

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

    if data_args.preprocessing_num_workers >= 2 and data_args.preprocess_batch_size > 10:
        logger.critical('kind warning: dataset preprocessing with multiple workers and large batch size may hang without any error message.'
                        '\nSee: https://discuss.huggingface.co/t/datasets-mapper-hanging-issue/32995'
                        '\nNote that the fix introduced in the link did not work for me')

    if len(raw_datasets_list) == 0:
        lm_datasets_list = []
    else:
        lm_datasets_list = []

        for raw_datasets in raw_datasets_list:
            if training_args.do_train:
                column_names = list(raw_datasets["train"].features)
            elif training_args.do_eval or data_args.do_eval_in_outerloop:
                column_names = list(raw_datasets["validation"].features)
            else:
                column_names = None
            text_column_name = data_args.text_column_name\
                or ("text" if "text" in column_names else column_names[0]) if column_names is not None else None

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

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
            def group_texts(examples):
                # Concatenate all texts.
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

            with training_args.main_process_first(desc="dataset map tokenization"):
                if not data_args.streaming:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        batch_size=data_args.preprocess_batch_size,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on dataset",
                    )
                else:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        batch_size=data_args.preprocess_batch_size,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                    )

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
            # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
            # to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

            with training_args.main_process_first(desc="grouping texts together"):
                if not data_args.streaming:
                    lm_datasets = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                        batch_size=data_args.preprocess_batch_size,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc=f"Grouping texts in chunks of {block_size}",
                    )
                else:
                    lm_datasets = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                        batch_size=data_args.preprocess_batch_size,
                        num_proc=data_args.preprocessing_num_workers,
                    )

            lm_datasets_list.append(lm_datasets)

    preprocessor_args = [
        LMType.CAUSAL,
        tokenizer,
    ]
    preprocessor_kwargs = {
        'prompt_prefix': data_args.source_prefix,
        # 'padding': logic_padding,
        'max_source_length': block_size,
        'max_target_length': block_size,
        'proof_sampling': False,
        'sample_negative_proof': False,
        'no_subproof_for_unknown': data_args.no_subproof_for_unknown,
        'include_prompt_for_causal_lm_loss': data_args.include_prompt_for_causal_lm_loss,
        'instruction': data_args.instruction,
        'log_examples': data_args.log_examples,
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
    logic_data_processor = processor_cls(*preprocessor_args, **preprocessor_kwargs)

    def _maybe_logic_preprocess(examples: Dict[str, List[Any]], mode: str):
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
        if data_args.log_examples:
            for i_example in range(num_non_logic_examples):
                logger.info(
                    '------------------------------ preprocess_function [non-FLD example=%d] ------------------------------', i_example)
                for key, values in non_logic_examples.items():
                    logger.info('%s: "%s"', key, values[i_example])

        if mode in ["train", "eval"]:
            logic_preproc_split = "train"
            # logic_padding = "max_length" if data_args.logic_dataset_prob != 1.0 else data_args.logic_proof_eval_padding
            logic_padding = "max_length"   # 'longest' leads to error as the shape of tensors will be diferrent in a batch
            feature_names = ['input_ids', 'attention_mask', 'labels']

        elif mode == "proof_eval":
            logic_preproc_split = "eval"
            logic_padding = data_args.logic_proof_eval_padding
            feature_names = list(logic_examples.keys())

        else:
            raise ValueError()

        if num_logic_examples > 0:
            logic_processed = logic_data_processor.preprocess(
                logic_examples,
                logic_preproc_split,
                padding=logic_padding,
            )

        else:
            logic_processed = {}

        if mode == "proof_eval":
            return logic_processed

        else:
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

    logic_lm_datasets = logic_raw_datasets

    def make_interleave_datasets(datasets: List[Dataset], logic_dataset: Optional[Dataset]):
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

        probs = [dataset_prob_tot * dataset_probs[i] / sum(dataset_probs) for i in range(len(datasets))]
        if logic_dataset is not None:
            datasets.append(logic_dataset)
            probs.append(logic_dataset_prob)

        if len(datasets) == 1:
            return datasets[0]
        else:
            if any(max_sample_arg is not None for max_sample_arg in [data_args.max_train_samples,
                                                                     data_args.random_sample_max_train_samples,
                                                                     data_args.max_eval_samples,
                                                                     data_args.random_sample_max_eval_samples,
                                                                     data_args.logic_max_eval_samples,
                                                                     data_args.random_sample_logic_max_eval_samples]):
                logger.warning('[kind warning] max sample seems to be set, with which only few datasets might be sampled.')
            return interleave_datasets(
                datasets,
                probabilities=probs,
                seed=0,
                stopping_strategy="all_exhausted",
            )

    if training_args.do_train:
        train_dataset = make_interleave_datasets([lm_datasets["train"] for lm_datasets in lm_datasets_list],
                                                 logic_lm_datasets.get("train", None))

        if data_args.num_train_examples_skip > 0:
            logger.info('skip %d examples from the training dataset', data_args.num_train_examples_skip)
            train_dataset = train_dataset.skip(data_args.num_train_examples_skip)

        if data_args.max_train_samples is not None:
            if isinstance(train_dataset, IterableDataset):
                train_dataset = train_dataset.take(data_args.max_train_samples)
            else:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                if data_args.random_sample_max_train_samples:
                    indexes = np.random.choice(len(train_dataset), max_train_samples, replace=False)
                else:
                    indexes = np.arange(max_train_samples)
                train_dataset = train_dataset.select(indexes)
    else:
        train_dataset = None

    if training_args.do_eval or data_args.do_eval_in_outerloop:
        eval_dataset = make_interleave_datasets([lm_datasets["validation"] for lm_datasets in lm_datasets_list],
                                                logic_lm_datasets.get("validation", None))
        if data_args.max_eval_samples is not None:
            if isinstance(eval_dataset, IterableDataset):
                eval_dataset = eval_dataset.take(data_args.max_eval_samples)
            else:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                if data_args.random_sample_max_eval_samples:
                    indexes = np.random.choice(len(eval_dataset), max_eval_samples, replace=False)
                else:
                    indexes = np.arange(max_eval_samples)
                eval_dataset = eval_dataset.select(indexes)

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

    # We set FLD preprocesssing function to the interleaved datasets.
    # Setting preprocesssing function directly to FLD_lm_datasets, e.g., FLD_lm_datasets["train"].set_transform(), does not work
    # as interleave_datasets() does not respect that processing in the current implementation
    if train_dataset:
        if MAP:
            train_dataset = train_dataset.map(
                lambda examples: _maybe_logic_preprocess(examples, 'train'),
                batched=True,
                batch_size=data_args.preprocess_batch_size,
                num_proc=data_args.preprocessing_num_workers,
            )
        else:
            train_dataset.set_transform(
                lambda examples: _maybe_logic_preprocess(examples, 'train'),
                num_proc=data_args.preprocessing_num_workers,
            )
    if eval_dataset:
        if MAP:
            eval_dataset = eval_dataset.map(
                lambda examples: _maybe_logic_preprocess(examples, 'eval'),
                batched=True,
                batch_size=data_args.preprocess_batch_size,
                num_proc=data_args.preprocessing_num_workers,
            )
        else:
            eval_dataset.set_transform(
                lambda examples: _maybe_logic_preprocess(examples, 'eval'),
                num_proc=data_args.preprocessing_num_workers,
            )

    collator = RemoveUnusedColumnsCollator(return_tensors='pt')

    new_generation_config = GenerationConfig.from_model_config(config)
    new_generation_config.max_time = data_args.generation_timeout
    training_args.generation_config = new_generation_config
    training_args.predict_with_generate = True
    generation_handle_args = [
        LMType.CAUSAL,
        tokenizer,
        model,
    ]
    generation_max_length = min(data_args.generation_max_length + 1, model.config.max_position_embeddings)
    generation_handled_kwargs = {
        'eos_token_id': tokenizer.eos_token_id,
        'top_k': data_args.generation_top_k,
        'num_beams': data_args.generation_num_beams,
        'num_return_sequences': data_args.generation_num_return_sequences,
        'do_sample': data_args.generation_do_sample,
        'temperature': data_args.generation_temperature,
        'repetition_penalty': data_args.generation_repetition_penalty,
        'max_length': generation_max_length,
        'max_new_tokens': data_args.generation_max_new_tokens,
    }
    ForceCallMetricsSeq2SeqTrainer.evaluate = generation_handled(
        ForceCallMetricsSeq2SeqTrainer.evaluate,
        *generation_handle_args,
        timeout_from_call=data_args.evaluation_timeout,
        timeout_msg_title='generation aborted because "evaluate()" took too long',
        **generation_handled_kwargs,
    )
    ForceCallMetricsSeq2SeqTrainer.predict = generation_handled(
        ForceCallMetricsSeq2SeqTrainer.predict,
        *generation_handle_args,
        timeout_from_call=data_args.evaluation_timeout,
        timeout_msg_title='generation aborted because "evaluate()" took too long',
        **generation_handled_kwargs,
    )

    if "validation" in logic_lm_datasets:
        logic_eval_dataset = logic_lm_datasets["validation"]

        if MAP:
            generation_handled_map = generation_handled(
                logic_eval_dataset.map,
                *generation_handle_args,
                **generation_handled_kwargs,
                is_generate_func=False,
            )
            logic_eval_dataset = generation_handled_map(
                lambda examples: _maybe_logic_preprocess(examples, 'proof_eval'),
                batched=True,
                batch_size=data_args.preprocess_batch_size,
                num_proc=data_args.preprocessing_num_workers,
            )
        else:
            logic_eval_dataset.set_transform(
                lambda examples: _maybe_logic_preprocess(examples, 'proof_eval'))

        if data_args.logic_max_eval_samples is not None:
            if isinstance(logic_eval_dataset, IterableDataset):
                if data_args.random_sample_logic_max_eval_samples:
                    logger.warning('random_sample_logic_max_eval_samples is ignored because of the streaming mode')
                logic_eval_dataset = logic_eval_dataset.take(data_args.logic_max_eval_samples)
            else:
                if data_args.logic_max_eval_samples < len(logic_eval_dataset):
                    if data_args.random_sample_logic_max_eval_samples:
                        indexes = np.random.choice(len(logic_eval_dataset),
                                                   data_args.logic_max_eval_samples, replace=False)
                    else:
                        indexes = np.arange(data_args.logic_max_eval_samples)
                    logic_eval_dataset = logic_eval_dataset.select(indexes)
    else:
        logic_eval_dataset = None

    logic_data_processor.eval_dataset = logic_eval_dataset
    logic_compute_metrics = logic_data_processor.compute_metrics

    def _build_logic_seq2seq_trainer(other_trainer: Optional[Trainer] = None,
                                   do_compute_metrics=True):
        return ForceCallMetricsSeq2SeqTrainer(
            model,
            other=other_trainer,
            args=training_args,
            data_collator=collator,
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
                metric_key_prefix="proof_eval"
            )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,

        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=collator,
        compute_metrics = compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available() else None,
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
        launch(
            _build_logic_seq2seq_trainer(other_trainer=trainer, do_compute_metrics=False),
            tokenizer,
            lambda examples: _maybe_logic_preprocess(examples, 'proof_eval'),
            data_args.interactive_mode,
            gradio_port=data_args.gradio_port,
        )
        return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
