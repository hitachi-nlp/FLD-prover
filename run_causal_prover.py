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

import deepspeed
import datasets
from datasets import interleave_datasets
import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    Seq2SeqTrainer,
)
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from peft import LoraConfig, TaskType as PeftTaskType, get_peft_model
from logger_setup import setup as setup_logger
from FLD_prover.data_processing import (
    preprocess_function as FLD_preprocess_function,
    compute_metrics as FLD_compute_metrics,
)
from FLD_prover.trainer import ForceCallMetricsSeq2SeqTrainer
from FLD_prover.tokenizers import load as load_tokenizer
from FLD_prover.lm_types import LMType
from FLD_prover.collators import RemoveUnusedColumnsCollator
from FLD_prover.generation import generation_handled
from FLD_prover.interactive import launch


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

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    do_eval_in_outerloop: bool = field(
        default=False,
    )
    file_type: Optional[str] = field(
        default=None, metadata={"help": "The input file type such as 'json' or 'csv'"}
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column of text field to be use from dataset"}
    )

    FLD_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    FLD_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    FLD_train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    FLD_validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    FLD_dataset_prob: Optional[float] = field(
        default=0.1,
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
    FLD_max_eval_samples: Optional[int] = field(
        default=None,
    )
    FLD_proof_eval_generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
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
    FLD_proof_eval_padding: Optional[str] = field(
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
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    FLD_proof_eval_generation_top_k: int = field(
        default=None,
    )

    FLD_proof_eval_generation_num_return_sequences: int = field(
        default=1,
    )

    FLD_proof_eval_generation_timeout: int = field(
        default=60,
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

        # if self.train_file is not None:
        #     extension = self.train_file.split(".")[-1]
        #     assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        # if self.validation_file is not None:
        #     extension = self.validation_file.split(".")[-1]
        #     assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
        pass
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

    # must be placed at top, so we extract string from sys.argv directry
    if any(arg.find('deepspeed') >= 0 for arg in sys.argv):
        deepspeed.init_distributed()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    # Initialize our Trainer
    if training_args.remove_unused_columns:
        raise ValueError('remove_unused_columns=True is not allowed because we transform dataset instances on-the-fly for augmentation.')

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
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
    def load_raw_dataset_by_name(dataset_name: str, dataset_config_name: str, streaming: bool):
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=streaming,
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=streaming,
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

        # extension = (
        #     train_file.split(".")[-1]
        #     if train_file is not None
        #     else validation_file.split(".")[-1]
        # )

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
            )
        else:
            raw_datasets = {}

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
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=streaming,
                **dataset_args,
            )
        return raw_datasets

    if data_args.dataset_name is not None:
        raw_datasets = load_raw_dataset_by_name(data_args.dataset_name,
                                                data_args.dataset_config_name,
                                                data_args.streaming)
    else:
        raw_datasets = load_raw_dataset_by_files(data_args.train_file,
                                                 data_args.validation_file,
                                                 data_args.file_type,
                                                 data_args.keep_linebreaks,
                                                 data_args.streaming)

    if data_args.FLD_dataset_name is not None:
        FLD_raw_datasets = load_raw_dataset_by_name(data_args.FLD_dataset_name,
                                                    data_args.FLD_dataset_config_name,
                                                    False)
    else:
        FLD_raw_datasets = load_raw_dataset_by_files(data_args.FLD_train_file,
                                                     data_args.FLD_validation_file,
                                                     'json',
                                                     False,
                                                     data_args.streaming)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
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

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = data_args.text_column_name or ("text" if "text" in column_names else column_names[0])

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
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def _maybe_FLD_preprocess(examples: Dict[str, List[Any]], mode: str):
        if "hypothesis" not in examples:
            return examples

        FLD_indexes = [i for i in range(len(examples["hypothesis"]))
                       if examples["hypothesis"][i] is not None]
        non_FLD_indexes = [i for i in range(len(examples["hypothesis"]))
                           if i not in FLD_indexes]
        num_FLD_examples = len(FLD_indexes)
        num_non_FLD_examples = len(non_FLD_indexes)

        FLD_examples = {
            key: [values[i] for i in FLD_indexes]
            for key, values in examples.items()
        }
        non_FLD_examples = {
            key: [values[i] for i in non_FLD_indexes]
            for key, values in examples.items()
        }

        if mode in ["train", "eval"]:
            FLD_preproc_split = "train"
            FLD_padding = "max_length" if data_args.FLD_dataset_prob != 1.0 else data_args.FLD_proof_eval_padding
            feature_names = ['input_ids', 'attention_mask', 'labels']

        elif mode == "FLD_proof_eval":
            FLD_preproc_split = "eval"
            FLD_padding = data_args.FLD_proof_eval_padding
            feature_names = list(FLD_examples.keys())

        else:
            raise ValueError()

        if num_FLD_examples > 0:
            FLD_processed = FLD_preprocess_function(
                FLD_examples,
                FLD_preproc_split,
                LMType.CAUSAL,
                tokenizer,
                prompt_prefix=data_args.source_prefix,
                padding=FLD_padding,
                max_source_length=block_size,
                max_target_length=block_size,
                proof_sampling=False,
                sample_negative_proof=False,
                no_subproof_for_unknown=False,
                include_prompt_for_causal_lm_loss=data_args.include_prompt_for_causal_lm_loss,
                log_examples=data_args.log_examples,
            )
        else:
            FLD_processed = {}

        if mode == "FLD_proof_eval":
            return FLD_processed
        else:
            if num_FLD_examples > 0 and num_non_FLD_examples > 0:
                processed = {
                    key: torch.concat((FLD_processed[key], torch.tensor(non_FLD_examples[key], dtype=FLD_processed[key].dtype)))
                    for key in feature_names
                }
            elif num_FLD_examples > 0:
                processed = {key: vals for key, vals in FLD_processed.items() if key in feature_names}
            else:
                processed = {key: vals for key, vals in non_FLD_examples.items() if key in feature_names}
            return processed

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
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
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )
    FLD_lm_datasets = FLD_raw_datasets

    # the below "map" does not work, because the trainer drops all the features before set_transform() is called,
    # as the features do not match forward() signatures.
    # if not data_args.streaming:
    #     FLD_lm_datasets = FLD_raw_datasets.map(
    #         lambda examples: _maybe_FLD_preprocess(examples, 'train'),
    #         batched=True,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #         desc="preprocessing FLD datasets",
    #     )
    # else:
    #     FLD_lm_datasets = FLD_raw_datasets.map(
    #         lambda examples: _maybe_FLD_preprocess(examples, 'train'),
    #         batched=True,
    #     )

    def make_interleave_datasets(dataset: Optional[Dataset], FLD_dataset: Optional[Dataset]):
        if data_args.FLD_dataset_prob == 0.0:
            if dataset is None:
                raise ValueError()
            return dataset
        elif data_args.FLD_dataset_prob == 1.0:
            if FLD_dataset is None:
                raise ValueError()
            return FLD_dataset

        datasets = []
        probs = []
        if dataset is not None:
            datasets.append(dataset)
            probs.append(1.0)
        if FLD_dataset is not None:
            datasets.append(FLD_dataset)
            probs.append(data_args.FLD_dataset_prob)
            probs[0] -= data_args.FLD_dataset_prob
        return interleave_datasets(
            datasets,
            probabilities=probs,
            seed=0,
            stopping_strategy="all_exhausted",
        )

    if training_args.do_train:
        if "train" not in lm_datasets and "train" not in FLD_lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = make_interleave_datasets(lm_datasets.get("train", None),
                                                 FLD_lm_datasets.get("train", None))
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    else:
        train_dataset = None

    if training_args.do_eval or data_args.do_eval_in_outerloop:
        if "validation" not in lm_datasets and "validation" not in FLD_lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = make_interleave_datasets(lm_datasets.get("validation", None),
                                                FLD_lm_datasets.get("validation", None))
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

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
        train_dataset.set_transform(
            lambda examples: _maybe_FLD_preprocess(examples, 'train'))
    if eval_dataset:
        eval_dataset.set_transform(
            lambda examples: _maybe_FLD_preprocess(examples, 'eval'))

    collator = RemoveUnusedColumnsCollator(return_tensors='pt')

    # hack for the training_args to fit into Seq2SeqTrainer()
    training_args.generation_config = None  # For Seq2SeqTrainer
    training_args.generation_max_length = block_size
    training_args.generation_num_beams = data_args.FLD_proof_eval_generation_num_beams
    training_args.predict_with_generate = True

    ForceCallMetricsSeq2SeqTrainer.evaluate = generation_handled(
        ForceCallMetricsSeq2SeqTrainer.evaluate,
        LMType.CAUSAL,
        tokenizer,
        model,
        timeout=data_args.FLD_proof_eval_generation_timeout,
        top_k=data_args.FLD_proof_eval_generation_top_k,
        num_return_sequences=data_args.FLD_proof_eval_generation_num_return_sequences,
    )
    ForceCallMetricsSeq2SeqTrainer.predict = generation_handled(
        ForceCallMetricsSeq2SeqTrainer.predict,
        LMType.CAUSAL,
        tokenizer,
        model,
        timeout=data_args.FLD_proof_eval_generation_timeout,
        top_k=data_args.FLD_proof_eval_generation_top_k,
        num_return_sequences=data_args.FLD_proof_eval_generation_num_return_sequences,
    )

    FLD_proof_eval_dataset = FLD_lm_datasets["validation"]
    FLD_proof_eval_dataset = FLD_proof_eval_dataset.select(
        range(min(len(FLD_proof_eval_dataset), data_args.FLD_max_eval_samples)),
    )
    FLD_proof_eval_dataset.set_transform(
        lambda examples: _maybe_FLD_preprocess(examples, 'FLD_proof_eval'))

    def _FLD_compute_metrics(eval_preds) -> Dict[str, Any]:
        return FLD_compute_metrics(
            eval_preds,
            tokenizer,
            FLD_proof_eval_dataset,
            LMType.CAUSAL,
        )

    class FLDEvaluationCallback(TrainerCallback):

        def __init__(self):
            self._FLD_seq2seq_trainer = ForceCallMetricsSeq2SeqTrainer(
                model,
                args=training_args,
                data_collator=collator,
                train_dataset=None,
                eval_dataset=FLD_proof_eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=_FLD_compute_metrics,
                # callbacks: Optional[List["TrainerCallback"]] = None,
                # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                # preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            )

        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            self._FLD_seq2seq_trainer.evaluate(
                metric_key_prefix="FLD_proof_eval"
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
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available() else None,

        callbacks=[FLDEvaluationCallback()],
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
    if data_args.do_eval_in_outerloop:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if data_args.interactive_mode is not None:
        launch(
            trainer,
            tokenizer,
            lambda examples: _maybe_FLD_preprocess(examples, 'FLD_proof_eval'),
            data_args.interactive_mode,
            gradio_port=data_args.gradio_port,
        )
        return

    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
