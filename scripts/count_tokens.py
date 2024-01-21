#!/usr/bin/env python
# coding=utf-8
import logging
import os
from typing import Optional, Dict, List, Any, Union, Tuple, Any
from pprint import pformat
import json
from pathlib import Path
import statistics

import click
import datasets
from datasets import DatasetDict
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
)
from logger_setup import setup as setup_logger
from FLD_prover.tokenizers import load as load_tokenizer
from FLD_task import load_deduction, serialize


logger = logging.getLogger(__name__)


@click.command()
@click.option('--output_dir', type=str, default=None)
@click.option('--fld_dataset_name', type=str, default=None)
@click.option('--fld_dataset_config_name', type=str, default=None)
@click.option('--fld_train_file', type=str, default=None)
@click.option('--fld_validation_file', type=str, default=None)
@click.option('--tokenizer_name', type=str, default=None)
@click.option('--log_level', type=str, default='INFO')
def main(
    output_dir,
    fld_dataset_name,
    fld_dataset_config_name,
    fld_train_file,
    fld_validation_file,
    tokenizer_name,
    log_level,
):
    logging.getLogger('absl').setLevel(logging.WARNING)
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    setup_logger(do_stderr=True, level=logging.INFO, block_other_handlers=True)

    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    def load_raw_dataset_by_name(dataset_name: str, dataset_config_name: str):
        raw_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=None,
            use_auth_token=True,
        )
        return raw_datasets

    def load_raw_dataset_by_files(train_file: Optional[str],
                                  validation_file: Optional[str],
                                  file_type: str,
                                  keep_linebreaks: bool):
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
                cache_dir=None,
                use_auth_token=True,
                **dataset_args,
            )
        else:
            raw_datasets = DatasetDict()

        return raw_datasets

    if fld_dataset_name is not None:
        fld_raw_datasets = load_raw_dataset_by_name(fld_dataset_name, fld_dataset_config_name)
    else:
        fld_raw_datasets = load_raw_dataset_by_files(fld_train_file,
                                                     fld_validation_file,
                                                     'json',
                                                     False)

    fld_raw_datasets = fld_raw_datasets.map(
        lambda example: load_deduction(example).dict(),
        batched=False,
        load_from_cache_file=False,
    )

    def count_tokens(example):
        # Tokenize the text and count the number of tokens
        serial = serialize(load_deduction(example),
                           stepwise=False,
                           sample_negative_proof=False,
                           newlines=False,
                           include_max_subproof_for_unknown=False,
                           proof_indicator=True,
                           instruction=False)
        text = serial.prompt + serial.proof
        return {'num_tokens': len(tokenizer.encode(text, add_special_tokens=True))}

    tokenizer = load_tokenizer(
        tokenizer_name,
        cache_dir=None,
        use_auth_token=True,
        use_fast_tokenizer=True,
        trust_remote_code=True,
    )
    train_dataset = fld_raw_datasets['train'].map(count_tokens)

    token_counts = train_dataset['num_tokens']

    count_stats = {
        'max_tokens': max(token_counts),
        'min_tokens': min(token_counts),
        'average_tokens': statistics.mean(token_counts),
        'std_dev_tokens': statistics.stdev(token_counts),
    }
    logger.info(pformat(count_stats))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(count_stats, open(output_dir / 'token_stats.json', 'w'), indent=2)


if __name__ == "__main__":
    main()
