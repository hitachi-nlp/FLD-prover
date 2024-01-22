#!/usr/bin/env python
import logging
from pathlib import Path

import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger

from lab import build_dir
from FLD_user_shared_settings import (
    get_base_setting,
    get_dataset_setting,
    get_batch_setting,
    get_save_eval_step_setting,
    get_model_setting,
    get_tokenizer_setting,
    get_learning_setting,
    get_generation_setting,
    make_output_dir,
    make_command,
    run_by_engine,
    maybe_option_value,
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # output_top_dir = Path('./outputs/A00.count_tokens.py/20231213.jpn')

    output_top_dir = Path('./outputs/A00.count_tokens.py/20230118.jpn.ICL')

    DATASETS_DIRS = [
        './outputs.FLD/00.create_corpus/20231203.jpn',
        './outputs.FLD/00.create_corpus/20231213.jpn',
        './outputs.FLD/00.create_corpus/20230118.jpn.ICL',
    ]

    FLD_dataset_unames = [
        # ---------------------------------- 20231213.jpn ------------------------------------
        # '20231213.jpn.D1_wo_dist',
        # '20231213.jpn.D1',
        # '20231213.jpn.D3',
        # '20231213.jpn.D8',

        # ---------------------------------- 20230118.jpn ------------------------------------

        # '20230118.jpn.wordnet.D3',
        # '20230118.jpn.wordnet.D3.argument_pred_arg_only',
        # '20230118.jpn.wordnet.D3.argument_pred_arg_only.no_kaku',
        # '20230118.jpn.BCCWJ.D3',
        # '20230118.jpn.punipuni.D3',

        # ---------------------------------- 20230118.jpn.ICL ------------------------------------
        '20230118.jpn.wordnet.D3.extension-3.distractor-10',
        '20230118.jpn.wordnet.D3.extension-3.distractor-5',
        '20230118.jpn.wordnet.D3.extension-3.distractor-3',
        '20230118.jpn.wordnet.D3.extension-2.distractor-5',
        '20230118.jpn.wordnet.D3.extension-2.distractor-3',
        '20230118.jpn.wordnet.D3.extension-1.distractor-5',
        '20230118.jpn.wordnet.D3.extension-1.distractor-3',
    ]

    model_names = [
        'line-corporation/japanese-large-lm-3.6b',
        'rinna/japanese-gpt-neox-3.6b',
        'cyberagent/calm2-7b',
        'stabilityai/japanese-stablelm-base-alpha-7b',
        'matsuo-lab/weblab-10b',
        'elyza/ELYZA-japanese-Llama-2-13b-fast',
        'stockmark/stockmark-13b',
        'pfnet/plamo-13b',
        'llm-jp/llm-jp-13b-v1.0',
        'tokyotech-llm/Swallow-13b-hf',
        'tokyotech-llm/Swallow-70b-hf',
        'tokyotech-llm/Swallow-70b-instruct-hf',
    ]

    # dry_run = True
    dry_run = False

    # engine = SubprocessEngine()
    engine = QsubEngine('ABCI', 'rt_C.small', n_resource=1)

    # --------------- run comaand ---------------

    for FLD_dataset_uname in FLD_dataset_unames:
        for model_name in model_names:
            setting = {
                'FLD_dataset_uname': FLD_dataset_uname,
                'model_name': model_name,
            }

            setting.update(
                get_dataset_setting(
                    'run_causal_prover',
                    dataset_uname=FLD_dataset_uname,
                    top_dirs=DATASETS_DIRS,
                    allow_not_found_splits=True,
                )
            )

            output_dir = build_dir(
                setting,
                top_dir=output_top_dir,
                short=True,
                dirname_ignore_params=[
                    'FLD_train_file',
                    'FLD_validation_file',
                    'FLD_test_file',
                ],
                save_params=True
            )

            command = ' '.join([
                'python ./scripts/count_tokens.py',
                f'--output_dir {str(output_dir)}',
                maybe_option_value('--fld_dataset_name', setting.get('FLD_dataset_name', None)),
                maybe_option_value('--fld_dataset_config_name', setting.get('FLD_dataset_config_name', None)),
                maybe_option_value('--fld_train_file', setting.get('FLD_train_file', None)),
                maybe_option_value('--fld_validation_file', setting.get('FLD_validation_file', None)),
                maybe_option_value('--fld_test_file', setting.get('FLD_test_file', None)),
                maybe_option_value('--tokenizer_name', setting.get('model_name', None)),
            ])

            run_by_engine(
                engine,
                command,
                output_dir,
                hours=2,
                dry_run=dry_run
            )


if __name__ == '__main__':
    main()
