#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
from collections import OrderedDict

# from logger_setup import setup as setup_logger
import click
from lab import make_name


# logger = logging.getLogger(__name__)


@click.command()
def main():
    # setup_logger()

    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-4-20.production.additional.additional/'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-4-22.production.llama3/'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-4-23.refine_production'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-4-24.finalize_production'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-4-24.finalize_production.additional'

    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-4-24.finalize_production.other_llms'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-4-24.finalize_production.other_llms.mistral_tokenizer'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-05-03.ablation'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-05-08.ref_prob'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-4-24.finalize_production.other_llms.tokenizer_unk'
    # TOP_DIR = './outputs.FLD-prover//01.train.py/2024-5-13.large_models'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-5-19.flight'

    # ================================================================= LPT =================================================================
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-08.LPT'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-11.do_cache.pyarrow.node--8.proc-32'
    TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-19.LPT.zero0'


    # ========================================================== transfer ==========================================================
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-19.transfer'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-22.transfer.formula'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-19.transfer.rec_adam_refactor'
    TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-26.sft'


    CHECKPOINTS = [
        # -------------------- NeurIPS ----------------------
        # 97,
        # 194,
        # 291,
        # 388,

        # -------------------- LPT ----------------------
        12000

        # -------------------- transfer ----------------------
        # 388,
        352,
    ]

    PARAMS = [
        # ------------------------ NeurIPS ------------------------
        # 'model_name',
        # 'logic_dataset_uname',
        # 'optimizer',
        # 'learning',
        # 'learning_rate',
        # 'rec_adam_fisher_coef',

        # ------------------------ LPT ------------------------
        'model_name',
        'logic_dataset_uname',
        'dataset_names',
        'logic_dataset_prob',
        'learning',
        'learning_rate',

        # -------------------- transfer ----------------------

        'model_name',
        'logic_dataset_uname',
        'dataset_names',
        # 'logic_dataset_prob',
        'surface_is_formula',
        'learning',
        'learning_rate',

    ]

    input_dir = Path(TOP_DIR)
    setting_paths = sorted(input_dir.glob('**/*/lab.params.json'))
    for setting_path in setting_paths:
        for checkpoint in CHECKPOINTS:
            settings = json.load(open(str(setting_path)))
            _settings = OrderedDict([
                (key, settings[key])
                for key in PARAMS
            ])
            name = make_name(_settings, sep='__', short=True)
            print(f"'{name}.chk-{checkpoint}': '" + str(setting_path.parent / f"checkpoint-{checkpoint}',"))


if __name__ == '__main__':
    main()
