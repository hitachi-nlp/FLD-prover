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
    TOP_DIR = './outputs.FLD-prover/01.train.py/2024-4-23.refine_production.more'

    CHECKPOINTS = [
        97,
        194,
        291,
        388,
        485,
        582,
        679,
        776,
        873,
        970,
        1067,
        1164,
    ]

    PARAMS = [
        'logic_dataset_uname',
        # 'optimizer',
        # 'proof_intermediate_steps',
        # 'block_size',
        'learning',
        'learning_rate',
        'rec_adam_anneal_schedule',
        'rec_adam_fisher_coef',
        # 'lr_scheduler_type',
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
