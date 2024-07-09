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



    # =================================== neurips.additional ===================================
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-06-22.neurip.additional')


    # ================================================================= LPT =================================================================
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-08.LPT'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-11.do_cache.pyarrow.node--8.proc-32'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-19.LPT.zero0'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-19.LPT.zero0.ALPT'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-0628.ALPT_first'
    TOP_DIR = './outputs.FLD-prover/01.train.py/2024-07-08.augmentation'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-07-08.steps'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-07-08.ALPT.re_run'

    # ========================================================== transfer ==========================================================
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-19.transfer'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-22.transfer.formula'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-19.transfer.rec_adam_refactor'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-26.sft'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-28.sft'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-28.few_shot'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-28.sft.others'

    # =================================== RWKV ===================================
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-07-03.RWKV')

    # ONLY_SHOW_EXISTING = False
    ONLY_SHOW_EXISTING = True

    CHECKPOINTS = [
        # ==================== NeurIPS ======================
        # 97,
        # 194,
        # 291,
        388,
        390,

        # ==================== LPT ======================
        # 12000
        # 2928,


        # ==================== transfer ======================
        # 352,   # FLD(eng,jpn,logical_formula)
        # 30,    # few-shot
        # 234,   # other datasets
        # 78,   # other datasets

        # -------------------- RWKV ----------------------
        # 390,
    ]

    PARAMS = [
        # ======================== NeurIPS ========================
        # 'model_name',
        # 'logic_dataset_uname',
        # 'optimizer',
        # 'learning',
        # 'learning_rate',
        # 'rec_adam_fisher_coef',


        # ======================== LPT ========================
        'model_name',
        'logic_dataset_uname',
        'dataset_names',
        'logic_dataset_prob',
        'learning',
        'learning_rate',
        'rec_adam_fisher_coef',
        'augmentation',

        # ==================== transfer ======================
        # 'model_name',
        # 'logic_dataset_uname',
        # 'dataset_names',
        # # 'logic_dataset_prob',
        # 'surface_is_formula',
        # 'learning',
        # 'learning_rate',

        # ======================== RWKV ========================
        # 'model_name',
        # 'logic_dataset_uname',
        # 'dataset_names',
        # 'logic_dataset_prob',
        # 'learning',
        # 'learning_rate',

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
            checkpoint_dir = setting_path.parent / f"checkpoint-{checkpoint}"
            if ONLY_SHOW_EXISTING:
                if not checkpoint_dir.exists():
                    continue
            # print(f"'{name}.chk-{checkpoint}': '" + str(setting_path.parent / f"checkpoint-{checkpoint}',"))
            print(f"'{name}.chk-{checkpoint}': '" + str(checkpoint_dir) + "',")


if __name__ == '__main__':
    main()
