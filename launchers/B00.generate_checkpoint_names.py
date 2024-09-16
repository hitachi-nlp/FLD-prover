#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json
import re
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
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-06-22.neurip.additional.rebuttal')


    # ================================================================= LPT =================================================================
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-08.LPT'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-11.do_cache.pyarrow.node--8.proc-32'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-19.LPT.zero0'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-06-19.LPT.zero0.ALPT'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-0628.ALPT_first'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-07-08.augmentation'
    # TOP_DIR = './outputs.FLD-prover/01.train.py/2024-07-08.steps'

    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-07-09.BLPT.re_run.scratch')
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-07-09.BLPT.re_run.scratch.PT')


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


    # =================================== ALPT_strong ===================================
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-07-15.ALPT_strong')
    # TOP_DIR = Path('./outputs.FLD-prover//01.train.py/2024-08-05.ALPT_strong')


    # =================================== 2024-08-12.neurips_camera_ready.towards_best_corpora ========================================
    # TOP_DIR = Path('./outputs.FLD-prover//01.train.py/2024-08-12.neurips_camera_ready.towards_best_corpora')

    # =================================== 2024-08-16.neurips_camera_ready ========================================
    # TOP_DIR = Path('./outputs.FLD-prover//01.train.py/2024-08-16.neurips_camera_ready')
    # TOP_DIR = Path('./outputs.FLD-prover//01.train.py/2024-08-16.neurips_camera_ready.1')
    # TOP_DIR = Path('./outputs.FLD-prover//01.train.py/2024-08-16.neurips_camera_ready.2')
    # TOP_DIR = Path('./outputs.FLD-prover//01.train.py/2024-08-16.neurips_camera_ready.3')
    # TOP_DIR = Path('./outputs.FLD-prover//01.train.py/2024-08-16.neurips_camera_ready.4')
    # TOP_DIR = Path('./outputs.FLD-prover//01.train.py/2024-08-16.neurips_camera_ready.5')
    # TOP_DIR = Path('./outputs.FLD-prover//01.train.py/2024-08-16.neurips_camera_ready.6.rec_adam')
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-08-16.neurips_camera_ready.7.proof_intermediate_steps_prob')
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-08-30.fix_ref_prob')
    

    # =================================== 2024-09-03.toward_camera_ready ========================================
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-09-03.toward_camera_ready')
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-09-10.fix_rec_adam')
    TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-09-12.camera_ready')


    # =================================== 2024-09-06.llama3 ========================================
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-09-03.toward_camera_ready')
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-09-06.llama3')
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-09-06.llama3.node--1')


    # =================================== 2024-08-26.really_camera_ready ========================================
    # XXX: REALY CAMERA READY
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-08-26.really_camera_ready')


    # =================================== ./outputs.FLD-augmentation/00.augment.py/2024-08-25 ========================================
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/00.augment.py.2024-08-25')


    # =================================== 2024-09-08.ALPT_strong ========================================
    # TOP_DIR = Path('./outputs/01.train.py/2024-09-06.ABCI_debug')
    # TOP_DIR = Path('./outputs/01.train.py/2024-09-08.ALPT_strong')
#     TOP_DIR = Path('./outputs/01.train.py/2024-09-10.ALPT_strong.fix_rec_adam')


    # =================================== 2024-09-08.factorized_reasoning ========================================
    # TOP_DIR = Path('./outputs/01.train.py/2024-09-08.factorized_reasoning')
    # TOP_DIR = Path('./outputs/01.train.py/2024-09-10.factorized_reasoning.fix_rec_adam')



    # =================================== 2024-09-06.factorized_reasoning ========================================
    # TOP_DIR = Path('./outputs.FLD-prover/01.train.py/2024-09-06.factorized_reasoning')




    # ONLY_SHOW_EXISTING = False
    ONLY_SHOW_EXISTING = True


    CHECKPOINTS = [
        # ==================== NeurIPS ======================
        # 93,
        # 97,
        # 194,
        # 291,
        # 388,
        # 390,

        # 244,
        # 488,
        # 732,
        # 976,

        # 1952,

        # ==================== LPT ======================
        # 2928,
        # 9765,
        # 12000


        # ==================== transfer ======================
        # 352,   # FLD(eng,jpn,logical_formula)
        # 30,    # few-shot
        # 234,   # other datasets
        # 78,   # other datasets

        # -------------------- RWKV ----------------------
        # 390,

        # -------------------- ALPT_strong ----------------------
        # 390,
        # 780,
        # 1170,

        # 585,
        # 1170,
        # 1755,
        # 2340,

        # 1302,
        # 2604,
        # 3906,

        None,
    ]


    PARAMS = [
        # ======================== NeurIPS ========================
        # 'model_name',
        # 'logic_dataset_uname',
        # 'optimizer',
        # 'learning',
        # 'learning_rate',
        # 'rec_adam_fisher_coef',
        # 'weight_decay',
        # 'lr_scheduler_type',


        # ======================== LPT ========================
        # 'model_name',
        # 'logic_dataset_uname',
        # 'dataset_names',
        # 'logic_dataset_prob',
        # 'learning',
        # 'learning_rate',
        # 'rec_adam_fisher_coef',
        # 'augmentation',


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


        # ======================== ALPT_strong ========================
        # 'model_name',
        # 'logic_dataset_uname',
        # 'optimizer',
        # 'learning',
        # 'learning_rate',
        # 'rec_adam_fisher_coef',
        # 'weight_decay',
        # 'lr_scheduler_type',
        # 'augmentation',


        # ======================== ALPT_strong2024-08-16.neurips_camera_ready ========================
        # 'model_name',
        # 'logic_dataset_uname',
        # 'optimizer',
        # 'learning',
        # 'learning_rate',
        # 'rec_adam_fisher_coef',
        # 'weight_decay',
        # # 'lr_scheduler_type',
        # 'augmentation',
        # 'augmentation_prob',
        # 'surface_is_formula',


        # ======================== ALPT_strong2024-08-16.neurips_camera_ready.1 ========================
        # 'model_name',
        # 'logic_dataset_uname',
        # 'optimizer',
        # 'learning',
        # 'learning_rate',
        # 'rec_adam_fisher_coef',
        # 'weight_decay',
        # # 'lr_scheduler_type',
        # 'augmentation',


        # ======================== ALPT_strong2024-08-16.neurips_camera_ready.2 ========================
        # 'model_name',
        # 'logic_dataset_uname',
        # 'optimizer',
        # 'learning',
        # 'learning_rate',
        # 'rec_adam_fisher_coef',
        # 'weight_decay',
        # 'augmentation',
        # 'proof_intermediate_steps',



        'model_name',
        'logic_dataset_uname',
        'optimizer',
        'learning',
        'learning_rate',
        'rec_adam_fisher_coef',
        # 'weight_decay',
        'augmentation',
        'proof_intermediate_steps',
        'proof_intermediate_steps_prob',
        'max_grad_norm',
    ]





    print(f'# ================================================================= {TOP_DIR} =================================================================')
    input_dir = Path(TOP_DIR)
    setting_paths = sorted(input_dir.glob('**/*/lab.params.json'))
    for setting_path in setting_paths:
        for checkpoint in CHECKPOINTS:
            settings = json.load(open(str(setting_path)))
            _params = PARAMS.copy()

            if 'prompt_indicate_theorems' in settings and settings['prompt_indicate_theorems'] is True:
                _params += ['prompt_indicate_theorems']
            if 'augmentation_prob' in settings and settings['augmentation_prob'] != 1.0:
                _params += ['augmentation_prob']
            if 'prompt_emphasize_theorems' in settings and settings['prompt_emphasize_theorems'] is True:
                _params += ['prompt_emphasize_theorems']
            if 'block_size' in settings and settings['block_size'] != 2048:
                _params += ['block_size']
            if 'dataset_names' in settings and settings['dataset_names']:
                _params += ['dataset_names']
            if 'logic_dataset_prob' in settings and settings['logic_dataset_prob'] != 1.0:
                _params += ['logic_dataset_prob']
            if 'dataset_probs' in settings and settings['dataset_probs']:
                _params += ['dataset_probs']
            if 'formula_prob' in settings and settings['formula_prob'] != 0.0:
                _params += ['formula_prob']
            if 'paraphrase_contradiction' in settings and settings['paraphrase_contradiction'] is True:
                _params += ['paraphrase_contradiction']

            _settings = OrderedDict([
                (key, settings.get(key, None))
                for key in _params
            ])
            name = make_name(_settings, sep='__', short=True)

            if checkpoint is None:
                checkpoint_dirs = setting_path.parent.glob('checkpoint-*')
            else:
                checkpoint_dirs = [setting_path.parent / f"checkpoint-{checkpoint}"]

            for checkpoint_dir in checkpoint_dirs:
                if ONLY_SHOW_EXISTING:
                    if not checkpoint_dir.exists():
                        continue
                _checkpoint = checkpoint if checkpoint is not None else re.search(r'checkpoint-(\d+)', str(checkpoint_dir)).group(1)
                # print(f"'{name}.chk-{checkpoint}': '" + str(setting_path.parent / f"checkpoint-{checkpoint}',"))
                print(f"'{name}.chk-{_checkpoint}': '" + str(checkpoint_dir) + "',")


if __name__ == '__main__':
    main()
