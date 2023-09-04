#!/usr/bin/env python
import logging
from pathlib import Path
import json
from typing import Optional
import math


import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger

from experimental_setting import (
    get_config,
    get_default_config_name,
    get_dataset_setting,
    get_batch_setting,
    make_output_dir,
    make_command,
    run_by_engine,
    CheckpointSpec,
    ICML_2023_NL_TRANSFER_MAJOR_DATASETS,
    ICML_2023_NL_TRANSFER_MAJOR_DATASETS_LARGE_DEPTH,
    LEARNING_SETTINGS,
    make_val_interval_setting,
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # output_top_dir = Path('./outputs/02.interactive.py/dev_null')
    # output_top_dir = Path('./outputs/02.interactive.py/20230802.case_study_finalize.fix.rerun/dtst_nm=20230729.case_study_finalize.D8')
    # output_top_dir = Path('./outputs/02.interactive.py/tmp')

    # # -- D3 --
    # checkpoint_top_dir = Path('./outputs/01.train.py/20230711.finalize/dtst_nm=20230711.finalize.D3/dtst_1_nm=None/bs_cnfg_nm=FLNLcorpus.20220827.base/chckpnt_nm=t5-base/ckpt_lrt=None/ckpt_mdl_nm=None/gnrtn_nm_bms=10/gnrtn_tp_k=10/lrnng_rt=0.0001/mx_stps=20000/mdl_nm_or_pth=t5-base/prf_smplng=stepwise/smpl_ngtv_prf=True/sd=0/sht=FT.step-20000/wrmp_stps=1000/checkpoint-19500')
    # # -- D8 --
    # checkpoint_top_dir = Path('./outputs/01.train.py/20230711.finalize/dtst_nm=20230711.finalize.D8/dtst_1_nm=None/bs_cnfg_nm=FLNLcorpus.20220827.base/chckpnt_nm=t5-base/ckpt_lrt=None/ckpt_mdl_nm=None/gnrtn_nm_bms=10/gnrtn_tp_k=10/lrnng_rt=0.0001/mx_stps=20000/mdl_nm_or_pth=t5-base/prf_smplng=stepwise/smpl_ngtv_prf=True/sd=0/sht=FT.step-20000/wrmp_stps=1000/checkpoint-19500/')

    # checkpoint_top_dir = Path('./outputs/01.train.py/20230718.case_study/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide/dtst_1_nm=None/bs_cnfg_nm=FLNLcorpus.20220827.base/chckpnt_nm=t5-base/ckpt_lrt=None/ckpt_mdl_nm=None/gnrtn_nm_bms=10/gnrtn_tp_k=10/lrnng_rt=0.0001/mx_stps=20000/mdl_nm_or_pth=t5-base/prf_smplng=stepwise/smpl_ngtv_prf=True/sd=0/sht=FT.step-20000/wrmp_stps=1000/checkpoint-19500')

    # ---------------------------------- 20230718.case_study ------------------------------------
    # checkpoint_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide/')
    # checkpoint_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10/')
    # checkpoint_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10.adj_verb_noun_equal/')

    # ---------------------------------- 2023-07-27.compare_models.large_steps ------------------------------------
    # checkpoint_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models.large_steps/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide')
    # checkpoint_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models.large_steps/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10.adj_verb_noun_equal')
    # checkpoint_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models.large_steps/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10.adj_verb_noun_equal')

    # checkpoint_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.fix.rerun/dtst_nm=20230729.case_study_finalize.D8')

    # checkpoint_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.steps-20000/dtst_nm=20230729.case_study_finalize.D8')

    checkpoint_top_dir = Path('./outputs/01.train.py/20230807.all_at_once/dtst_nm=20230729.case_study_finalize.D8')

    interactive_mode = 'gradio'
    # interactive_mode = 'console'
    gradio_port = 9200

    engine = SubprocessEngine()   # debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    n_gpus = 1  # debug
    # n_gpus = 4

    # do_torchrun = False  # for debug
    do_torchrun = True

    dry_run = False

    # ------------------------ fixed ------------------------
    hours = 72
    dataset_uname = '20230711.finalize.D3'

    checkpoint_configs = [path for path in checkpoint_top_dir.glob('**/*/tokenizer_config.json')
                          if str(path).find('checkpoint-') < 0]  # this finds the final checkpoint output to the top dir
    if len(checkpoint_configs) == 0:
        checkpoint_configs = [path for path in checkpoint_top_dir.glob('**/*/tokenizer_config.json')]

    if len(checkpoint_configs) == 0:
        raise ValueError(f'No checkpoint found under "{str(checkpoint_top_dir)}"')
    elif len(checkpoint_configs) >= 2:
        raise ValueError(f'multiple checkpoint  found under "{str(checkpoint_top_dir)}"')
    checkpoint_dir = checkpoint_configs[0].parent

    setting = {}
    setting.update({
        'do_train': False,
        'do_eval': False,
        'do_predict': False,
        'interactive_mode': interactive_mode,
        'gradio_port': gradio_port,
    })
    base_config_name = get_default_config_name(dataset_uname)
    base_setting = get_config(base_config_name)
    setting.update(base_setting)

    dataset_setting = get_dataset_setting(dataset_uname)
    setting.update(dataset_setting)

    model_name = json.load(open(str(checkpoint_dir / 'config.json')))['_name_or_path']
    batch_setting = get_batch_setting(model_name, n_gpus)
    setting.update(batch_setting)

    setting.update({
        'seed': 0,

        'dataset_uname': dataset_uname,

        'base_config_name': base_config_name,

        'model_name_or_path': str(checkpoint_dir),
        'evaluation_strategy': None,  # should specify None, otherwise --do_eval is forced to be True

        'dataloader_num_workers': 0,

        'generation_num_return_sequences': 10,

        'log_examples': True,
    })

    output_dir = make_output_dir(setting, output_top_dir,
                                 dirname_ignore_params=['model_name_or_path'])
    command = make_command(output_dir,
                           setting,
                           'torchrun' if do_torchrun else 'debug',
                           n_gpus=n_gpus)

    run_by_engine(
        engine,
        command,
        output_dir,
        hours=hours,
        dry_run=dry_run
    )

    logger.info('------------- ./02.interactive.py finished !! -----------')


if __name__ == '__main__':
    main()
