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
    get_base_setting,
    get_dataset_setting,
    get_batch_setting,
    get_qsub_gpu_setting,
    get_other_setting,
    get_model_setting,
    get_tokenizer_setting,
    make_output_dir,
    make_command,
    run_by_engine,
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    output_top_dir = Path('./outputs/02.interactive.py')

    # ---------------------------------- 2023-07-27.compare_models.large_steps ------------------------------------
    # checkpoint_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models.large_steps/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide')
    # checkpoint_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models.large_steps/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10.adj_verb_noun_equal')
    # checkpoint_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models.large_steps/dtst_nm=20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10.adj_verb_noun_equal')

    # checkpoint_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.fix.rerun/dtst_nm=20230729.case_study_finalize.D8')

    # checkpoint_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.steps-20000/dtst_nm=20230729.case_study_finalize.D8')

    # checkpoint_top_dir = Path('./outputs/01.train.py/20230807.all_at_once/dtst_nm=20230729.case_study_finalize.D8')
    checkpoint_top_dir = Path('./outputs/01.train.py/20231010.run_causal_prover.large_models.save_models/dtst_nm=20230729.case_study_finalize.D3/bs_cnfg_nm=default/chckpnt_nm=None/FLD_dtst_prb=1.0/FLD_prf_evl_gnrtn_nm_bms=1/blck_sz=2000/dtst_nm=wikitext/gnrtn_nm_bms=1/gnrtn_tp_k=10/instrctn=True')

    # script_type = 'run_prover'
    script_type = 'run_causal_prover'

    instruction = True

    interactive_mode = 'gradio'
    # interactive_mode = 'console'
    gradio_port = 9200

    run_mode = 'vanilla'
    # run_mode = 'torchrun'
    # run_mode = 'deepspeed'   # XXX not implemented. See FLD_prover/interactive.py

    engine = SubprocessEngine()
    # engine = QsubEngine('ABCI', 'rt_G.small', n_resource=1)
    # engine = QsubEngine('ABCI', 'rt_G.large', n_resource=1)
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=2)   # XXX only for weblab

    if not isinstance(engine, QsubEngine):
        # n_gpus = 1  # debug
        n_gpus = 4
        # n_gpus = None  # specify this when running through QsubEngine

    hours = 12

    dry_run = False

    # ------------------------ fixed ------------------------
    if isinstance(engine, QsubEngine):
        n_gpus, gpu_name_for_batch_size = get_qsub_gpu_setting(engine, run_mode)

    base_setting_name = 'default'

    # slow generatoin is most likely the repetitions coming from underfitting, so we discard such generations.
    generation_timeout = 60 * 10  # For LLMs
    # generation_timeout = 6

    checkpoint_configs = [path for path in checkpoint_top_dir.glob('**/*/tokenizer_config.json')
                          if str(path).find('checkpoint-') < 0]  # this finds the final checkpoint output to the top dir
    if len(checkpoint_configs) == 0:
        checkpoint_configs = [path for path in checkpoint_top_dir.glob('**/*/tokenizer_config.json')]

    if len(checkpoint_configs) == 0:
        raise ValueError(f'No checkpoint found under "{str(checkpoint_top_dir)}"')
    elif len(checkpoint_configs) >= 2:
        raise ValueError(f'multiple checkpoint  found under "{str(checkpoint_top_dir)}"')
    checkpoint_dir = checkpoint_configs[0].parent
    lab_setting = json.load(open(str(checkpoint_dir / 'lab.params.json')))

    lm_type = lab_setting['lm_type']

    setting = {}

    setting.update(get_base_setting(base_setting_name))

    setting.update(
        get_dataset_setting(
            script_type,
            instruction=instruction,
        )
    )

    model_name = json.load(open(str(checkpoint_dir / 'config.json')))['_name_or_path']
    proof_sampling = lab_setting['proof_sampling']
    setting.update(
        get_batch_setting(
            script_type,
            for_interactive=True,
        )
    )

    setting.update(get_model_setting(model_name))

    setting.update(get_tokenizer_setting(model_name))

    setting.update(get_other_setting(script_type, generation_timeout))

    setting.update({
        'do_train': False,
        'do_eval': False,
        'do_predict': False,
        'interactive_mode': interactive_mode,
        'gradio_port': gradio_port,
    })

    setting.update({
        'seed': 0,

        'base_setting_name': base_setting_name,

        'lm_type': lm_type,
        'fp16': model_name.find('t5-') < 0 and model_name.find('rinna/japanese-gpt2-medium') < 0,

        'proof_sampling': proof_sampling,

        'model_name_or_path': str(checkpoint_dir),
        'evaluation_strategy': None,  # should specify None, otherwise --do_eval is forced to be True

        'dataloader_num_workers': 0,

        'use_auth_token': True,
        'log_examples': True,
    })

    output_dir = make_output_dir(setting, output_top_dir,
                                 dirname_ignore_params=['model_name_or_path'])
    command = make_command(script_type,
                           output_dir,
                           setting,
                           run_mode,
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
