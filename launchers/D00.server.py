#!/usr/bin/env python
import logging
from pathlib import Path
import json
from typing import Optional, Union
import math


import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger

from FLD_user_shared_settings import (
    get_base_setting,
    get_dataset_setting,
    get_batch_setting,
    get_qsub_cpu_gpu_setting,
    get_setting,
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
    output_top_dir = Path('./outputs/D00.server.py')

    checkpoint = 'mdl_nm=mdl_nm=meta-llama@Meta-Llama-3.1-8B__lgc_dtst_unm=2024-09-03.trnsl-thing_person-v0__optmzr=rec_adam__lrnng=FT.bs-256__step-1953.wrmp-200__lrnng_rt=1e-05__rc_adm_fshr_cf=1000__augmnttn=False__prf_intrmdt_stps=randomly_include__prf_intrmdt_stps_prb=0.5__mx_grd_nrm=0.5__prmpt_indct_thrms=True.chk-1953__lgc_dtst_unm=hf.hitachi-nlp@ruletaker__optmzr=None__lrnng=FT.bs-128__step-94.wrmp-20__lrnng_rt=3e-05__rc_adm_fshr_cf=None__augmnttn=False__prf_intrmdt_stps=randomly_include__prf_intrmdt_stps_prb=0.5__mx_grd_nrm=1.0__prmpt_indct_thrms=True__dtst_prbs=1.0.chk-100'
    gradio_port = 9500




    temperature = 1.0
    top_k = 10
    top_p = 0.95
    repetition_penalty = 1.2  # XXX must tune for each model
    max_length = 2000
    max_new_tokens = 1000
    timeout = 60 * 5

    engine = SubprocessEngine('haic', 'xhn_s.middle', n_resource=1)






















    # ------------------------------------------- MEMORY REQUIREMENTS --------------------------------------------
    dry_run = False

    # ------------------------ fixed ------------------------

    model_name, model_name_or_path = find_checkpoint(checkpoint)

    setting = {
        'model_name': model_name,
        'model_name_or_path': model_name_or_path,
    }

    output_dir = make_output_dir(
        setting,
        output_top_dir / f'mdl_nm={model_name}',
        dirname_ignore_params=[
            'model_name',
            'model_name_or_path',
        ],
    )

    command = ' '.join([
        'python',
        './llm-inference/scripts/server.py',

    ])

    run_by_engine(
        engine,
        command,
        output_dir,
        hours=12,
        dry_run=dry_run
    )

    logger.info('------------- D00.server.py finished !! -----------')





























def find_checkpoint(checkpoint: Union[str, Path]) -> Path:
    if isinstance(checkpoint, Path):
        checkpoint_configs = [path for path in checkpoint.glob('**/*tokenizer_config.json')
                              if str(path).find('checkpoint-') < 0]  # this finds the final checkpoint output to the top dir
        if len(checkpoint_configs) == 0:
            checkpoint_configs = [path for path in checkpoint.glob('**/*tokenizer_config.json')]

        if len(checkpoint_configs) == 0:
            raise ValueError(f'No checkpoint found under "{str(checkpoint)}"')
        elif len(checkpoint_configs) >= 2:
            raise ValueError(f'multiple checkpoint  found under "{str(checkpoint)}')

        checkpoint_dir = checkpoint_configs[0].parent
        if (checkpoint_dir / 'lab.params.json').exists():
            lab_setting = json.load(open(str(checkpoint_dir / 'lab.params.json')))
        elif (checkpoint_dir.parent / 'lab.params.json').exists():
            lab_setting = json.load(open(str(checkpoint_dir.parent / 'lab.params.json')))
        else:
            lab_setting = {}

        _model_name = json.load(open(str(checkpoint_dir / 'config.json')))['_name_or_path']
        model_name_or_path = checkpoint_dir
    else:
        _model_name = checkpoint
        model_name_or_path = None

    return model_name_or_path, _model_name


if __name__ == '__main__':
    main()
