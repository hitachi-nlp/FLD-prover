#!/usr/bin/env python
import logging
from pathlib import Path
import json
from typing import Optional
import math


import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger

from FLD_user_shared_settings import (
    get_base_setting,
    get_dataset_setting,
    get_batch_setting,
    get_qsub_cpu_gpu_setting,
    get_generation_setting,
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

    # checkpoint = 'meta-llama/Meta-Llama-3-8B'
    # gradio_port = 9200

    # checkpoint = 'lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing__prf_intrmdt_stps=randomly_include__optmzr=rec_adam__lrnng=FT.bs-256__step-390.wrmp-200__lrnng_rt=6e-06__rc_adm_annl_schdl=immediately_from_beginning__rc_adm_fshr_cf=3000.chk-388'
    # gradio_port = 9200


    # checkpoint = 'meta-llama/Meta-Llama-3-70B'
    # gradio_port = 9200

    # checkpoint = 'mdl_nm=meta-llama@Meta-Llama-3-70B__lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1__optmzr=rec_adam__lrnng=FT.bs-256__step-390.wrmp-200.few_save__lrnng_rt=6e-06__rc_adm_fshr_cf=300.chk-388'
    # gradio_port = 9200


    # ------------------------------------ LPT --------------------------------

    # checkpoint = 'mdl_nm=EleutherAI@pythia-1b__lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing.large__dtst_nms=DarqueDante@SlimPajama-62B-Text-1of6__lgc_dtst_prb=0.0__lrnng=LPT.bs-2048__step-12000.wrmp-100__lrnng_rt=0.0003.chk-12000'
    # gradio_port = 9200

    # checkpoint = 'mdl_nm=EleutherAI@pythia-1b__lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing.large__dtst_nms=DarqueDante@SlimPajama-62B-Text-1of6__lgc_dtst_prb=0.03__lrnng=LPT.bs-2048__step-12000.wrmp-100__lrnng_rt=0.0003.chk-12000'
    # gradio_port = 9201

    # checkpoint = 'mdl_nm=meta-llama@Meta-Llama-3-8B__lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing__lrnng=FT.bs-256__step-390.wrmp-200.few_save__lrnng_rt=6e-06.chk-388'
    # gradio_port = 9200

    # checkpoint = 'mdl_nm=meta-llama@Meta-Llama-3-8B__lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing.large__dtst_nms=None__lgc_dtst_prb=1.0__lrnng=FT.bs-256__step-390.wrmp-200.few_save__lrnng_rt=3e-06__rc_adm_fshr_cf=5000__augmnttn=True.chk-390'
    # gradio_port = 9201

    checkpoint = 'meta-llama/Meta-Llama-3-70B-Instruct'
    gradio_port = 9200

    # checkpoint = 'mdl_nm=meta-llama@Meta-Llama-3-70B-Instruct__lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing__optmzr=rec_adam__lrnng=FT.bs-256__step-390.wrmp-200.few_save__lrnng_rt=3e-06__rc_adm_fshr_cf=300.chk-390'
    # gradio_port = 8500



    instruction = True
    generation_do_sample = False
    generation_temperature = 1.0
    generation_top_k = 10
    generation_repetition_penalty = 1.2  # XXX must tune for each model
    generation_max_length = 2000
    generation_max_new_tokens = 300
    generation_timeout = 60 * 5
    interactive_mode = 'gradio'
    # interactive_mode = 'console'
























    # ------------------------------------------- MEMORY REQUIREMENTS --------------------------------------------

    # script_type = 'run_prover'
    script_type = 'run_causal_prover'

    context_len = 2048

    run_mode = 'vanilla'
    # run_mode = 'torchrun'
    # run_mode = 'deepspeed'
    
    engine = SubprocessEngine()
    # engine = QsubEngine('ABCI', 'rt_G.small', n_resource=1)
    # engine = QsubEngine('ABCI', 'rt_G.large', n_resource=1)
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=1)
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=2)   # XXX only for weblab

    if isinstance(engine, SubprocessEngine):
        # n_gpus = 1  # debug
        n_gpus = 4
        # n_gpus = None  # specify this when running through QsubEngine
    elif isinstance(engine, QsubEngine):
        n_gpus, gpu_name_for_batch_size = get_qsub_gpu_setting(engine, run_mode)
        _, n_gpus, _, _ = get_qsub_cpu_gpu_setting(engine, context_len, run_mode)

    hours = 12

    dry_run = False

    # ------------------------ fixed ------------------------

    base_setting_name = 'default'

    if isinstance(checkpoint, Path):
        checkpoint_configs = [path for path in checkpoint.glob('**/*tokenizer_config.json')
                              if str(path).find('checkpoint-') < 0]  # this finds the final checkpoint output to the top dir
        if len(checkpoint_configs) == 0:
            checkpoint_configs = [path for path in checkpoint.glob('**/*tokenizer_config.json')]

        if len(checkpoint_configs) == 0:
            raise ValueError(f'No checkpoint found under "{str(checkpoint)}"')
        elif len(checkpoint_configs) >= 2:
            raise ValueError(f'multiple checkpoint  found under "{str(checkpoint)}"')

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
        lm_type = 'causal'
        proof_sampling = 'all_at_once'
        model_name_or_path = None

    setting = {}

    setting.update(get_base_setting(base_setting_name))

    setting.update(
        get_dataset_setting(
            script_type,
            instruction=instruction,
        )
    )

    setting.update(
        get_batch_setting(
            script_type,
            for_interactive=True,
            n_gpus=n_gpus,
        )
    )

    setting.update(get_model_setting(_model_name))

    setting.update(get_tokenizer_setting(_model_name))

    setting.update(
        get_generation_setting(
            script_type,
            generation_do_sample=generation_do_sample,
            generation_top_k=generation_top_k,
            generation_temperature=generation_temperature,
            generation_repetition_penalty=generation_repetition_penalty,
            generation_max_length=generation_max_length,
            generation_max_new_tokens=generation_max_new_tokens,
            generation_timeout=generation_timeout,
        ),
    )

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
        'fp16': _model_name.find('t5-') < 0 and _model_name.find('rinna/japanese-gpt2-medium') < 0,

        'proof_sampling': proof_sampling,

        'evaluation_strategy': None,  # should specify None, otherwise --do_eval is forced to be True

        'dataloader_num_workers': 0,

        'use_auth_token': True,
        'log_examples': True,
    })
    if model_name_or_path is not None:
        setting['model_name_or_path'] = str(model_name_or_path)

    output_dir = make_output_dir(setting, output_top_dir,
                                 dirname_ignore_params=['model_name_or_path'])
    command = make_command(output_dir,
                           setting,
                           run_mode,
                           engine.region,
                           script_type=script_type,
                           n_gpus_per_node=n_gpus)

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
