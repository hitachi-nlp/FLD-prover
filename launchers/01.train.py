#!/usr/bin/env python
import logging
from pathlib import Path
import json
from typing import Optional, Set
import math

import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger

from experimental_setting import (
    get_config,
    get_default_config_name,
    get_checkpoints,
    get_dataset_setting,
    get_batch_setting,
    get_save_eval_step_setting,
    get_model_name_settings,
    get_learning_setting,
    make_output_dir,
    make_command,
    run_by_engine,
    CheckpointSpec,
    LEARNING_SETTINGS,
    make_val_interval_setting,
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # output_top_dir = Path('./outputs/01.train.py/20230729.case_study_finalize')
    # output_top_dir = Path('./outputs/01.train.py/20230801.case_study_finalize.fix')
    # output_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.fix.rerun')

    # output_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.steps-20000')

    # output_top_dir = Path('./outputs/01.train.py/20230807.all_at_once')

    # output_top_dir = Path('./outputs/01.train.py/20230826.jpn')
    # output_top_dir = Path('./outputs/01.train.py/20230901.random_transitive_verbs')

    # output_top_dir = Path('./outputs/01.train.py/20230901.find_batch_size')
    # output_top_dir = Path('./outputs/01.train.py/20230901.overfit')

    # output_top_dir = Path('./outputs/01.train.py/20230903.find_batch_size')
    # output_top_dir = Path('./outputs/01.train.py/20230903.overfit')
    # output_top_dir = Path('./outputs/01.train.py/20230903.LLM_FS')

    # output_top_dir = Path('./outputs/01.train.py/20230904.LLM_FS')
    # output_top_dir = Path('./outputs/01.train.py/20230905.LLM_FS')
    # output_top_dir = Path('./outputs/01.train.py/20230905.LLM_FS.max_steps_upper=1000')
    # output_top_dir = Path('./outputs/01.train.py/20230910.find_bugs')
    # output_top_dir = Path('./outputs/01.train.py/20230910.rigid_comparison')

    # output_top_dir = Path('./outputs/01.train.py/20230910.preliminary')
    # output_top_dir = Path('./outputs/01.train.py/20230911.FT.gpt')

    output_top_dir = Path('./outputs/01.train.py/20230912.FT.gpt')

    # output_top_dir = Path('./outputs/01.train.py/debug')

    DATASETS_DIRS = [
        # './outputs.FLD/00.create_corpus/20230729.case_study_finalize',
        './outputs.FLD/00.create_corpus/20230801.case_study_finalize.fix',
        './outputs.FLD/00.create_corpus/20230826.jpn',
        './outputs.FLD/00.create_corpus/20230901.random_transitive_verbs',
        './outputs.FLD/00.create_corpus/20230904.jpn',
    ]

    dataset_unames = [

        # ---------------------------------- 20230729.case_study_finalize ------------------------------------
        # '20230729.case_study_finalize.D3',
        # '20230729.case_study_finalize.D8',

        # 'hf.hitachi-nlp/FLD.v2',
        # 'hf.hitachi-nlp/FLD-star.v2',

        # ---------------------------------- 20230826.jpn ------------------------------------
        # '20230826.jpn.D3',
        # '20230826.jpn.D8',

        # ---------------------------------- 202320230901.random_transitive_verbs.D3 ------------------------------------
        # '20230901.random_transitive_verbs.D3',
        # '20230901.random_transitive_verbs.D8',

        # ---------------------------------- 20230904.jpn ------------------------------------
        # '20230904.jpn.D1.wo_brnch.wo_dstrct',
        # '20230904.jpn.D1.wo_brnch',
        # '20230904.jpn.D1',
        '20230904.jpn.D3',
    ]

    model_settings = [
        # ============================ english      ============================
        # ('t5-base', 'seq2seq', 't5-base'),

        # ============================ multilingual ============================
        # ('google/mt5-base', 'seq2seq', 'google/mt5-base'),
        # ('google/mt5-large', 'seq2seq', 'google/mt5-large'),

        # TODO: other models such as mBART

        # # # ============================ japanese     ============================

        # # # -------------- < 1B params --------------

        # ('retrieva-jp/t5-small-long', 'seq2seq', 'retrieva-jp/t5-base-long'),
        # ('retrieva-jp/t5-base-long', 'seq2seq', 'retrieva-jp/t5-base-long'),
        # ('retrieva-jp/t5-large-long', 'seq2seq', 'retrieva-jp/t5-large-long'),
        # ('megagonlabs/t5-base-japanese-web', 'seq2seq', 'retrieva-jp/t5-base-long'),

        # ('cyberagent/open-calm-small', 'causal', 'cyberagent/open-calm-small'),
        ('cyberagent/open-calm-medium', 'causal', 'cyberagent/open-calm-medium'),
        # ('cyberagent/open-calm-large', 'causal', 'cyberagent/open-calm-large'),

        ('rinna/japanese-gpt-neox-small', 'causal', 'cyberagent/open-calm-small'),

        # # # # # -------------- > 1B params --------------

        # ('elyza/ELYZA-japanese-Llama-2-7b-fast', 'causal', 'matsuo-lab/weblab-10b'),
        # ('elyza/ELYZA-japanese-Llama-2-7b-fast-instruct', 'causal', 'matsuo-lab/weblab-10b'),

        # ('retrieva-jp/t5-xl', 'seq2seq', 'retrieva-jp/t5-xl'),

        # ('cyberagent/open-calm-1b', 'causal', 'cyberagent/open-calm-1b'),
        # ('cyberagent/open-calm-3b', 'causal', 'cyberagent/open-calm-3b'),
        # ('cyberagent/open-calm-7b', 'causal', 'cyberagent/open-calm-7b'),

        # ('line-corporation/japanese-large-lm-1.7b', 'causal', 'cyberagent/open-calm-1b'),
        # ('line-corporation/japanese-large-lm-1.7b-instruction-sft', 'causal', 'cyberagent/open-calm-1b'),
        # ('line-corporation/japanese-large-lm-3.6b', 'causal', 'cyberagent/open-calm-3b'),
        # ('line-corporation/japanese-large-lm-3.6b-instruction-sft', 'causal', 'cyberagent/open-calm-3b'),

        # ('rinna/japanese-gpt-neox-3.6b', 'causal', 'cyberagent/open-calm-3b'),
        # ('rinna/japanese-gpt-neox-3.6b-instruction-sft-v2', 'causal', 'cyberagent/open-calm-3b'),
        # ('rinna/japanese-gpt-neox-3.6b-instruction-ppo', 'causal', 'cyberagent/open-calm-3b'),

        # ('matsuo-lab/weblab-10b', 'causal', 'matsuo-lab/weblab-10b'),
        # ('matsuo-lab/weblab-10b-instruction-sft', 'causal', 'matsuo-lab/weblab-10b'),

        # ('stabilityai/japanese-stablelm-base-alpha-7b', 'causal', 'matsuo-lab/weblab-10b'),

    ]

    learnings = [
        # 'debug.ZS',
        # 'debug.step-10',
        # 'debug.micro',
        # 'debug.micro.deepspeed',
        # 'debug.tiny',
        # 'debug.middle',
        # 'debug.find_batch_size',

        # 'FS.shot-0',
        # 'FS.shot-10',
        # 'FS.shot-100',
        # 'FT.step-5000',
        # 'FT.step-8100',
        'FT.step-20000',
        # 'FT.step-50000',
        # 'FT.step-100000',

        # 'LLM_FS.shot-1',
        # 'LLM_FS.shot-10',
        # 'LLM_FS.shot-100',
        # 'LLM_FS.shot-1000',
        # 'LLM_FS.shot-10000',
    ]

    seeds = [
        0,
        # 1,
    ]

    epochs_list = [
        None,

        # 50,
    ]
    max_steps_upper = 300

    # engine = SubprocessEngine()
    engine = QsubEngine('ABCI', 'rt_G.large')
    # engine = QsubEngine('ABCI', 'rt_AG.small')

    # n_gpus = 1  # debug
    # n_gpus = 4
    n_gpus = None  # specify this when running through QsubEngine

    # gpu_name_for_batch_size = 'A100_48_1'
    # gpu_name_for_batch_size = 'V100_16_4'
    # gpu_name_for_batch_size = 'V100_16_4.deepspeed'
    gpu_name_for_batch_size = None   # specify this when running through QsubEngine


    # run_mode = 'vanilla'
    # run_mode = 'torchrun'
    run_mode = 'deepspeed'

    # save_model = True
    save_model = False

    # generation_timeout = 0
    generation_timeout = 60 * 5  # slow generatoin is most likely the repetitions coming from underfitting.

    dry_run = False

    # hours = 12
    hours = 72

    # ---------------------- pushing datasets to hub -------------------
    # XXX: BE CAREFUL specifying "dataset_push_to_hub_repo_name" will OVERWRITE the remote hub.
    # if you push to hub:
    # XXX: SPECIFY use_test_as_train = False, use_test_as_val = False
    # XXX: Additionally, DO NOT DELETE THE REPOSITORY MANUALLY before pushing,
    #      as it will delete all the statistics such as # downloads and likes.

    # learning = ['push_to_hub']  # XXX turn on

    dataset_push_to_hub_repo_name = None
    # dataset_push_to_hub_repo_name = 'hitachi-nlp/FLD.v2'
    # dataset_push_to_hub_repo_name = 'hitachi-nlp/FLD-star.v2'

    # ------------------------------------------------------------

    # seq2seq_proof_sampling = 'stepwise'
    seq2seq_proof_sampling = 'all_at_once'

    sample_negative_proof_args = [
        # True,
        False,
    ]

    no_subproof_for_unknown_args = [
        True,   # better
        # False,
    ]

    if isinstance(engine, QsubEngine):
        if gpu_name_for_batch_size is not None:
            raise ValueError()
        if n_gpus is not None:
            raise ValueError()

        if engine.resource == 'rt_G.small':
            n_gpus = 1
            gpu_name_for_batch_size = 'V100_16_1'
        elif engine.resource == 'rt_G.large':
            n_gpus = 4
            if run_mode == 'deepspeed':
                gpu_name_for_batch_size = 'V100_16_4.deepspeed'
            else:
                gpu_name_for_batch_size = 'V100_16_4'
        elif engine.resource == 'rt_AG.small':
            n_gpus = 1
            gpu_name_for_batch_size = 'A100_48_1'
        elif engine.resource == 'rt_AF':
            n_gpus = 8
            gpu_name_for_batch_size = 'A100_48_8'
        else:
            raise ValueError()

    lrates = [
        1e-4,   # faster convergence
        # 1e-5,
    ]

    # max_train_samples = 15000
    max_train_samples = None

    # max_eval_samples = 500  # for short evaluation
    max_eval_samples = None

    for dataset_uname in dataset_unames:

        for learning in learnings:
            for epoch in epochs_list:
                if dataset_push_to_hub_repo_name is not None:
                    if n_gpus != 1 or run_mode != 'vanilla':
                        # this does not work
                        raise ValueError()
                    if learning != 'push_to_hub':
                        raise ValueError()

                for sample_negative_proof in sample_negative_proof_args:
                    for no_subproof_for_unknown in no_subproof_for_unknown_args:
                        for seed in seeds:
                            for model_name, lm_type, model_name_for_batch_size in model_settings:
                                if lm_type == 'causal':
                                    proof_sampling = 'all_at_once'
                                else:
                                    proof_sampling = seq2seq_proof_sampling

                                for _lrate in lrates:
                                    setting = {}

                                    base_config_name = get_default_config_name(dataset_uname)
                                    base_setting = get_config(base_config_name)
                                    setting.update(base_setting)

                                    learning_setting = get_learning_setting(learning,
                                                                            epoch=epoch,
                                                                            max_steps_upper=max_steps_upper)
                                    setting.update(learning_setting)

                                    dataset_setting = get_dataset_setting(
                                        dataset_uname,
                                        DATASETS_DIRS,
                                        use_test_as_val=setting.get('use_test_as_val', False),
                                        use_test_as_train=setting.get('use_test_as_train', False))
                                    setting.update(dataset_setting)

                                    setting.update({
                                        'do_train': True,
                                        'do_eval': False,
                                        'do_predict': False,
                                    })

                                    setting['max_train_samples'] = max_train_samples or setting['max_train_samples']
                                    setting['max_eval_samples'] = max_eval_samples or setting['max_eval_samples']

                                    setting.update(get_save_eval_step_setting(
                                        max_steps=setting['max_steps'],
                                        eval_steps=setting['eval_steps'],
                                        do_save_model=save_model,
                                    ))

                                    modelwise_setting = get_batch_setting(
                                        gpu_name_for_batch_size,
                                        model_name_for_batch_size + '.all_at_once' if proof_sampling == 'all_at_once' else model_name_for_batch_size,
                                    )

                                    accum_steps = int(learning_setting['train_effective_batch_size']
                                                      / (modelwise_setting['per_device_train_batch_size'] * n_gpus))
                                    if accum_steps < 1:
                                        _per_device_train_batch_size = int(learning_setting['train_effective_batch_size'] / n_gpus)
                                        logger.warning(
                                            'change per_device_train_batch_size from %d to %d so that the train_effective_batch_size becomes %d',
                                            modelwise_setting['per_device_train_batch_size'],
                                            _per_device_train_batch_size,
                                            learning_setting['train_effective_batch_size'],
                                        )
                                        modelwise_setting['per_device_train_batch_size'] = _per_device_train_batch_size
                                        accum_steps = 1
                                    setting['gradient_accumulation_steps'] = accum_steps

                                    setting.update(modelwise_setting)

                                    setting.update(get_model_name_settings(model_name))

                                    setting.update({
                                        'seed': seed,

                                        'dataset_uname': dataset_uname,
                                        'dataset_push_to_hub_repo_name': dataset_push_to_hub_repo_name,

                                        'base_config_name': base_config_name,

                                        'lm_type': lm_type,
                                        'fp16': model_name.find('t5-') < 0 and model_name.find('rinna/japanese-gpt2-medium') < 0,

                                        # 'save_total_limit': save_total_limit,

                                        # 'trainer_ckpt_for_resume_training': None,  # Specify if you want to resume training
                                        'proof_sampling': proof_sampling,
                                        'learning': learning,
                                        'sample_negative_proof': sample_negative_proof,
                                        'no_subproof_for_unknown': no_subproof_for_unknown,

                                        'learning_rate': _lrate,

                                        # 'n_gpu': 1,
                                        'dataloader_num_workers': 0,

                                        'generation_timeout': generation_timeout,

                                        'use_auth_token': True,

                                        'log_examples': True,
                                    })

                                    output_dir = make_output_dir(setting, output_top_dir)
                                    command = make_command(output_dir,
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

    logger.info('------------- ./01.train.py finished !! -----------')


if __name__ == '__main__':
    main()
