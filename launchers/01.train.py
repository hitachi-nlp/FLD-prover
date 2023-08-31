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
    get_logging_step_setting,
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
    # output_top_dir = Path('./outputs/01.train.py/2023-05-15')
    # output_top_dir = Path('./outputs/01.train.py/debug/2023-05-13.no_torchrun')
    # output_top_dir = Path('./outputs/01.train.py/debug/2023-05-13.torchrun.large_steps')

    # output_top_dir = Path('./outputs/01.train.py/2023-05-16.sFLD-impl')
    # output_top_dir = Path('./outputs/01.train.py/2023-05-16.sFLD-impl.batch_size-64')
    # output_top_dir = Path('./outputs/01.train.py/2023-05-16.sFLD-impl.batch_size-64.no_negative_proof')
    # output_top_dir = Path('./outputs/01.train.py/2023-05-16.sFLD-impl.batch_size-64.lrate-5e-5')

    # output_top_dir = Path('./outputs/01.train.py/2023-05-16.sFLD-impl')
    # output_top_dir = Path('./outputs/01.train.py/2023-05-16.FLD-impl')

    # output_top_dir = Path('./outputs/01.train.py/2023-05-17.sFLD-impl.large_steps')
    # output_top_dir = Path('./outputs/01.train.py/2023-05-17.FLD-impl.large_steps')
    # output_top_dir = Path('./outputs/01.train.py/FLD.2.large_steps')

    # output_top_dir = Path('./outputs/01.train.py/20230626.many_bugs_fixed')
    # output_top_dir = Path('./outputs/01.train.py/20230628.make_harder')
    # output_top_dir = Path('./outputs/01.train.py/20230628.make_harder.scoring_disallow_any_proof_for_unknown')
    # output_top_dir = Path('./outputs/01.train.py/20230701.finalize')
    # output_top_dir = Path('./outputs/01.train.py/debug')

    # output_top_dir = Path('./outputs/01.train.py/20230707.finalize')
    # output_top_dir = Path('./outputs/01.train.py/20230707.finalize.max_train_samples=15000')
    # output_top_dir = Path('./outputs/01.train.py//20230711.refactor_distractors')

    # output_top_dir = Path('./outputs/01.train.py/20230711.finalize')

    # output_top_dir = Path('./outputs/01.train.py/20230718.case_study')
    # output_top_dir = Path('./outputs/01.train.py/20230718.case_study')
    # output_top_dir = Path('./outputs/01.train.py/20230718.case_study')

    # output_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models')
    # output_top_dir = Path('./outputs/01.train.py/2023-07-27.compare_models.large_steps')

    # output_top_dir = Path('./outputs/01.train.py/20230729.case_study_finalize')
    # output_top_dir = Path('./outputs/01.train.py/20230801.case_study_finalize.fix')
    # output_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.fix.rerun')

    # output_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.steps-20000')

    # output_top_dir = Path('./outputs/01.train.py/20230807.all_at_once')

    # output_top_dir = Path('./outputs/01.train.py/20230826.jpn')

    output_top_dir = Path('./outputs/01.train.py/debug')

    dataset_unames = [
        # 'FLD.debug.2023-05-13',

        # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP',   # sFLD-impl
        # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',              # FLD-impl
        # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000'                   # FLD.2

        # ---------------------------------- 20230626.many_bugs_fixed ------------------------------------
        # '20230626.many_bugs_fixed.20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',
        # '20230626.many_bugs_fixed.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000.plus_quantifiers',


        # ---------------------------------- 20230628.make_harder ------------------------------------
        # '20230626.many_bugs_fixed.D3.hard',
        # '20230626.many_bugs_fixed.D3.hard.dist-trees',
        # '20230626.many_bugs_fixed.D3.hard.unk-0.1',
        # '20230626.many_bugs_fixed.D3.hard.brnch-high',
        # '20230626.many_bugs_fixed.D3.hard.dist-neg-1.0',
        # '20230626.many_bugs_fixed.D3.hard.dist-neg-0.5',
        # '20230626.many_bugs_fixed.D3.hard.dist-neg-0.0',
        # '20230626.many_bugs_fixed.D3.hard.dist-trees-only',

        # '20230626.many_bugs_fixed.D8.hard',
        # '20230626.many_bugs_fixed.D8.hard.dist-trees',

        # ---------------------------------- 20230701.finalize ------------------------------------
        # '20230701.D3.default',
        # '20230701.D3.wo_transl_dist',
        # '20230701.D3.brnch-small',
        # '20230701.D3.dist-small',
        # '20230701.D8.default',

        # ---------------------------------- 20230707.finalize ------------------------------------
        # '20230707.finalize.D3.dist-double',
        # '20230707.finalize.D3.dist-triple',
        # '20230707.finalize.D3.dist-quadruple',

        # '20230707.finalize.D8.dist-double',
        # '20230707.finalize.D8.dist-triple',
        # '20230707.finalize.D8.dist-quadruple',

        # ---------------------------------- 20230711 ------------------------------------
        # '20230711.dist-fallback',
        # '20230711.finalize.D3',
        # '20230711.finalize.D8',

        # ---------------------------------- 20230718.case_study ------------------------------------
        # '20230718.case_study.D3.dist-mixture',
        # '20230718.case_study.D3.num_dist-wide',
        # '20230718.case_study.D8.dist-mixture.num_dist-wide',
        
        # '20230718.case_study.D3.dist-mixture.num_dist-wide',
        # '20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_logE',
        # '20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10',
        # '20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10.adj_verb_noun_equal',


        # ---------------------------------- 20230729.case_study_finalize ------------------------------------
        # '20230729.case_study_finalize.D3',
        # '20230729.case_study_finalize.D8',

        # 'hf.hitachi-nlp/FLD.v2',
        # 'hf.hitachi-nlp/FLD-star.v2',

        # ---------------------------------- 20230826.jpn ------------------------------------
        '20230826.jpn.D3',
        # '20230826.jpn.D8',
    ]

    DATASETS_DIRS = [
        # './outputs.FLD/10.create_FLD_corpus/20221203.first_exp',
        # './outputs.FLD/10.create_FLD_corpus/20221217.back_to_the_past',
        # './NLProofS/outputs/00.create_cc100_corpus.py/',

        # './outputs.FLD/10.create_FLD_corpus/20230626.many_bugs_fixed',
        # './outputs.FLD/10.create_FLD_corpus/20230628.make_harder',
        # './outputs.FLD/10.create_FLD_corpus/20230701.finalize',

        # './outputs.FLD/10.create_FLD_corpus/20230707.finalize',

        # './outputs.FLD/00.create_corpus/20230710.update_translation',
        # './outputs.FLD/00.create_corpus/20230710.update_translation.bf51eb2',
        # './outputs.FLD/00.create_corpus/20230710.update_translation.7485fef',

        # './outputs.FLD/00.create_corpus/20230711.refactor_distractors',
        # './outputs.FLD/00.create_corpus/20230711.finalize',

        # './outputs.FLD/00.create_corpus/20230718.case_study',
        # './outputs.FLD/00.create_corpus/2023-07-27.compare_models',

        # './outputs.FLD/00.create_corpus/20230729.case_study_finalize',
        './outputs.FLD/00.create_corpus/20230801.case_study_finalize.fix',
        './outputs.FLD/00.create_corpus/20230826.jpn',
    ]

    model_settings = [
        # ---------------------------- English models ----------------------------
        # ('t5-base', 'seq2seq', 't5-base'),
        # ('t5-large', 'seq2seq', 't5-base'),

        # ---------------------------- Japanese models ----------------------------
        # ('google/mt5-base', 'causal', 't5-base'),

        # ('retrieva-jp/t5-base-long', 'seq2seq', 't5-base'),
        # ('retrieva-jp/t5-large-long', 'seq2seq', 't5-base'),
        # ('retrieva-jp/t5-xl', 'seq2seq', 't5-base'),

        # ('cyberagent/open-calm-small', 'causal', 'cyberagent/open-calm-small'),
        # ('cyberagent/open-calm-medium', 'causal', 'cyberagent/open-calm-medium'),
        # ('cyberagent/open-calm-large', 'causal', 'cyberagent/open-calm-large'),
        # ('cyberagent/open-calm-3b', 'causal', 'cyberagent/open-calm-3b'),
        # ('cyberagent/open-calm-7b', 'causal', 'cyberagent/open-calm-7b'),

        # ('izumi-lab/stormy-7b-10ep', 'causal', 't5-base'),

        ('abeja/gpt2-large-japanese', 'causal', 'abeja/gpt2-large-japanese'),

        # ('matsuo-lab/weblab-10b', 'causal', 't5-base'),
        # ('matsuo-lab/weblab-10b-instruction-sft', 'causal', 't5-base'),

        # ('stabilityai/japanese-stablelm-base-alpha-7b', 'causal', 't5-base'),
        # ('stabilityai/japanese-stablelm-instruct-alpha-7b', 'causal', 't5-base'),

        # ('line-corporation/japanese-large-lm-1.7b', 'causal', 't5-base'),
        # ('line-corporation/japanese-large-lm-3.6b', 'causal', 't5-base'),
        # ('line-corporation/japanese-large-lm-1.7b-instruction-sft', 'causal', 't5-base'),
        # ('line-corporation/japanese-large-lm-3.6b-instruction-sft', 'causal', 't5-base'),

        # ('rinna/japanese-gpt2-medium', 'causal', 'cyberagent/open-calm-small'),
        # ('rinna/japanese-gpt-1b', 'causal', 't5-base'),
        # ('rinna/japanese-gpt-neox-3.6b', 'causal', 't5-base'),
        # ('rinna/japanese-gpt-neox-3.6b-instruction-sft-v2', 'causal', 't5-base'),
        # ('rinna/japanese-gpt-neox-3.6b-instruction-ppo', 'causal', 't5-base'),


        # ---------------------------- rejected models ----------------------------
        # tokenizer have too many unknowns for alphabet, e.g., "U" and "l"
        # [rejected] ('sonoisa/t5-base-japanese', 'seq2seq', 't5-base'),
        # [rejected] ('sonoisa/t5-base-japanese-v1.1', 'seq2seq', 't5-base'),
    ]

    # learning = 'debug.ZS'
    learning = 'debug.micro'
    # learning = 'debug.tiny'
    # learning = 'FS.shot-0'
    # learning = 'FS.shot-10'
    # learning = 'FS.shot-100'
    # learning = 'FT.step-5000'
    # learning = 'FT.step-8100'
    # learning = 'FT.step-20000'   # 20k steps are not enough wrt the qualitative analysis
    # learning = 'FT.step-50000'
    # learning = 'FT.step-100000'

    use_test_as_train = True   # debug
    use_test_as_val = True

    # proof_sampling = 'stepwise'
    proof_sampling = 'all_at_once'

    lora = False
    # lora = True

    engine = SubprocessEngine()   # debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    # n_gpus = 1  # debug
    n_gpus = 4

    gpu_name_for_batch_size = 'A100_48_1'

    # run_mode = 'debug'
    # run_mode = 'torchrun'
    run_mode = 'deepspeed'

    dry_run = False

    # ---------------------- pushing datasets to hub -------------------
    # XXX: BE CAREFUL specifying "dataset_push_to_hub_repo_name" will OVERWRITE the remote hub.
    # if you push to hub:
    # XXX: SPECIFY use_test_as_train = False, use_test_as_val = False
    # XXX: Additionally, DO NOT DELETE THE REPOSITORY MANUALLY before pushing,
    #      as it will delete all the statistics such as # downloads and likes.

    dataset_push_to_hub_repo_name = None
    # dataset_push_to_hub_repo_name = 'hitachi-nlp/FLD.v2'
    # dataset_push_to_hub_repo_name = 'hitachi-nlp/FLD-star.v2'

    # Specify as follows so that we push all the splits.
    # use_test_as_train = False
    # use_test_as_val = False
    # ------------------------------------------------------------

    # ------------------------ fixed ------------------------

    if isinstance(engine, QsubEngine):
        if engine.resource == 'rt_G.small':
            n_gpus = 1
            gpu_name_for_batch_size = 'V100_16_1'
        elif engine.resource == 'rt_G.large':
            n_gpus = 4
            gpu_name_for_batch_size = 'V100_16_4'
        elif engine.resource == 'rt_AG.small':
            n_gpus = 1
            gpu_name_for_batch_size = 'A100_48_1'
        elif engine.resource == 'rt_AF':
            n_gpus = 8
            gpu_name_for_batch_size = 'A100_48_8'
        else:
            raise ValueError()
   
    if dataset_push_to_hub_repo_name is not None:
        if n_gpus != 1 or run_mode != 'debug':
            # this does not work
            raise ValueError()

    hours = 72

    lrates = [
        1e-4,
        # 5e-5,
    ]

    seeds = [
        0,
    ]

    do_predict = False
    
    sample_negative_proof_args = [
        # False,
        True
    ]

    # max_steps = 100
    max_steps = None

    # eval_steps = 100
    eval_steps = None

    # max_train_samples = 15000
    max_train_samples = None

    # max_eval_samples = 500  # for short evaluation
    max_eval_samples = None

    for dataset_uname in dataset_unames:

        for sample_negative_proof in sample_negative_proof_args:

            for seed in seeds:
                for model_name, lm_type, model_name_for_batch_size in model_settings:
                    for _lrate in lrates:
                        setting = {}

                        dataset_setting = get_dataset_setting(dataset_uname,
                                                              DATASETS_DIRS,
                                                              use_test_as_val=use_test_as_val,
                                                              use_test_as_train=use_test_as_train)
                        setting.update(dataset_setting)

                        setting.update({
                            'do_train': True,
                            'do_eval': True,
                            'do_predict': do_predict,
                        })

                        base_config_name = get_default_config_name(dataset_uname)
                        base_setting = get_config(base_config_name)
                        setting.update(base_setting)

                        learning_setting = LEARNING_SETTINGS[learning].copy()
                        setting.update(learning_setting)

                        setting['max_train_samples'] = max_train_samples or setting['max_train_samples']
                        setting['max_eval_samples'] = max_eval_samples or setting['max_eval_samples']

                        setting.update(get_logging_step_setting(max_steps=max_steps,
                                                                eval_steps=eval_steps))

                        modelwise_setting = get_batch_setting(
                            gpu_name_for_batch_size,
                            model_name_for_batch_size + '.all_at_once' if proof_sampling == 'all_at_once' else model_name_for_batch_size,
                        )

                        accum_steps = int(learning_setting['train_effective_batch_size']\
                                          / (setting['per_device_train_batch_size'] * n_gpus))
                        if accum_steps < 1:
                            raise ValueError()
                        setting['gradient_accumulation_steps'] = accum_steps

                        setting.update(modelwise_setting)

                        setting.update({
                            'seed': seed,

                            'dataset_uname': dataset_uname,
                            'dataset_push_to_hub_repo_name': dataset_push_to_hub_repo_name,

                            'base_config_name': base_config_name,

                            'model_name_or_path': model_name,
                            'lm_type': lm_type,
                            'fp16': model_name.find('t5-') < 0 and model_name.find('rinna/japanese-gpt2-medium') < 0,

                            'save_total_limit': 1,

                            # 'trainer_ckpt_for_resume_training': None,  # Specify if you want to resume training
                            'proof_sampling': proof_sampling,
                            'learning': learning,
                            'sample_negative_proof': sample_negative_proof,

                            'learning_rate': _lrate,

                            'lora': lora,

                            # 'n_gpu': 1,
                            'dataloader_num_workers': 0,

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
