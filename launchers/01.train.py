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
    get_checkpoints,
    get_dataset_setting,
    get_batch_setting,
    get_logging_step_setting,
    make_output_dir,
    make_command,
    run_by_engine,
    CheckpointSpec,
    ICML_2023_NL_TRANSFER_MAJOR_DATASETS,
    ICML_2023_NL_TRANSFER_MAJOR_DATASETS_LARGE_DEPTH,
    SHOT_SETTINGS,
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

        # 'hf.hitachi-nlp/FLD.3',
        'hf.hitachi-nlp/FLD.4',
    ]

    DATASETS_DIRS = [
        # './NLProofS/outputs.FLD/10.create_FLD_corpus/20221203.first_exp',
        # './NLProofS/outputs.FLD/10.create_FLD_corpus/20221217.back_to_the_past',
        # './NLProofS/outputs/00.create_cc100_corpus.py/',

        # './outputs/00.fix_FLD_schema.py/2023-05-15/',
        # './outputs/00.fix_FLD_schema.py/20230626.many_bugs_fixed',
        # './outputs/00.fix_FLD_schema.py/20230628.make_harder',
        # './outputs/00.fix_FLD_schema.py/20230701.finalize',

        # './outputs/00.fix_FLD_schema.py/20230707.finalize',
        # './outputs/00.fix_FLD_schema.py/20230711.refactor_distractors',
        # './outputs/00.fix_FLD_schema.py/20230711.finalize',

        # './outputs/00.fix_FLD_schema.py/20230718.case_study',
        # './outputs/00.fix_FLD_schema.py/2023-07-27.compare_models',

        # './outputs/00.fix_FLD_schema.py/20230729.case_study_finalize',
        # './outputs/00.fix_FLD_schema.py/20230801.case_study_finalize.fix',
        './outputs/00.fix_FLD_schema.py/20230801.case_study_finalize.fix.test_large',
    ]

    use_test_as_train = True  # debug
    # use_test_as_train = False

    shot = 'debug.tiny'  # debug
    # shot = 'FS.shot-0'
    # shot = 'FS.shot-10'
    # shot = 'FS.shot-100'
    # shot = 'FT.step-5000'
    # shot = 'FT.step-8100'
    # shot = 'FT.step-20000'   # 20k steps are not enough wrt the qualitative analysis
    # shot = 'FT.step-50000'
    # shot = 'FT.step-100000'

    proof_sampling = 'stepwise'
    # proof_sampling = 'all_at_once'

    # max_steps = 100
    max_steps = None

    # eval_steps = 100
    eval_steps = None

    # max_train_samples = 15000
    max_train_samples = None

    # max_eval_samples = 500  # for short evaluation
    max_eval_samples = None

    engine = SubprocessEngine()   # debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    n_gpus = 1  # debug
    # n_gpus = 4

    do_torchrun = False  # for debug
    # do_torchrun = True

    # XXX: BE CAREFUL NOT TO OVERWRITE THE OFFICIAL DATASETS
    # dataset_push_to_hub_repo_name = 'hitachi-nlp/FLD.3'
    # dataset_push_to_hub_repo_name = 'hitachi-nlp/FLD.4'
    dataset_push_to_hub_repo_name = None

    dry_run = False

    # ------------------------ fixed ------------------------
    hours = 72

    lrates = [
        1e-4,
        # 5e-5,
    ]

    seeds = [
        0,
    ]

    CHECKPOINTS_DIRS = [
        # './outputs/10.train.py/20221203.first_exp.large_models.seed--7.small_lrate',
    ]

    # checkpoint_names = [None]
    # checkpoint_names = ALL_CHECKPOINT_NAMES
    # checkpoint_names = ICML_2023_NL_TRANSFER_MAJOR_DATASETS
    # checkpoint_names = ICML_2023_NL_TRANSFER_MAJOR_DATASETS_LARGE_DEPTH
    checkpoint_names = [
        't5-base',
        # 't5-large'
    ]

    do_predict = False
    use_test_as_val = True

    sample_negative_proof_args = [
        # False,
        True
    ]

    # local_dataset_1_name = None
    # local_dataset_1_name = 'ruletaker.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled'
    # local_dataset_1_name = 'ruletaker.ours.20221202'
    # local_dataset_1_name = 'cc100.20221103.small'

    # only_warn_if_checkpoint_is_not_found = False
    only_warn_if_checkpoint_is_not_found = True

    for dataset_uname in dataset_unames:

        for sample_negative_proof in sample_negative_proof_args:

            for seed in seeds:
                for checkpoint_name in checkpoint_names:

                    checkpoint_spec = CheckpointSpec(
                        name_or_local_dataset_name=checkpoint_name,

                        # checkpoint_model_name_or_path='t5-large',
                        checkpoint_model_name_or_path=None,

                        # checkpoint_lrate = 1e-4
                        checkpoint_lrate=None,

                        # add_final_reference_to_proofs=None,
                    )

                    found_checkpoint_infos = get_checkpoints(
                        checkpoint_spec,
                        check_point_dirs=CHECKPOINTS_DIRS,
                    )
                    if len(found_checkpoint_infos) == 0:
                        if only_warn_if_checkpoint_is_not_found:
                            logger.warning(f'No checkpoints found under {str(CHECKPOINTS_DIRS)}')
                            continue
                        else:
                            raise ValueError(f'No checkpoints found under {str(CHECKPOINTS_DIRS)}')

                    for checkpoint_path, found_checkpoint_spec in found_checkpoint_infos:

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

                            shot_setting = SHOT_SETTINGS[shot].copy()
                            setting.update(shot_setting)

                            batch_setting = get_batch_setting(
                                checkpoint_path + '.all_at_once' if proof_sampling == 'all_at_once' else checkpoint_path,
                                n_gpus,
                            )
                            setting.update(batch_setting)

                            setting['max_train_samples'] = max_train_samples or setting['max_train_samples']
                            setting['max_eval_samples'] = max_eval_samples or setting['max_eval_samples']

                            setting.update(get_logging_step_setting(max_steps=max_steps,
                                                                    eval_steps=eval_steps))

                            setting.update({
                                'seed': seed,

                                'dataset_uname': dataset_uname,
                                'dataset_push_to_hub_repo_name': dataset_push_to_hub_repo_name,

                                'base_config_name': base_config_name,

                                'checkpoint_name': checkpoint_name,
                                'checkpoint_path': checkpoint_path,
                                'model_name_or_path': checkpoint_path,

                                'save_total_limit': 1,

                                # 'trainer_ckpt_for_resume_training': None,  # Specify if you want to resume training
                                'proof_sampling': proof_sampling,
                                'shot': shot,
                                'sample_negative_proof': sample_negative_proof,

                                'learning_rate': _lrate,

                                # 'n_gpu': 1,
                                'dataloader_num_workers': 0,

                                'log_examples': True,
                            })

                            setting.update({
                                f'ckpt_{key}': val
                                for key, val in found_checkpoint_spec.dict().items()
                                if key != 'name_or_local_dataset_name'
                            })

                            output_dir = make_output_dir(setting, output_top_dir)
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

    logger.info('------------- ./01.train.py finished !! -----------')


if __name__ == '__main__':
    main()
