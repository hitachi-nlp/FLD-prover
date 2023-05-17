#!/usr/bin/env python
import logging
from pathlib import Path
import json
from typing import Optional
import math


import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger
# from stance_indication import StanceIndicationMethod

from experimental_setting import (
    get_config,
    get_default_config_name,
    get_checkpoints,
    get_dataset_paths,
    get_dataset_setting,
    get_batch_setting,
    make_output_dir,
    make_command,
    run_by_engine,
    guess_dataset_type,
    CheckpointSpec,
    ICML_2023_NL_TRANSFER_MAJOR_DATASETS,
    ICML_2023_NL_TRANSFER_MAJOR_DATASETS_LARGE_DEPTH,
    SHOT_SETTINGS,
    make_val_interval_setting,
)

logger = logging.getLogger(__name__)


ALL_DATASET_NAMES = [
    # ---------------------------------- 20221203.first_exp ------------------------------------
    '20221203.first_exp__arg-RT__frml-smpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-RT__frml-cmpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-AA__frml-cmpl__dist-20__transl-nrrw__tree-1__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000',

    # ---------------------------------- 20221215 additional experiments ------------------------------------
    '20221203.first_exp__arg-RT__frml-smpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',

    # ---------------------------------- 20221216 additional experiments ------------------------------------
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-smpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-5__dataset_size-30000',

    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',
    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-8__dataset_size-30000.G_MP',

    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT',

    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000.G_MP',

    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-wide__tree-5__dataset_size-30000.G_MP',
    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.G_MP',

    # ---------------------------------- 20221217.back_to_the_past ------------------------------------
    '20221217.back_to_the_past__arg-FLNL__frml-cmpl__dist-10__transl-wide__tree-10__dataset_size-100000',

    # ---------------------------------- baselines ------------------------------------

    'ruletaker.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',
    'ruletaker.D5.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',
    'ruletaker.NatLang.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',
    'ruletaker.birds-electricity.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',

    'ruletaker.ours.20221202',
    'ruletaker.NL.ours.20221202',
    'ruletaker.BE.ours.20221202',

    'EB-task1.shuffled',
    'EB-task2.shuffled',

]


ALL_CHECKPOINT_NAMES = [

    '20221203.first_exp__arg-RT__frml-smpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-RT__frml-cmpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-AA__frml-cmpl__dist-20__transl-nrrw__tree-1__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000',

    # ---------------------------------- 20221216 additional experiments ------------------------------------
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-0__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-smpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-5__dataset_size-30000',

    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',
    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-8__dataset_size-30000.G_MP',

    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT',

    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-wide__tree-5__dataset_size-30000.G_MP',
    '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.G_MP',

    # ---------------------------------- multitask ------------------------------------
    # XXX: missing!
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.local_dataset_1_name--ruletaker.ours.20221202',
    '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-8__dataset_size-100000.local_dataset_1_name--cc100.20221103.small',

    # ---------------------------------- 20221217.back_to_the_past ------------------------------------
    '20221217.back_to_the_past__arg-FLNL__frml-cmpl__dist-10__transl-wide__tree-10__dataset_size-100000',

    # ---------------------------------- baselines ------------------------------------
    'ruletaker.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',
    'ruletaker.D5.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',
    'ruletaker.NatLang.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled',

    # -- for multitask --
    'ruletaker.ours.20221202',
    'ruletaker.NL.ours.20221202',

    'EB-task1.shuffled',
    'EB-task2.shuffled',

    None,

]


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

    output_top_dir = Path('./outputs/01.train.py/2023-05-17.sFLD-impl.large_steps')

    local_dataset_names = [
        # 'FLD.debug.2023-05-13',

        '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP',   # sFLD-impl
        # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',              # FLD-impl
    ]

    # shot = 'debug.tiny'  # debug
    # shot = 'FS.shot-0'
    # shot = 'FS.shot-10'
    # shot = 'FS.shot-100'
    # shot = 'FT.step-5000'
    shot = 'FT.step-20000'

    # max_steps = 100
    max_steps = None

    # eval_steps = 500
    eval_steps = None

    # engine = SubprocessEngine()   # for debug
    engine = QsubEngine('ABCI', 'rt_G.large')

    n_gpus = 4

    # do_torchrun = False  # for debug
    do_torchrun = True

    lrates = [
        1e-4,
        # 5e-5,
    ]

    sample_negative_proof_args = [
        # False,
        True
    ]

    # ------------------------ fixed ------------------------
    dry_run = False
    hours = 72

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
    ]

    do_predict = False
    scoring_similarity_threshold = False
    use_test_as_train = False  # for debugging
    use_test_as_val = True
    do_transfer_on_same_dataset = True

    local_dataset_1_name = None
    # local_dataset_1_name = 'ruletaker.include_all_answers.unknown_with_collapsed_proof.reference_unknown_proof_ratio=0.3.negative_proof_prob=0.0.shuffled'
    # local_dataset_1_name = 'ruletaker.ours.20221202'
    # local_dataset_1_name = 'cc100.20221103.small'

    # only_warn_if_checkpoint_is_not_found = False
    only_warn_if_checkpoint_is_not_found = True

    DATASETS_DIRS = [
        './outputs/00.fix_FLD_schema.py/2023-05-15/'
        # './NLProofS/outputs.FLD/10.create_FLD_corpus/20221203.first_exp',
        # './NLProofS/outputs.FLD/10.create_FLD_corpus/20221217.back_to_the_past',
        # './NLProofS/outputs/00.create_cc100_corpus.py/',
    ]

    for local_dataset_name in local_dataset_names:
        dataset_paths = get_dataset_paths(local_dataset_name,
                                          DATASETS_DIRS,
                                          use_test_as_val=use_test_as_val,
                                          use_test_as_train=use_test_as_train)
        dataset_setting = get_dataset_setting(local_dataset_name)

        for split_name, path in dataset_paths.items():
            if not Path(path).exists:
                raise Exception(f'{split_name} dataset does not exist at {path}')

        if local_dataset_1_name is not None:
            _dataset_1_paths = get_dataset_paths(local_dataset_1_name,
                                                 DATASETS_DIRS,
                                                 use_test_as_val=use_test_as_val,
                                                 use_test_as_train=use_test_as_train)
            for split_name, path in _dataset_1_paths.items():
                if not Path(path).exists:
                    raise Exception(f'{split_name} dataset does not exist at {path}')

            dataset_1_type = guess_dataset_type(local_dataset_1_name)

            dataset_1_paths = {
                'dataset_1': dataset_1_type,
                'train_file_1': _dataset_1_paths['train_file']

                # currently, the different types of datasets can not be used in valid and test split, due to the limitation of the EntailmentWriter()
                # 'validation_file_1': dataset_1_paths['validation_file']
                # 'test_file_1': dataset_1_paths['test_file']
            }
        else:
            dataset_1_paths = {}

        for sample_negative_proof in sample_negative_proof_args:

            setting = SHOT_SETTINGS[shot]
            if max_steps is not None:
                setting['max_steps'] = max_steps
                if eval_steps is not None:
                    setting['eval_steps'] = eval_steps
                if setting['eval_steps'] > max_steps:
                    setting['eval_steps'] = max_steps
                
            setting.update(dataset_setting)
            setting.update({
                'do_train': 'train_file' in dataset_paths,
                'do_eval': 'validation_file' in dataset_paths,
                # 'do_predict': 'test_file' in dataset_paths,
                'do_predict': do_predict,
            })

            for seed in seeds:
                for checkpoint_name in checkpoint_names:
                    if not do_transfer_on_same_dataset and checkpoint_name == local_dataset_name:
                        logger.info('skip transfer from checkpoint_name="%s" to local_dataset_name=%s since do_transfer_on_same_dataset=False',
                                    checkpoint_name,
                                    local_dataset_name)
                        continue

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

                            base_config_name = get_default_config_name(local_dataset_name)
                            all_setting = get_config(base_config_name)

                            all_setting.update({
                                'seed': seed,

                                'local_dataset_name': local_dataset_name,
                                'local_dataset_1_name': local_dataset_1_name,
                                # 'exclude_unknown': False,

                                'base_config_name': base_config_name,
                                # 'base_config_path': base_config_path,

                                'checkpoint_name': checkpoint_name,
                                'checkpoint_path': checkpoint_path,
                                'model_name_or_path': checkpoint_path,

                                # 'stance_indication_method': StanceIndicationMethod.STANCE_MARKER_IN_PROOF.value,
                                # 'trainer_ckpt_for_resume_training': None,  # Specify if you want to resume training
                                'shot': shot,
                                'scoring_similarity_threshold': scoring_similarity_threshold,
                                'scoring_allowed_additional_proof_steps': 5,
                                'sample_negative_proof': sample_negative_proof,

                                'learning_rate': _lrate,

                                # 'n_gpu': 1,
                                'dataloader_num_workers': 0,

                                'log_examples': True,
                            })
                            all_setting.update(dataset_paths)
                            all_setting.update(dataset_1_paths)
                            all_setting.update(get_batch_setting(all_setting['model_name_or_path'], n_gpus))
                            all_setting.update(setting)
                            all_setting.update({
                                f'ckpt_{key}': val
                                for key, val in found_checkpoint_spec.dict().items()
                                if key != 'name_or_local_dataset_name'
                            })
                            all_setting['save_steps'] = all_setting['eval_steps']

                            # if all_setting.get('max_steps', None) is not None:
                            #     all_setting['num_train_epochs'] = -1

                            # if all_setting.get('num_val_stage_throught_training', None) is not None:
                            #     all_setting.update(make_val_interval_setting(all, dataset_paths['train_file']))

                            # if 'train_file' in dataset_paths:
                            #     dataset_setting_path = Path(dataset_paths['train_file']).parent / 'lab.params.json'
                            #     if dataset_setting_path.exists():
                            #         dataset_setting = json.load(open(str(dataset_setting_path)))
                            #         all_setting.update({
                            #             f'dataset_setting.{key}': val
                            #             for key, val in dataset_setting.items()
                            #         })

                            output_dir = make_output_dir(all_setting, output_top_dir)
                            command = make_command(output_dir,
                                                   all_setting,
                                                   'torchrun' if do_torchrun else 'debug',
                                                   n_gpus=n_gpus)

                            run_by_engine(
                                engine,
                                command,
                                output_dir,
                                hours=hours,
                                dry_run=dry_run
                            )

    logger.info('------------- ./10.train.py finished !! -----------')


if __name__ == '__main__':
    main()
