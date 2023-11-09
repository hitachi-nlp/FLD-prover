#!/usr/bin/env python
import logging
from pathlib import Path


import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger
from lab import build_dir

from experimental_setting import (
    get_dataset_setting,
    run_by_engine,
    maybe_option_value,
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # output_top_dir = Path('./outputs/10.make_prompts.py/20230711.refactor_distractors')
    # output_top_dir = Path('./outputs/10.make_prompts.py/2023-08-31.jpn')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230905.LLM_FS')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230919.jpn')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20231106.refaactor')

    # output_top_dir = Path('./outputs/10.make_prompts.py/20231107.preliminary')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20231107.preliminary.seed--1')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20231107.preliminary.seed--2')

    # output_top_dir = Path('./outputs/10.make_prompts.py/20231107.preliminary.many_seeds')

    # output_top_dir = Path('./outputs/10.make_prompts.py/20231109.icl_max_proof_by_contradiction_per_label')
    output_top_dir = Path('./outputs/10.make_prompts.py/20231109.3-shot')

    DATASETS_DIRS = [
        # './outputs/00.fix_FLD_schema.py/20230711.refactor_distractors',
        './outputs.FLD/00.create_corpus/20230801.case_study_finalize.fix',
        './outputs.FLD/00.create_corpus/20230826.jpn',
        './outputs.FLD/00.create_corpus/20230901.random_transitive_verbs',
        './outputs.FLD/00.create_corpus/20230904.jpn',
        './outputs.FLD/00.create_corpus/20230912.jpn',
        './outputs.FLD/00.create_corpus/20230916.jpn',
    ]

    dataset_unames = [
        # ---------------------------------- 20230729.case_study_finalize ------------------------------------
        # '20230729.case_study_finalize.D3',
        # '20230729.case_study_finalize.D8',

        'hf.hitachi-nlp/FLD.v2__default',
        'hf.hitachi-nlp/FLD.v2__star',

        # ---------------------------------- 20230826.jpn ------------------------------------
        # '20230826.jpn.D3',
        # '20230826.jpn.D8',

        # ---------------------------------- 20230904.jpn ------------------------------------
        # '20230904.jpn.D1.wo_brnch.wo_dstrct',
        # '20230904.jpn.D1.wo_brnch',
        # '20230904.jpn.D1',
        # '20230904.jpn.D3',

        # ---------------------------------- 20230916.jpn ------------------------------------
        # '20230916.jpn.D1_wo_dist',
        # '20230916.jpn.D1',
        # '20230916.jpn.D3',
        # '20230916.jpn.D5',
    ]

    n_shot_list = [
        3,
        # 10,   # for 8k context
        # 32, # for 16k context
    ]

    seeds = [
        0,
        1,
        2,
        # 3,
        # 4,
    ]

    icl_max_proof_by_contradiction_per_label_args = [
        # 0,
        # 1,
        # 2,
        None,
    ]

    prompt_types = [
        # 'ICL',
        # 'ICL-COT',
        # 'ICL-COT.v1',
        'ICL-COT.v2',
    ]

    wait_until_finish = False

    engine = SubprocessEngine()   # for debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    dry_run = False

    # -------------------------- fixed settings --------------------------

    # -------------------------- running --------------------------
    for dataset_uname in dataset_unames:
        for prompt_type in prompt_types:
            for n_shot in n_shot_list:
                for seed in seeds:
                    for icl_max_proof_by_contradiction_per_label in icl_max_proof_by_contradiction_per_label_args:
                        setting = {
                            'dataset_uname': dataset_uname,
                            'prompt_type': prompt_type,
                            'n_shot': n_shot,
                            'seed': seed,
                            'icl_max_proof_by_contradiction_per_label': icl_max_proof_by_contradiction_per_label,
                        }
                        setting.update(
                            get_dataset_setting(
                                'run_prover',
                                dataset_uname=dataset_uname,
                                top_dirs=DATASETS_DIRS,
                            )
                        )

                        output_dir = build_dir(
                            setting,
                            top_dir=str(
                                Path(output_top_dir) /
                                f'dtst_nm={setting["dataset_uname"]}' /
                                f'prmpt_typ={setting["prompt_type"]}' /
                                f'n_sht={setting["n_shot"]}'
                            ),
                            short=True,
                            dirname_ignore_params=[
                                'dataset_uname',
                                'dataset_config_name',
                                'instruction',
                                'predict_with_generate',
                                'remove_unused_columns',
                                'streaming',
                                'train_file',
                                'validation_file',
                                'test_file',
                                'prompt_type',
                                'n_shot',
                            ],
                            save_params=True
                        )

                        command = ' '.join([
                            'python ./make_prompts.py',
                            f'--output-dir {str(output_dir)}',
                            maybe_option_value('--dataset-name', setting.get('dataset_name', None)),
                            maybe_option_value('--dataset-config-name', setting.get('dataset_config_name', None)),
                            maybe_option_value('--train-file', setting.get('train_file', None)),
                            maybe_option_value('--test-file', setting.get('test_file', None)),
                            f'--prompt-type {setting["prompt_type"]}',
                            f'--n-shot {setting["n_shot"]}',
                            maybe_option_value('--icl-max-proof-by-contradiction-per-label', setting.get('icl_max_proof_by_contradiction_per_label', None)),
                            f'--seed {setting["seed"]}',
                        ])

                        run_by_engine(
                            engine,
                            command,
                            output_dir,
                            hours=1,
                            wait_until_finish=wait_until_finish,
                            dry_run=dry_run
                        )

    logger.info('------------- 10.make_prompts.py finished !! -----------')


if __name__ == '__main__':
    main()
