#!/usr/bin/env python
import logging
from pathlib import Path


import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger
from lab import build_dir

from experimental_setting import (
    get_local_dataset_paths,
    run_by_engine,
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)
    # output_top_dir = Path('./outputs/10.make_prompts.py/2023-05-29/sFLD-impl.use_fixed_translation')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230529.use_fixed_translation_for_LLM')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230529.use_fixed_translation_for_LLM.fewshot_label_wise')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230529.use_fixed_translation_for_LLM.fewshot_label_wise')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230601.fix_translation')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230615.formula_checkers')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230621.formula_checkers')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230628.make_harder')

    # output_top_dir = Path('./outputs/10.make_prompts.py/20230707.finalize')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230707.finalize.fix')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230710.update_translation')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230710.update_translation.bf51eb2')
    # output_top_dir = Path('./outputs/10.make_prompts.py/20230710.update_translation.7485fef')

    # output_top_dir = Path('./outputs/10.make_prompts.py/20230711.refactor_distractors')

    # output_top_dir = Path('./outputs/10.make_prompts.py/2023-08-31.jpn')

    output_top_dir = Path('./outputs/10.make_prompts.py/20230905.LLM_FS')

    DATASETS_DIRS = [
        # './NLProofS/outputs.FLD/10.create_FLD_corpus/20221203.first_exp',
        # './NLProofS/outputs.FLD/10.create_FLD_corpus/20221217.back_to_the_past',
        # './NLProofS/outputs/00.create_cc100_corpus.py/',
        # './FLD-generator/outputs/10.create_FLD_corpus/20230529.use_fixed_translation_for_LLM',
        # './FLD-generator/outputs/10.create_FLD_corpus/20230601.fix_translation',
        # './FLD-generator/outputs/10.create_FLD_corpus/20230615.formula_checkers',
        # './FLD-generator/outputs/10.create_FLD_corpus/20230621.formula_checkers',

        # './outputs/00.fix_FLD_schema.py/20230626.many_bugs_fixed',
        # './outputs/00.fix_FLD_schema.py/20230628.make_harder',
        # './outputs/00.fix_FLD_schema.py/20230701.finalize',

        # './outputs/00.fix_FLD_schema.py/20230707.finalize',

        # './outputs/00.fix_FLD_schema.py/20230710.update_translation',
        # './outputs/00.fix_FLD_schema.py/20230710.update_translation.bf51eb2',
        # './outputs/00.fix_FLD_schema.py/20230710.update_translation.7485fef',

        # './outputs/00.fix_FLD_schema.py/20230711.refactor_distractors',

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

        # ---------------------------------- 20230904.jpn ------------------------------------
        '20230904.jpn.D1.wo_brnch.wo_dstrct',
        '20230904.jpn.D1.wo_brnch',
        '20230904.jpn.D1',
        '20230904.jpn.D3',
    ]

    prompt_types = [
        # 'ICL',
        # 'ICL-COT',
        # 'ICL-COT.v1',
        'ICL-COT.v2',
    ]

    n_shot_list = [
        # 3,
        # 6,  # for chatGPT web, which have limited context length.

        8,
        # 10,
        # 12,

        # -- over token limit = 8k token --
        # 15,
        # 20,
        # 30,
        # 100,
        # 300,
    ]

    engine = SubprocessEngine()   # for debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    dry_run = False

    # -------------------------- fixed settings --------------------------
    seeds = [0]
    use_test_as_train = False  # for debugging

    # -------------------------- running --------------------------
    for dataset_uname in dataset_unames:
        dataset_paths = get_local_dataset_paths(dataset_uname,
                                                DATASETS_DIRS,
                                                use_test_as_val=False,
                                                use_test_as_train=use_test_as_train,
                                                allow_not_found_splits=True)

        for prompt_type in prompt_types:
            for n_shot in n_shot_list:
                for seed in seeds:

                    setting = {
                        'dataset_uname': dataset_uname,
                        'prompt_type': prompt_type,
                        'n_shot': n_shot,
                        'seed': seed,
                    }
                    setting.update(dataset_paths)

                    output_dir = build_dir(
                        setting,
                        top_dir=str(
                            Path(output_top_dir)
                            / f'dtst_nm={setting["dataset_uname"]}'
                            / f'prmpt_typ={setting["prompt_type"]}'
                            / f'n_sht={setting["n_shot"]}'
                        ),
                        short=True,
                        dirname_ignore_params=[
                            'dataset_uname',
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
                        str(setting['test_file']),
                        str(output_dir),
                        f'--train-path {setting["train_file"]}' if setting.get('train_file', None) is not None else '',
                        f'--prompt-type {prompt_type}',
                        f'--n-shot {str(n_shot)}',
                        f'--seed {str(seed)}',
                    ])

                    run_by_engine(
                        engine,
                        command,
                        output_dir,
                        hours=1,
                        dry_run=dry_run
                    )

    logger.info('------------- 10.make_prompts.py finished !! -----------')


if __name__ == '__main__':
    main()
