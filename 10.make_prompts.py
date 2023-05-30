#!/usr/bin/env python
import logging
from pathlib import Path


import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger
from lab import build_dir

from experimental_setting import (
    get_dataset_paths,
    run_by_engine,
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)
    output_top_dir = Path('./outputs/10.make_prompts.py/2023-05-29/sFLD-impl.use_fixed_translation')

    local_dataset_names = [
        # 'FLD.debug.2023-05-13',

        '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP',   # sFLD-impl
        # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',              # FLD-impl
        # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000'                   # FLD.2
    ]

    seeds = [0]
    n_shot_list = [
        3,
        10,
        30,
        # 100,
        # 300,
    ]

    prompt_types = [
        'in_context_examples',
        'in_context_examples.COT',
    ]

    engine = SubprocessEngine()   # for debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    use_test_as_train = False  # for debugging

    DATASETS_DIRS = [
        # './NLProofS/outputs.FLD/10.create_FLD_corpus/20221203.first_exp',
        # './NLProofS/outputs.FLD/10.create_FLD_corpus/20221217.back_to_the_past',
        # './NLProofS/outputs/00.create_cc100_corpus.py/',
        './FLD-generator/outputs/10.create_FLD_corpus/20230529.use_fixed_translation_for_LLM',
    ]

    dry_run = False

    for local_dataset_name in local_dataset_names:
        dataset_paths = get_dataset_paths(local_dataset_name,
                                          DATASETS_DIRS,
                                          use_test_as_val=False,
                                          use_test_as_train=use_test_as_train)

        for split_name, path in dataset_paths.items():
            if path is None:
                continue
            if not Path(path).exists:
                raise Exception(f'{split_name} dataset does not exist at {path}')

        for prompt_type in prompt_types:
            for n_shot in n_shot_list:
                for seed in seeds:

                    setting = {
                        'local_dataset_name': local_dataset_name,
                        'prompt_type': prompt_type,
                        'n_shot': n_shot,
                        'seed': seed,
                    }
                    setting.update(dataset_paths)

                    output_dir = build_dir(
                        setting,
                        top_dir=str(
                            Path(output_top_dir)
                            / f'dtst_nm={setting["local_dataset_name"]}'
                            / f'prmpt_typ={setting["prompt_type"]}'
                        ),
                        short=True,
                        dirname_ignore_params=[
                            'local_dataset_name',
                            'train_file',
                            'validation_file',
                            'test_file',
                            'prompt_type',
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
