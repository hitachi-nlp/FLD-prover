#!/usr/bin/env python
import logging
from pathlib import Path

import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger
# from stance_indication import StanceIndicationMethod

from experimental_setting import (
    get_dataset_paths,
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)
    output_top_dir = Path('./outputs/00.fix_FLD_schema.py/2023-05-15')

    local_dataset_names = [
        # 'EB-task1.shuffled',
        # 'EB-task2.shuffled',

        # 'FLD.debug.2023-05-13',
        # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP',
        '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-5__dataset_size-30000',
    ]

    engine = SubprocessEngine()
    # engine = QsubEngine('ABCI', 'rt_G.large')

    dry_run = False

    DATASETS_DIRS = [
        './NLProofS/outputs.FLD/10.create_FLD_corpus/20221203.first_exp',
        './NLProofS/outputs.FLD/10.create_FLD_corpus/20221217.back_to_the_past',
        './NLProofS/outputs/00.create_cc100_corpus.py/',
    ]

    for local_dataset_name in local_dataset_names:
        output_dir = output_top_dir / local_dataset_name
        output_dir.mkdir(exist_ok=True, parents=True)

        dataset_paths = get_dataset_paths(local_dataset_name,
                                          DATASETS_DIRS,
                                          use_test_as_val=False,
                                          use_test_as_train=False)

        is_settings_copied = False
        for input_path_str in dataset_paths.values():
            if input_path_str is None:
                continue
            input_path = Path(input_path_str)

            output_path = output_dir / input_path.name

            engine.run(
                f'python ./fix_FLD_schema.py {input_path} {str(output_path)}',
                wait_until_finish=True,
                dry_run=dry_run,
            )

            setting_path = input_path.parent.parent / 'lab.params.json'
            if not is_settings_copied and setting_path.exists():
                engine.run(f'cp {str(setting_path)} {str(output_dir)}')
                is_settings_copied = True

    logger.info('------------- ./00.fix_FLD_schema.py finished !! -----------')


if __name__ == '__main__':
    main()
