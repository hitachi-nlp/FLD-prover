#!/usr/bin/env python
import logging
from pathlib import Path
import time
import json

import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger
from lab import build_dir

from experimental_setting import run_by_engine

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # ----------------- input output paths ---------------------
    input_top_dir = Path('./outputs/11.reason_by_llm/sFLD-impl')
    output_top_dir = Path('./outputs/12.evaluate_llm_proofs')

    # ----------------- settings ---------------------

    engine = SubprocessEngine()   # for debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    allowed_additional_proof_steps = 5
    similarity_threshold = False

    dry_run = False

    # ----------------- running ---------------------

    for cmpl_path in input_top_dir.glob('**/completions.jsonl'):
        setting = {
            'input_path': str(cmpl_path),
            'allowed_additional_proof_steps': allowed_additional_proof_steps,
            'similarity_threshold': similarity_threshold,
        }

        cmpl_setting = json.load(open(cmpl_path.parent / 'lab.params.json'))
        setting.update({
            f'completion.{name}': val
            for name, val in cmpl_setting.items()
        })

        output_dir = build_dir(
            setting,
            top_dir=str(
                Path(output_top_dir)
                / f'dtst_nm={setting.get("completion.dataset.local_dataset_name", None)}'
            ),
            short=True,
            dirname_ignore_params=[
                'completion.dataset.local_dataset_name',
                'completion.dataset.train_file',
                'completion.dataset.validation_file',
                'completion.dataset.test_file',

                'completion.input_path',

                'input_path',
            ],
            save_params=True
        )

        command = ' '.join([
            'python ./evaluate_llm_proofs.py',
            str(cmpl_path),
            str(output_dir),
            '--similarity-threshold' if similarity_threshold else '',
            f'--allowed-additional-proof-steps {allowed_additional_proof_steps}',
        ])

        run_by_engine(
            engine,
            command,
            output_dir,
            hours=1,
            dry_run=dry_run
        )

    logger.info('------------- ./12.evaluate_llm_proofs.py finished !! -----------')


if __name__ == '__main__':
    main()
