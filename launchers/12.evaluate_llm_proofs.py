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

    # input_top_dir = Path('./outputs/11.reason_by_llm/2023-08-31.jpn/')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/2023-08-31.jpn/')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230905.LLM_FS/')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230905.LLM_FS/')

    input_top_dir = Path('./outputs/11.reason_by_llm/20230919.jpn/')
    output_top_dir = Path('./outputs/12.evaluate_llm_proofs/220230919.jpn/')

    # ----------------- settings ---------------------

    engine = SubprocessEngine()   # for debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    # allowed_additional_proof_steps = 5
    # similarity_threshold = False

    skip_if_exists = False
    dry_run = False

    # ----------------- running ---------------------

    for reply_path in input_top_dir.glob('**/replies.jsonl'):
        setting = {
            'input_path': str(reply_path),
            # 'allowed_additional_proof_steps': allowed_additional_proof_steps,
            # 'similarity_threshold': similarity_threshold,
        }

        reply_setting = json.load(open(reply_path.parent / 'lab.params.json'))
        setting.update({
            f'reply.{name}': val
            for name, val in reply_setting.items()
        })

        output_dir = build_dir(
            setting,
            top_dir=str(
                Path(output_top_dir)
                / f'dtst_nm={setting.get("reply.dataset.dataset_uname", None)}'
            ),
            short=True,
            dirname_ignore_params=[
                'reply.dataset.dataset_uname',
                'reply.dataset.train_file',
                'reply.dataset.validation_file',
                'reply.dataset.test_file',

                'reply.input_path',

                'input_path',
            ],
            save_params=True
        )
        
        command = ' '.join([
            'python ./evaluate_llm_proofs.py',
            str(reply_path),
            str(output_dir),
            # '--similarity-threshold' if similarity_threshold else '',
            # f'--allowed-additional-proof-steps {allowed_additional_proof_steps}',
        ])

        if skip_if_exists and (output_dir / 'metrics_summary.json').exists():
            logger.warning('skip evaluating for the existing results "%s"', str(output_dir))
        else:
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
