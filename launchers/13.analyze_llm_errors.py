#!/usr/bin/env python
import logging
from pathlib import Path
import time
import json
import shutil 

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

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230711.refactor_distractors')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230711.refactor_distractors')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/2023-08-31.jpn/')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/2023-08-31.jpn/')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20231106.refaactor/')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20231106.refaactor/')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20231107.preliminary/')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20231107.preliminary/')

    input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20231107.preliminary.many_samples/')
    output_top_dir = Path('./outputs/13.analyze_llm_errors/20231107.preliminary.many_samples/')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20231107.preliminary.many_seeds/')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20231107.preliminary.many_seeds/')

    # ----------------- settings ---------------------
    engine = SubprocessEngine()   # for debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    # ----------------- run ---------------------
    dry_run = False
    # answer_accuracy_threshold = 0.1

    for metrics_path in input_top_dir.glob('**/metrics.jsonl'):
        setting = {
            'input_path': str(metrics_path),
            # 'answer_accuracy_threshold': answer_accuracy_threshold,
        }

        metrics_setting = json.load(open(metrics_path.parent / 'lab.params.json'))
        setting.update({
            f'metrics.{name}': val
            for name, val in metrics_setting.items()
        })

        output_dir = build_dir(
            setting,
            top_dir=str(
                Path(output_top_dir)
                / f'dtst_nm={setting.get("metrics.reply.dataset.local_dataset_name", None)}'
            ),
            short=True,
            dirname_ignore_params=[
                'metrics.reply.dataset.local_dataset_name',
                'metrics.reply.dataset.train_file',
                'metrics.reply.dataset.validation_file',
                'metrics.reply.dataset.test_file',
                'metrics.reply.prompt_path',
                'metrics.reply.input_path',

                'metrics.input_path',

                'input_path',
            ],
            save_params=True,
        )
        
        command = ' '.join([
            'python ./analyze_llm_errors.py',
            str(metrics_path),
            str(output_dir),
            # f'--answer-accuracy-threshold {answer_accuracy_threshold}',
        ])

        SubprocessEngine().run(f'cp {str(metrics_path.parent / "*")} {str(output_dir)}')
        run_by_engine(
            engine,
            command,
            output_dir,
            hours=1,
            dry_run=dry_run
        )

    logger.info('------------- ./13.analyze_llm_errors.py finished !! -----------')

if __name__ == '__main__':
    main()
