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
    # input_top_dir = Path('./outputs/11.reason_by_llm/2023-05-29/sFLD-impl')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/2023-05-29/sFLD-impl')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230529.use_fixed_translation_for_LLM')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230529.use_fixed_translation_for_LLM')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230529.use_fixed_translation_for_LLM.fewshot_label_wise')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230529.use_fixed_translation_for_LLM.fewshot_label_wise')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230601.fix_translation/')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230601.fix_translation')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230615.formula_checkers')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230615.formula_checkers')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230621.formula_checkers/dtst_nm=20230621.formula_checkers.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000.wo_theorems.wo_translation_dist')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230621.formula_checkers/')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230707.finalize')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230707.finalize')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230707.finalize.fix')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230707.finalize.fix')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230710.update_translation')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230710.update_translation')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230710.update_translation.bf51eb2')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230710.update_translation.bf51eb2')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230710.update_translation.7485fef')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230710.update_translation.7485fef')

    # input_top_dir = Path('./outputs/11.reason_by_llm/20230711.refactor_distractors/dtst_nm=20230711.finalize.D3/')
    # output_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230711.refactor_distractors')

    input_top_dir = Path('./outputs/11.reason_by_llm/2023-08-31.jpn/')
    output_top_dir = Path('./outputs/12.evaluate_llm_proofs/2023-08-31.jpn/')

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
