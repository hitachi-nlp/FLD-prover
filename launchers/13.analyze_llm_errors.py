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
    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/2023-05-29/sFLD-impl')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/2023-05-29/sFLD-impl')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230529.use_fixed_translation_for_LLM')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230529.use_fixed_translation_for_LLM')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230529.use_fixed_translation_for_LLM.fewshot_label_wise')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230529.use_fixed_translation_for_LLM.fewshot_label_wise')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230601.fix_translation/dtst_nm=20230529.use_fixed_translation_for_LLM.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000/allwd_addtnl_prf_stps=5/rply.dtst.n_sht=20/rply.dtst.prmpt_typ=in_context_examples.COT.v2')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230601.fix_translation')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230615.formula_checkers')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230615.formula_checkers')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230621.formula_checkers/dtst_nm=20230621.formula_checkers.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000.wo_theorems.wo_translation_dist')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230621.formula_checkers')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230707.finalize')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230707.finalize')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230707.finalize.fix')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230707.finalize.fix')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230710.update_translation')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230710.update_translation')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230710.update_translation.bf51eb2')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230710.update_translation.bf51eb2')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230710.update_translation.7485fef')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230710.update_translation.7485fef')

    # input_top_dir = Path('./outputs/12.evaluate_llm_proofs/20230711.refactor_distractors')
    # output_top_dir = Path('./outputs/13.analyze_llm_errors/20230711.refactor_distractors')

    input_top_dir = Path('./outputs/12.evaluate_llm_proofs/2023-08-31.jpn/')
    output_top_dir = Path('./outputs/13.analyze_llm_errors/2023-08-31.jpn/')

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
