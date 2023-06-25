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
    # input_top_dir = Path('./outputs/10.make_prompts.py/2023-05-29/sFLD-impl.use_fixed_translation/dtst_nm=20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP/prmpt_typ=in_context_examples.COT/n_sht=10/sd=0/')
    # output_top_dir = Path('./outputs/11.reason_by_llm/2023-05-29/sFLD-impl')

    # input_top_dir = Path('./outputs/10.make_prompts.py/20230529.use_fixed_translation_for_LLM/dtst_nm=20230529.use_fixed_translation_for_LLM.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000/prmpt_typ=in_context_examples.COT/n_sht=10/sd=0/')
    # output_top_dir = Path('./outputs/11.reason_by_llm/20230529.use_fixed_translation_for_LLM')

    # input_top_dir = Path('./outputs/10.make_prompts.py/20230529.use_fixed_translation_for_LLM.fewshot_label_wise/dtst_nm=20230529.use_fixed_translation_for_LLM.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000/prmpt_typ=in_context_examples/n_sht=10')
    # output_top_dir = Path('./outputs/11.reason_by_llm/20230529.use_fixed_translation_for_LLM.fewshot_label_wise')

    # input_top_dir = Path('./outputs/10.make_prompts.py/20230529.use_fixed_translation_for_LLM.fewshot_label_wise/dtst_nm=20230529.use_fixed_translation_for_LLM.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000/prmpt_typ=in_context_examples.COT/n_sht=10')
    # output_top_dir = Path('./outputs/11.reason_by_llm/20230529.use_fixed_translation_for_LLM.fewshot_label_wise')

    # input_top_dir = Path('./outputs/10.make_prompts.py/20230601.fix_translation/dtst_nm=20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP/prmpt_typ=in_context_examples.COT/n_sht=10')
    # input_top_dir = Path('./outputs/10.make_prompts.py/20230601.fix_translation/dtst_nm=20230529.use_fixed_translation_for_LLM.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000/prmpt_typ=in_context_examples.COT/n_sht=10')
    # input_top_dir = Path('./outputs/10.make_prompts.py/20230601.fix_translation/dtst_nm=20230529.use_fixed_translation_for_LLM.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000/prmpt_typ=in_context_examples.COT/n_sht=20')
    # input_top_dir = Path('./outputs/10.make_prompts.py/20230601.fix_translation/dtst_nm=20230529.use_fixed_translation_for_LLM.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000/prmpt_typ=in_context_examples.COT.v1/n_sht=20/sd=0/')
    # output_top_dir = Path('./outputs/11.reason_by_llm/20230601.fix_translation')

    # input_top_dir = Path('./outputs/10.make_prompts.py/20230601.fix_translation/dtst_nm=20230529.use_fixed_translation_for_LLM.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000/prmpt_typ=in_context_examples.COT.v2/n_sht=20/sd=0/')
    # output_top_dir = Path('./outputs/11.reason_by_llm/20230601.fix_translation')

    # input_top_dir = Path('./outputs/10.make_prompts.py/20230615.formula_checkers/dtst_nm=20230615.formula_checkers.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000.wo_theorems')
    # output_top_dir = Path('./outputs/11.reason_by_llm/20230615.formula_checkers')

    input_top_dir = Path('./outputs/10.make_prompts.py/20230621.formula_checkers/dtst_nm=20230621.formula_checkers.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000.wo_theorems.wo_translation_dist')
    output_top_dir = Path('./outputs/11.reason_by_llm/20230621.formula_checkers/')

    # ----------------- settings ---------------------
    engine = SubprocessEngine()   # for debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    model_names = [
        # 'text-davinci-003',   # stupid. proof accuracy = 0.0 for sFLD-impl :(

        # 'gpt-3.5-turbo',      # stupid. proof accuracy = 0.2 for sFLD-impl :(
        'gpt-3.5-turbo-16k'

        # 'gpt-4',                # smart. proof accuracy = 0.5 for sFLD-impl :)
        # 'gpt-4-32k',
    ]

    # max_samples = 1
    # max_samples = 5
    max_samples = 10
    # max_samples = 30
    # max_samples = None

    dry_run = False

    # ----------------- running ---------------------

    prompt_paths = sorted(input_top_dir.glob('**/prompts.jsonl'))

    logger.warning('********* will start after sleeping 10 seconds ... Please verify the following setting is on your demand ********* ')
    logger.info('== paths ==')
    for path in prompt_paths:
        logger.info('     - ' + str(path))
    logger.info('== models ==')
    for model_name in model_names:
        logger.info('     - ' + model_name)

    # wait a min to consider carmly whether the paths are right
    # since the reasoning will be charged.
    time.sleep(10)

    for prompt_path in prompt_paths:
        for model_name in model_names:
            setting = {
                'input_path': str(prompt_path),
                'model_name': model_name,
                'max_samples': max_samples,
            }

            dataset_setting = json.load(open(prompt_path.parent / 'lab.params.json'))
            setting.update({
                f'dataset.{name}': val
                for name, val in dataset_setting.items()
            })

            output_dir = build_dir(
                setting,
                top_dir=str(
                    Path(output_top_dir)
                    / f'dtst_nm={setting.get("dataset.local_dataset_name", None)}'
                ),
                short=True,
                dirname_ignore_params=[
                    'dataset.local_dataset_name',
                    'dataset.train_file',
                    'dataset.validation_file',
                    'dataset.test_file',

                    'input_path',
                ],
                save_params=True
            )
            output_path = output_dir / 'replies.jsonl'

            command = ' '.join([
                'python ./reason_by_llm.py',
                str(prompt_path),
                str(output_path),
                f'--model-name {model_name}',
                f'--max-samples {max_samples}',
            ])

            run_by_engine(
                engine,
                command,
                output_dir,
                hours=1,
                dry_run=dry_run
            )

    logger.info('------------- 11.reason_by_llm.py finished !! -----------')


if __name__ == '__main__':
    main()
