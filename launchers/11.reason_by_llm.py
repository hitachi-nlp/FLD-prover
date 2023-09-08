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

    # input_top_dir = Path('./outputs/10.make_prompts.py/20230711.refactor_distractors/dtst_nm=20230711.finalize.D3/prmpt_typ=in_context_examples.COT.v1/n_sht=10')
    # output_top_dir = Path('./outputs/11.reason_by_llm/20230711.refactor_distractors')

    # input_top_dir = Path('./outputs/10.make_prompts.py/2023-08-31.jpn/')
    # output_top_dir = Path('./outputs/11.reason_by_llm/2023-08-31.jpn/')

    input_top_dir = Path('./outputs/10.make_prompts.py/20230905.LLM_FS/')
    output_top_dir = Path('./outputs/11.reason_by_llm/20230905.LLM_FS/')

    # ----------------- settings ---------------------

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

    engine = SubprocessEngine()   # for debug
    # engine = QsubEngine('ABCI', 'rt_G.large')

    model_names = [
        # 'openai.text-davinci-003',

        # 'openai.gpt-3.5-turbo',
        # 'openai.gpt-3.5-turbo-16k'

        # 'openai.gpt-4-0314',
        'openai.gpt-4',
        # 'openai.gpt-4-32k',
    ]

    # max_samples = 1
    # max_samples = 5
    # max_samples = 10
    max_samples = 30
    # max_samples = 50
    # max_samples = 100
    # max_samples = None

    skip_if_exists = False
    dry_run = False

    # ----------------- running ---------------------

    prompt_paths = sorted(input_top_dir.glob('**/prompts.jsonl'))

    logger.warning('********* will start after sleeping 5 seconds ... Please verify the following setting is on your demand ********* ')
    logger.info('== paths ==')
    for path in prompt_paths:
        logger.info('     - ' + str(path))
    logger.info('== models ==')
    for model_name in model_names:
        logger.info('     - ' + model_name)

    # wait a min to consider carmly whether the paths are right
    # since the reasoning will be charged.
    time.sleep(5)

    for prompt_path in prompt_paths:
        for model_name in model_names:
            setting = {
                'input_path': str(prompt_path),
                'model_name': model_name,
                'max_samples': max_samples,
            }

            dataset_setting = json.load(open(prompt_path.parent / 'lab.params.json'))
            if dataset_setting['dataset_uname'] not in dataset_unames:
                continue

            setting.update({
                f'dataset.{name}': val
                for name, val in dataset_setting.items()
            })

            output_dir = build_dir(
                setting,
                top_dir=str(
                    Path(output_top_dir)
                    / f'dtst_nm={setting.get("dataset.dataset_uname", None)}'
                ),
                short=True,
                dirname_ignore_params=[
                    'dataset.dataset_uname',
                    'dataset.train_file',
                    'dataset.validation_file',
                    'dataset.test_file',

                    'input_path',
                ],
                save_params=True
            )
            output_path = output_dir / 'replies.jsonl'

            if skip_if_exists and output_path.exists():
                logger.warning('skip evaluating for the existing results "%s"', str(output_path))
            else:
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
