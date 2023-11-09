#!/usr/bin/env python
import logging
from pathlib import Path

import click
from script_engine import SubprocessEngine
from logger_setup import setup as setup_logger

from experimental_setting import run_by_engine

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # input_dirs = [
    #     './outputs/12.evaluate_llm_proofs/2023-08-31.jpn/',
    # ]
    # output_dir = Path('./outputs/13.aggregate_llm_results.py/2023-08-31.jpn/')

    # input_dirs = [
    #     './outputs/12.evaluate_llm_proofs/20230905.LLM_FS/',
    # ]
    # output_dir = Path('./outputs/13.aggregate_llm_results.py/20230905.LLM_FS/')

    # input_dirs = [
    #     './outputs/12.evaluate_llm_proofs/220230919.jpn/',
    # ]
    # output_dir = Path('./outputs/13.aggregate_llm_results.py/220230919.jpn/')

    # input_dirs = [
    #     './outputs/12.evaluate_llm_proofs/20231107.preliminary/',
    # ]
    # output_dir = Path('./outputs/13.aggregate_llm_results.py/20231107.preliminary/')

    input_dirs = [
        './outputs/12.evaluate_llm_proofs/20231107.preliminary.many_samples/',
        './outputs/12.evaluate_llm_proofs/20231107.preliminary.many_samples.seed--1/',
        './outputs/12.evaluate_llm_proofs/20231107.preliminary.many_samples.seed--2/',
    ]
    output_dir = Path('./outputs/13.aggregate_llm_results.py/20231107.preliminary.many_samples/')

    # input_dirs = [
    #     './outputs/12.evaluate_llm_proofs/20231107.preliminary.many_seeds/',
    # ]
    # output_dir = Path('./outputs/13.aggregate_llm_results.py/20231107.preliminary.many_seeds/')

    # input_dirs = [
    #     './outputs/12.evaluate_llm_proofs/20231109.icl_max_proof_by_contradiction_per_label',
    # ]
    # output_dir = Path('./outputs/13.aggregate_llm_results.py/20231109.icl_max_proof_by_contradiction_per_label')

    # input_dirs = [
    #     './outputs/12.evaluate_llm_proofs/20231109.3-shot',
    # ]
    # output_dir = Path('./outputs/13.aggregate_llm_results.py/20231109.3-shot')

    command = ' '.join([
        'python ./aggregate_llm_results.py',
        ' '.join([f'--input_dir {str(input_dir)}' for input_dir in input_dirs]),
        f'--output_dir {str(output_dir)}',
    ])

    run_by_engine(
        SubprocessEngine(),
        command,
        str(output_dir),
    )


if __name__ == '__main__':
    main()
