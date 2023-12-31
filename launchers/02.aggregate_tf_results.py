#!/usr/bin/env python
import logging
from pathlib import Path

import click
from script_engine import SubprocessEngine
from logger_setup import setup as setup_logger

from FLD_user_shared_settings import run_by_engine

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # input_dirs = [
    #     './outputs/01.train.py/20230826.jpn',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230826.jpn')

    # input_dirs = [
    #     './outputs/01.train.py/20230901.overfit',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230901.overfit')

    # input_dirs = [
    #     './outputs/01.train.py/20230903.overfit',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230903.overfit')

    # input_dirs = [
    #     './outputs/01.train.py/20230904.LLM_FS',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230904.LLM_FS')

    # input_dirs = [
    #     './outputs/01.train.py/20230905.LLM_FS',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230905.LLM_FS')

    # input_dirs = [
    #     './outputs/01.train.py/20230910.preliminary',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230910.preliminary')

    # input_dirs = [
    #     './outputs/01.train.py/20230911.FT.gpt',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230911.FT.gpt')

    # input_dirs = [
    #     './outputs/01.train.py/20230916.jpn',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230916.jpn')

    # input_dirs = [
    #     './outputs/01.train.py/20230916.jpn.FT',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230916.jpn.FT')

    # input_dirs = [
    #     './outputs/01.train.py/20230919.jpn',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230919.jpn')

    # [!] ---------- JFLD submission -----------
    # input_dirs = [
    #     './outputs/01.train.py/20230919.jpn',
    #     './outputs/01.train.py/20230919.jpn.seed--1',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20230919.jpn.seed--0-1')

    # input_dirs = [
    #     './outputs/01.train.py/20231005.jpn.seed--0',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231005.jpn.seed--0')

    # input_dirs = [
    #     './outputs/01.train.py/20231008.jpn.run_causal_prover',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231008.jpn.run_causal_prover')

    # input_dirs = [
    #     './outputs/01.train.py/20231203.jpn',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231203.jpn')

    # input_dirs = [
    #     './outputs/01.train.py/20231203.jpn.no_subproof_for_unknown',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231203.jpn.no_subproof_for_unknown20231203.jpn')

    # input_dirs = [
    #     './outputs/01.train.py/20231206.new_models',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231206.new_models')

    # input_dirs = [
    #     './outputs/01.train.py/2023-12-06.elyza_fix',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/2023-12-06.elyza_fix')

    # input_dirs = [
    #     './outputs/01.train.py/2023-12-07.elyza_before',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/2023-12-07.elyza_before')

    # input_dirs = [
    #     './outputs/01.train.py/2023-12-06.no_subproof_for_unknown.max_new_tokens=None',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/2023-12-06.no_subproof_for_unknown.max_new_tokens=None')

    # input_dirs = [
    #     './outputs/01.train.py/2023-12-06.D8',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/2023-12-06.D8')

    # input_dirs = [
    #     './outputs/01.train.py/2023-12-06.D8',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/2023-12-06.D8')

    # input_dirs = [
    #     './outputs/01.train.py/20231213.jpn',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231213.jpn')

    # input_dirs = [
    #     './outputs/01.train.py/20231213.jpn.seed--1',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231213.jpn.seed--1')

    # input_dirs = [
    #     './outputs/01.train.py/20231213.jpn',
    #     './outputs/01.train.py/20231213.jpn.seed--1',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231213.jpn.seed--0-1')

    # input_dirs = [
    #     './outputs/01.train.py/20231223.seed--1.timeout_fix',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231223.seed--1.timeout_fix')

    # input_dirs = [
    #     './outputs/01.train.py/20231226.jpn.epoch--10',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231226.jpn.epoch--10')

    # input_dirs = [
    #     './outputs/01.train.py/20231230.jpn.seed--0',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231230.jpn.seed--0')

    # input_dirs = [
    #     './outputs/01.train.py/20231230.jpn.seed--0',
    #     './outputs/01.train.py/20231213.jpn.seed--1',
    # ]
    # output_dir = Path('./outputs/02.aggregate_tf_results.py/20231231.jpn.seed--0-1')

    input_dirs = [
        './outputs/01.train.py/20231230.jpn.seed--0',
        './outputs/01.train.py/20231213.jpn.seed--1',
        './outputs/01.train.py/20231230.jpn.swallow-70b.seed--0',
    ]
    output_dir = Path('./outputs/02.aggregate_tf_results.py/20231231.jpn.seed--0-1.with_swallow-70b')

    command = ' '.join([
        'python ./scripts/aggregate_tf_results.py',
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
