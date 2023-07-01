#!/usr/bin/env python
import os
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple
import json
from pprint import pprint

from logger_setup import setup as setup_logger
import click
from FLD_task import prettify_proof_text, prettify_context_text


logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_path')
@click.argument('output_dir')
# @click.option('--answer-accuracy-threshold', type=float, default=0.1)
@click.option('--log-level', default='INFO')
def main(input_path, output_dir, log_level):
    setup_logger(level=log_level)
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    errors_path = output_dir / 'errors.txt'
    with open(errors_path, 'w') as f_err:
        for i_line, line in enumerate(open(input_path)):
            sample = json.loads(line.strip('\n'))

            # proof_accuracy = sample['metrics']['proof_accuracy']
            # if proof_accuracy >= proof_accuracy_threshold:
            #     continue

            proof_accuracy = sample['metrics']['proof_accuracy.zero_one']
            if proof_accuracy > 0.0:
                continue

            context = sample['example']['context']
            hypothesis = sample['example']['hypothesis']
            proof_gold = sample['gold_proof']
            proof_pred = sample['reply']

            f_err.write('\n\n\n\n\n')
            f_err.write(f'****************************************** example-{i_line} ******************************************')

            f_err.write('\n\n===================== context =====================\n')
            f_err.write(prettify_context_text(context))

            f_err.write('\n\n===================== hypothesis =====================\n')
            f_err.write(hypothesis)

            f_err.write('\n\n===================== proof_gold =====================\n')
            f_err.write(prettify_proof_text(proof_gold))

            f_err.write('\n\n===================== proof_pred =====================\n')
            f_err.write(prettify_proof_text(proof_pred))


if __name__ == '__main__':
    main()
