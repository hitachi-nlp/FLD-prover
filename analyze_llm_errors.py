#!/usr/bin/env python
import os
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple
import json
from pprint import pprint

from logger_setup import setup as setup_logger
import click
from FLD_task import prettify_proof_text, prettify_facts_text


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

    samples = [
        json.loads(line.strip('\n'))
        for line in open(input_path)
    ]
    metrics_types = list(samples[0]['metrics'].keys())

    for metric_type in metrics_types:
        errors_path = output_dir / f'errors.metric--{metric_type}.txt'
        with open(errors_path, 'w') as f_err:

            for i_sample, sample in enumerate(samples):
                proof_accuracy = sample['metrics'][metric_type]['proof_accuracy.zero_one']
                if proof_accuracy > 0.0:
                    continue

                facts = sample['example']['facts']
                hypothesis = sample['example']['hypothesis']
                proof_golds = sample['gold_proofs']
                if len(proof_golds) >= 2:
                    raise NotImplementedError()
                proof_pred = sample['reply']

                f_err.write('\n\n\n\n\n')
                f_err.write(f'****************************************** example-{i_sample} ******************************************')

                f_err.write('\n\n===================== facts =====================\n')
                f_err.write(prettify_facts_text(facts))

                f_err.write('\n\n===================== hypothesis =====================\n')
                f_err.write(hypothesis)

                f_err.write('\n\n===================== proof_gold =====================\n')
                f_err.write(prettify_proof_text(proof_golds[0]))

                f_err.write('\n\n===================== proof_pred =====================\n')
                f_err.write(prettify_proof_text(proof_pred))


if __name__ == '__main__':
    main()
