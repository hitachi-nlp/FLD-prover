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
        for metric in ['proof_accuracy.zero_one', 'answer_accuracy']:
            errors_path = output_dir / f'metric_type--{metric_type}.metric--{metric}.errors.txt'
            corrects_path = output_dir / f'metric_type--{metric_type}.metric--{metric}.corrects.txt'
            with open(errors_path, 'w') as f_err, open(corrects_path, 'w') as f_corr:

                for i_sample, sample in enumerate(samples):
                    if metric_type not in sample['metrics']:
                        f_err.write('\n\n\n\n\n')
                        f_err.write(f'****************************************** example-{i_sample} ******************************************')
                        f_err.write('metrics not found in this example, might be failed to calculate the metrics due to some errors.')
                        continue

                    accuracy = sample['metrics'][metric_type][metric]
                    if accuracy > 0.0:
                        f_out = f_corr
                    else:
                        f_out = f_err

                    facts = sample['example']['facts']
                    hypothesis = sample['example']['hypothesis']
                    proof_gold = sample['gold_proof']
                    proof_pred = sample['reply']

                    f_out.write('\n\n\n\n\n')
                    f_out.write(f'****************************************** example-{i_sample} ******************************************')

                    f_out.write('\n\n===================== facts =====================\n')
                    f_out.write(prettify_facts_text(facts))

                    f_out.write('\n\n===================== hypothesis =====================\n')
                    f_out.write(hypothesis)

                    f_out.write('\n\n===================== proof_gold =====================\n')
                    f_out.write(prettify_proof_text(proof_gold))

                    f_out.write('\n\n===================== proof_pred =====================\n')
                    f_out.write(prettify_proof_text(proof_pred))


if __name__ == '__main__':
    main()
