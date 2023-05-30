#!/usr/bin/env python
import os
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple
import json
from collections import defaultdict
import statistics
from pprint import pprint

from logger_setup import setup as setup_logger
import click
from FLD_task.evaluate.scoring import calc_metrics


logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_path')
@click.argument('output_dir')
@click.option('--similarity-threshold', is_flag=True, default=False)
@click.option('--allowed-additional-proof-steps', type=int, default=None)
@click.option('--log-level', default='INFO')
def main(input_path, output_dir, similarity_threshold, allowed_additional_proof_steps, log_level):
    setup_logger(level=log_level)
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    metrics_path = output_dir / 'metrics.jsonl'
    all_metrics: Dict[str, float] = defaultdict(list)
    with open(metrics_path, 'w') as f_out:
        for line in open(input_path):
            sample = json.loads(line.strip('\n'))

            gold = sample['serial']['next_step']
            pred = sample['completion']

            metrics = calc_metrics(
                gold,
                pred,
                similarity_threshold=similarity_threshold,
                allowed_additional_proof_steps=allowed_additional_proof_steps,
            )
            sample['metrics'] = metrics

            for name, val in metrics.items():
                all_metrics[name].append(val)
            f_out.write(json.dumps(sample) + '\n')

    metrics_summary = {}
    for name, vals in all_metrics.items():
        metrics_summary[name] = statistics.mean(vals)

    json.dump(metrics_summary,
              open(str(output_dir / 'metrics_summary.json'), 'w'),
              ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    pprint(metrics_summary)


if __name__ == '__main__':
    main()
