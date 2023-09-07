#!/usr/bin/env python
import logging
from pathlib import Path
import json
from collections import defaultdict

import pandas as pd
from logger_setup import setup as setup_logger
import click


logger = logging.getLogger(__name__)


@click.command()
@click.option('--input_dir', multiple=True, default=[])
@click.option('--output_dir', default=None)
@click.option('--log-level', default='INFO')
def main(input_dir, output_dir, log_level):
    setup_logger(level=log_level)
    input_dirs = [Path(_input_dir) for _input_dir in input_dir]
    output_dir = Path(output_dir) if output_dir is not None else None

    LAB_ATTR_NAMES = [
        'reply.dataset.dataset_uname',
        'reply.dataset.n_shot',
        'reply.model_name',

        'reply.dataset.seed',
        'reply.dataset.prompt_type',
        'reply.max_samples',
    ]

    METRIC_NAMES = [
        'proof_accuracy.zero_one',
        'answer_accuracy',
    ]

    df_dict = defaultdict(list)
    for _input_dir in input_dirs:
        for metrich_summary_path in _input_dir.glob('**/*/metrics_summary.json'):
            metrics = json.load(open(str(metrich_summary_path)))
            lab_setting = json.load(open(str(metrich_summary_path.parent / 'lab.params.json')))

            for name in LAB_ATTR_NAMES:
                df_dict[name].append(lab_setting.get(name, None))


            for type_name, type_metrics in metrics.items():
                for name in METRIC_NAMES:
                    val = type_metrics[name]
                    df_dict['__'.join([type_name, name])].append(val)

    merged_df = pd.DataFrame(df_dict)
    print(merged_df)

    if output_dir is not None:
        out_path = output_dir / 'results.tsv'
        logger.info('write into %s', str(out_path))
        output_dir.mkdir(exist_ok=True, parents=True)
        merged_df.to_csv(out_path, sep='\t', index=None)


if __name__ == '__main__':
    main()
