#!/usr/bin/env python
import logging
import re
from pathlib import Path
import json
import statistics
from collections import defaultdict

import pandas as pd
from logger_setup import setup as setup_logger
from machine_learning.analysis.dataframe import aggregate
from machine_learning.analysis.visualization import setup as setup_visualization, VisualizationConfig
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
    # setup_visualization(VisualizationConfig())

    LAB_ATTR_NAMES = [
        'reply.dataset.dataset_uname',
        'reply.dataset.seed',
        'reply.dataset.n_shot',
        'reply.dataset.icl_max_proof_by_contradiction_per_label',

        'reply.model_name',
        # 'reply.dataset.prompt_type',

        'reply.max_samples',
    ]

    METRIC_NAMES = [
        'proof_accuracy.zero_one',
        'answer_accuracy',
    ]

    def lab_short_name(name):
        return re.sub(r'^dataset\.', '', re.sub(r'^reply\.', '', name))

    df_dict = defaultdict(list)
    for _input_dir in input_dirs:
        for metrich_summary_path in _input_dir.glob('**/*/metrics_summary.json'):
            metrics = json.load(open(str(metrich_summary_path)))
            lab_setting = json.load(open(str(metrich_summary_path.parent / 'lab.params.json')))

            if len(metrics) == 0:
                logger.warning('no metrics found in %s', str(metrich_summary_path))
                continue

            for name in LAB_ATTR_NAMES:
                df_dict[lab_short_name(name)].append(lab_setting.get(name, None))

            for type_name, type_metrics in metrics.items():
                for name in METRIC_NAMES:
                    val = type_metrics[name]
                    df_dict['__'.join([type_name, name])].append(val)

    def is_metric_col(name: str) -> bool:
        return name.find('accuracy') >= 0

    def is_variation_col(name: str) -> bool:
        return name.find('seed') >= 0 or name.find('max_samples') >= 0

    merged_df = pd.DataFrame(df_dict)
    merged_df = merged_df.sort_values(by=merged_df.columns.tolist())
    agg_df = aggregate(merged_df,
                       [lab_short_name(col) for col in LAB_ATTR_NAMES + METRIC_NAMES
                        if not is_metric_col(col) and not is_variation_col(col)],
                        aggregate_funcs={col: lambda vals: statistics.mean(vals)
                                         for col in merged_df.columns
                                         if is_metric_col(col)})

    print('\n================ results =============== ')
    print(merged_df)
    print('\n================ results aggregated =============== ')
    print(agg_df)

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info('write into %s', str(output_dir))

        merged_df.to_csv(str(output_dir / 'results.tsv'), sep='\t', index=None)
        agg_df.to_csv(str(output_dir / 'results_agg.tsv'), sep='\t', index=None)


if __name__ == '__main__':
    main()
