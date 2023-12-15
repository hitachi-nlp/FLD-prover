#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List
import json
from collections import defaultdict
import re

import pandas as pd
from logger_setup import setup as setup_logger
import click
from machine_learning.analysis.tensorboard import read_as_dataframe
from machine_learning.analysis.visualization import setup as setup_visualization, VisualizationConfig


logger = logging.getLogger(__name__)


LAB_ATTR_NAMES = [
    'FLD_dataset_uname',
    'learning',

    'model_name_or_path',
    'seed',
    'learning_rate',

    # 'base_config_name',
    # 'generation_max_proof_steps',
    # 'generation_num_beams',
    # 'generation_top_k',
    # 'gradient_accumulation_steps',
    # 'lm_type',
    # 'lora',
    # 'max_grad_norm',
    # 'max_predict_samples',
    # 'max_proof_steps',
    # 'max_source_length',
    # 'max_steps',
    # 'max_target_length',
    # 'max_train_samples',
    # 'per_device_eval_batch_size',
    # 'per_device_train_batch_size',
    # 'proof_sampling',
    # 'sample_negative_proof',
    # 'source_prefix',
    # 'tokenizer_padding',
    # 'warmup_steps',
]

METRIC_NAMES = [
    'eval/extr_stps.D-0.proof_accuracy.zero_one',
    'eval/extr_stps.D-1.proof_accuracy.zero_one',
    'eval/extr_stps.D-2.proof_accuracy.zero_one',
    'eval/extr_stps.D-3.proof_accuracy.zero_one',
    'eval/extr_stps.D-4.proof_accuracy.zero_one',
    'eval/extr_stps.D-5.proof_accuracy.zero_one',
    'eval/extr_stps.D-6.proof_accuracy.zero_one',
    'eval/extr_stps.D-7.proof_accuracy.zero_one',
    'eval/extr_stps.D-8.proof_accuracy.zero_one',
    'eval/extr_stps.D-None.proof_accuracy.zero_one',
    'eval/extr_stps.D-all.proof_accuracy.zero_one',

    'eval/strct.D-0.proof_accuracy.zero_one',
    'eval/strct.D-1.proof_accuracy.zero_one',
    'eval/strct.D-2.proof_accuracy.zero_one',
    'eval/strct.D-3.proof_accuracy.zero_one',
    'eval/strct.D-4.proof_accuracy.zero_one',
    'eval/strct.D-5.proof_accuracy.zero_one',
    'eval/strct.D-6.proof_accuracy.zero_one',
    'eval/strct.D-7.proof_accuracy.zero_one',
    'eval/strct.D-8.proof_accuracy.zero_one',
    'eval/strct.D-None.proof_accuracy.zero_one',
    'eval/strct.D-all.proof_accuracy.zero_one',

    'eval/extr_stps.D-0.answer_accuracy',
    'eval/extr_stps.D-1.answer_accuracy',
    'eval/extr_stps.D-2.answer_accuracy',
    'eval/extr_stps.D-3.answer_accuracy',
    'eval/extr_stps.D-4.answer_accuracy',
    'eval/extr_stps.D-5.answer_accuracy',
    'eval/extr_stps.D-6.answer_accuracy',
    'eval/extr_stps.D-7.answer_accuracy',
    'eval/extr_stps.D-8.answer_accuracy',
    'eval/extr_stps.D-None.answer_accuracy',
    'eval/extr_stps.D-all.answer_accuracy',

    'eval/strct.D-0.answer_accuracy',
    'eval/strct.D-1.answer_accuracy',
    'eval/strct.D-2.answer_accuracy',
    'eval/strct.D-3.answer_accuracy',
    'eval/strct.D-4.answer_accuracy',
    'eval/strct.D-5.answer_accuracy',
    'eval/strct.D-6.answer_accuracy',
    'eval/strct.D-7.answer_accuracy',
    'eval/strct.D-8.answer_accuracy',
    'eval/strct.D-None.answer_accuracy',
    'eval/strct.D-all.answer_accuracy',


    'train/FLD_proof_eval_extr_stps.D-0.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-1.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-2.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-3.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-4.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-5.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-6.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-7.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-8.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-None.proof_accuracy.zero_one',
    'train/FLD_proof_eval_extr_stps.D-all.proof_accuracy.zero_one',

    'train/FLD_proof_eval_strct.D-0.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-1.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-2.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-3.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-4.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-5.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-6.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-7.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-8.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-None.proof_accuracy.zero_one',
    'train/FLD_proof_eval_strct.D-all.proof_accuracy.zero_one',

    'train/FLD_proof_eval_extr_stps.D-0.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-1.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-2.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-3.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-4.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-5.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-6.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-7.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-8.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-None.answer_accuracy',
    'train/FLD_proof_eval_extr_stps.D-all.answer_accuracy',

    'train/FLD_proof_eval_strct.D-0.answer_accuracy',
    'train/FLD_proof_eval_strct.D-1.answer_accuracy',
    'train/FLD_proof_eval_strct.D-2.answer_accuracy',
    'train/FLD_proof_eval_strct.D-3.answer_accuracy',
    'train/FLD_proof_eval_strct.D-4.answer_accuracy',
    'train/FLD_proof_eval_strct.D-5.answer_accuracy',
    'train/FLD_proof_eval_strct.D-6.answer_accuracy',
    'train/FLD_proof_eval_strct.D-7.answer_accuracy',
    'train/FLD_proof_eval_strct.D-8.answer_accuracy',
    'train/FLD_proof_eval_strct.D-None.answer_accuracy',
    'train/FLD_proof_eval_strct.D-all.answer_accuracy',
]


@click.command()
@click.option('--input_dir', multiple=True, default=[])
@click.option('--output_dir', default=None)
@click.option('--log-level', default='INFO')
@click.option('--read-from-eval-json', is_flag=True, default=False)
def main(input_dir, output_dir, log_level, read_from_eval_json):
    setup_logger(level=log_level)
    setup_visualization(VisualizationConfig())
    input_dirs = [Path(_input_dir) for _input_dir in input_dir]
    output_dir = Path(output_dir) if output_dir is not None else None

    df_dict = defaultdict(list)
    for _input_dir in input_dirs:
        for tensorboard_dir in _input_dir.glob('**/*/tensorboard_log'):
            if read_from_eval_json:
                lab_setting = json.load(open(str(tensorboard_dir.parent / 'lab.params.json')))
                for name in LAB_ATTR_NAMES:
                    df_dict[name].append(lab_setting.get(name, None))

                eval_results = json.load(open(tensorboard_dir.parent / 'eval_results.json'))
                for metric_name in METRIC_NAMES:
                    json_metric_name = re.sub('eval/', 'eval_', metric_name)
                    df_dict[metric_name].append(eval_results.get(json_metric_name, None))

            else:
                tf_df = read_as_dataframe(str(tensorboard_dir))
                if 'step' not in tf_df.columns:
                    logger.warning('skip the results under "%s" as "step" not found in tensorboard log', str(tensorboard_dir))
                    continue

                try:
                    evaluated_metric = [metric_name for metric_name in METRIC_NAMES
                                        if metric_name in tf_df['tag'].values][0]
                except IndexError as e:
                    logger.warning('loading metric failed from "%s"', str(tensorboard_dir))
                    continue

                lab_setting = json.load(open(str(tensorboard_dir.parent / 'lab.params.json')))
                for name in LAB_ATTR_NAMES:
                    df_dict[name].append(lab_setting.get(name, None))

                final_evalation_step = tf_df[tf_df['tag'] == evaluated_metric]['step'].max()
                final_evalation_df = tf_df[tf_df['step'] == final_evalation_step]
                logger.info('loading metirc results from the final evaluation step = %d', final_evalation_step)

                for metric_name in METRIC_NAMES:
                    tag_df = final_evalation_df[final_evalation_df['tag'] == metric_name]

                    if len(tag_df) == 0:
                        df_dict[metric_name].append(None)
                    elif len(tag_df) == 1:
                        df_dict[metric_name].append(tag_df.iloc[0]['value'])
                    elif len(tag_df) >= 2:
                        first_value = tag_df['value'].iloc[0]
                        if not tag_df['value'].map(lambda val: val == first_value).all():
                            logger.warning('multiple values found for metric=%s under directory="%s"',
                                           metric_name,
                                           str(tensorboard_dir))
                            df_dict[metric_name].append(tag_df['value'].mean())
                        else:
                            df_dict[metric_name].append(first_value)

    merged_df = pd.DataFrame(df_dict)
    merged_df = merged_df.sort_values(by=list(merged_df.columns))

    pretty_df = merged_df.drop(
        columns=[
            col for col in merged_df.columns
            if col.find('accuracy') >= 0 and (col.find('D-all') < 0 or col.find('eval/') >= 0)

        ]
    )

    if output_dir is not None:
        logger.info('write into %s', str(output_dir))
        output_dir.mkdir(exist_ok=True, parents=True)

        merged_df.to_csv(output_dir / 'results.tsv', sep='\t', index=None)

        with open(output_dir / 'pretty.txt', 'w') as f:
            print(pretty_df, file=f)
        print(pretty_df)




if __name__ == '__main__':
    main()
