#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List
import json
from collections import defaultdict

import pandas as pd
from logger_setup import setup as setup_logger
import click
from machine_learning.analysis.tensorboard import read_as_dataframe


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
        'base_config_name',
        'dataset_uname',
        'generation_max_proof_steps',
        'generation_num_beams',
        'generation_input_k',
        'gradient_accumulation_steps',
        'learning_rate',
        'lm_type',
        'lora',
        'max_grad_norm',
        'max_predict_samples',
        'max_proof_steps',
        'max_source_length',
        'max_steps',
        'max_target_length',
        'max_train_samples',
        'model_name_or_path',
        'per_device_eval_batch_size',
        'per_device_train_batch_size',
        'proof_sampling',
        'sample_negative_proof',
        'seed',
        'shot',
        'source_prefix',
        'tokenizer_padding',
        'warmup_steps',
    ]

    TF_NAMES = [
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
    ]

    df_dict = defaultdict(list)
    for _input_dir in input_dirs:
        for tensorboard_dir in _input_dir.glob('**/*/tensorboard_log'):
            tf_df = read_as_dataframe(str(tensorboard_dir))
            if 'step' not in tf_df.columns:
                logger.warning('skip the results under "%s"', str(tensorboard_dir))
                continue

            lab_setting = json.load(open(str(tensorboard_dir.parent / 'lab.params.json')))

            for name in LAB_ATTR_NAMES:
                df_dict[name].append(lab_setting.get(name, None))

            max_step = tf_df['step'].max()

            max_step_df = tf_df[tf_df['step'] == max_step]
            logger.info('loading results from max_step=%d', max_step)
            for tf_name in TF_NAMES:
                tag_df = max_step_df[max_step_df['tag'] == tf_name]
                if len(tag_df) == 0:
                    df_dict[tf_name].append(None)
                elif len(tag_df) == 1:
                    df_dict[tf_name].append(tag_df.iloc[0]['value'])
                elif len(tag_df) >= 2:
                    first_value = tag_df['value'].iloc[0]
                    if not tag_df['value'].map(lambda val: val == first_value).all():
                        raise ValueError()
                    df_dict[tf_name].append(first_value)

    merged_df = pd.DataFrame(df_dict)
    print(merged_df)

    if output_dir is not None:
        out_path = output_dir / 'results.tsv'
        logger.info('write into %s', str(out_path))
        output_dir.mkdir(exist_ok=True, parents=True)
        merged_df.to_csv(out_path, sep='\t', index=None)



if __name__ == '__main__':
    main()
