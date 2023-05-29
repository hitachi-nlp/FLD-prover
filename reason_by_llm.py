#!/usr/bin/env python
import os
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple
import json

from logger_setup import setup as setup_logger
import click
from langchain.llms import OpenAI


logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_path')
@click.argument('output_path')
@click.option('--model-name', default='text-davinci-003',
              help='choose from https://platform.openai.com/docs/models/model-endpoint-compatibility')
@click.option('--api-key', default=None)
@click.option('--max-samples', type=int, default=None)
@click.option('--log-level', default='INFO')
def main(input_path, model_name, output_path, api_key, max_samples, log_level):
    setup_logger(level=log_level)
    input_path = Path(input_path)
    output_path = Path(output_path)

    api_key = api_key or os.environ.get('OPENAI_API_KEY', None)
    if api_key is None:
        raise ValueError()

    llm = OpenAI(model_name=model_name,
                 openai_api_key=api_key)

    with open(output_path, 'w') as f_out:
        for i_sample, line in enumerate(open(input_path)):
            if i_sample >= max_samples:
                break
            sample = json.loads(line.strip('\n'))
            prompt = sample['prompt']
            completion = llm(prompt)
            sample['completion'] = completion
            f_out.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    main()
