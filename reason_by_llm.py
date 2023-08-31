#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv()  # must be here
import os
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional
import json

from logger_setup import setup as setup_logger
import click
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import openai
from tqdm import tqdm


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

    model_service, _model_name = model_name.split('.', 1)

    if model_service == 'hf':
        hf = HuggingFacePipeline.from_model_id(
            model_id=model_name.lstrip('hf.'),
            task='text-generation',
            # pipeline_kwargs={"max_new_tokens": 10},
        )

        def get_reply(prompt: str) -> Optional[str]:
            results = hf.generate([prompt])
            return results.generations[0][0].text

    elif model_service == 'openai':

        if api_key is None:
            if _model_name.startswith('openai.gpt-4'):
                api_key = os.environ.get('OPENAI_API_KEY_GPT4', None)
            else:
                api_key = os.environ.get('OPENAI_API_KEY', None)
        if api_key is None:
            raise ValueError()

        if _model_name == 'text-davinci-003':
            llm = OpenAI(model_name=_model_name,
                         openai_api_key=api_key)

            def get_reply(prompt: str) -> str:
                return llm(prompt)

        else:
            chat_model = ChatOpenAI(model_name=_model_name,
                                    openai_api_key=api_key)

            def get_reply(prompt: str) -> Optional[str]:
                try:
                    return chat_model([HumanMessage(content=prompt)]).content
                except openai.error.InvalidRequestError as e:
                    logger.critical(e)
                    return None

    with open(output_path, 'w') as f_out:
        for i_sample, line in tqdm(enumerate(open(input_path))):
            if i_sample >= max_samples:
                break
            sample = json.loads(line.strip('\n'))
            prompt = sample['prompt']

            logger.info('-- running the LLM on a example ... --')
            logger.info('prompt # words: %d', len(prompt.split()))

            reply = get_reply(prompt)

            if reply is not None:
                logger.info('reply # words: %d', len(reply.split()))

            sample['reply'] = reply
            f_out.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    main()
