#!/usr/bin/env python
import json
from pathlib import Path
import logging
from pathlib import Path

import click

from logger_setup import setup as setup_logger


logger = logging.getLogger(__name__)


@click.command()
@click.argument('input-path')
@click.argument('output-path')
def main(input_path, output_path):
    setup_logger(do_stderr=True, level=logging.INFO)
    logger.info('input_path: %s', str(input_path))
    logger.info('output_path: %s', str(output_path))

    schema_conversion = {
        'answer': str,
        'negative_answer': str,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f_out:
        for line in open(input_path):
            instance = json.loads(line.rstrip())
            for key, type_ in schema_conversion.items():
                instance[key] = type_(instance[key])
            f_out.write(json.dumps(instance) + '\n')


if __name__ == '__main__':
    main()
