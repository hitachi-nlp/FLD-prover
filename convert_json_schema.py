#!/usr/bin/env python
import json
import logging
from pathlib import Path

import click
from logger_setup import setup as setup_logger
from FLD_task import load_deduction


logger = logging.getLogger(__name__)


@click.command()
@click.argument('input-path', type=str)
@click.argument('output-path', type=str)
def main(input_path, output_path):
    setup_logger(do_stderr=True, level=logging.INFO)
    logger.info('input_path: %s', str(input_path))
    logger.info('output_path: %s', str(output_path))

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f_out:
        for line in open(input_path):
            instance = load_deduction(json.loads(line.rstrip()))
            f_out.write(json.dumps(instance.dict()) + '\n')


if __name__ == '__main__':
    main()
