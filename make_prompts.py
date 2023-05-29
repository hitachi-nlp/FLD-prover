#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple
import json
import random
from collections import defaultdict

from logger_setup import setup as setup_logger
import click
from FLD_task.loaders import load
from FLD_task.schema import DeductionExample, SerializedDeductionStep
from FLD_task.preprocess import serialize


logger = logging.getLogger(__name__)


def load_examples(path: Union[str, Path])\
        -> Tuple[List[DeductionExample],
                 Dict[Union[str, bool], List[DeductionExample]]]:
    examples: List[DeductionExample] = []
    label_examples: Dict[Union[str, bool], List[DeductionExample]] = defaultdict(list)
    for line in open(path):
        example = load(json.loads(line.rstrip('\n')))
        label = example.answer

        examples.append(example)
        label_examples[label].append(example)
    return examples, label_examples


@click.command()
@click.argument('train_path')
@click.argument('eval_path')
@click.argument('output_dir')
@click.option('--n-shot', type=int, default=10)
@click.option('--seed', type=int, default=0)
@click.option('--log-level', default='INFO')
def main(train_path, eval_path, n_shot, output_dir, seed, log_level):
    setup_logger(level=log_level)
    random.seed(seed)

    train_path = Path(train_path)
    eval_path = Path(eval_path)

    train_exs, train_label_exs = load_examples(train_path)

    n_shot_per_label = int(n_shot / len(train_label_exs))
    fewshot_exs: List[DeductionExample] = []
    for _, label_exs in train_label_exs.items():
        fewshot_exs.extend(random.sample(label_exs, n_shot_per_label))
    random.shuffle(fewshot_exs)

    eval_exs, _ = load_examples(eval_path)

    def dump_serial(serial: SerializedDeductionStep,
                    no_label=False) -> str:
        if no_label:
            return f'input: {serial.input}' + '\n' + 'output: '
        else:
            return f'input: {serial.input}' + '\n' + f'output: {serial.next_step}'

    def join_serial_dumps(serial_dumps: List[str]) -> str:
        return '\n\n'.join(serial_dumps)

    fewshot_serials = [serialize(fewshot_ex, stepwise=False)
                       for fewshot_ex in fewshot_exs]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'prompts.jsonl', 'w') as f_jsonl,\
            open(output_dir / 'prompts.txt', 'w') as f_txt:

        for eval_ex in eval_exs:
            f_txt.write('\n\n============ prompt ============\n')

            eval_serial = serialize(eval_ex, stepwise=False)
            prompt = join_serial_dumps(
                [dump_serial(serial) for serial in fewshot_serials]
                + [dump_serial(eval_serial, no_label=True)]
            )
            instance = {
                'example': eval_ex.dict(),
                'fewshot_examples': [fewshot_ex.dict()
                                     for fewshot_ex in fewshot_exs],
                'serial': eval_serial.dict(),
                'fewshot_serials': [fewshot_serial.dict()
                                    for fewshot_serial in fewshot_serials],
                'prompt': prompt,
            }

            f_txt.write(prompt)
            f_jsonl.write(json.dumps(instance) + '\n')


if __name__ == '__main__':
    main()
