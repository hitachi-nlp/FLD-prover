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
import kern_profiler


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
@click.argument('eval_path')
@click.argument('output_dir')
@click.option('--train-path')
@click.option('--prompt-type',
              type=click.Choice(['in_context_examples', 'in_context_examples.COT']),
              default='in_context_examples')
@click.option('--n-shot', type=int, default=10)
@click.option('--seed', type=int, default=0)
@click.option('--log-level', default='INFO')
def main(eval_path, output_dir, train_path, n_shot, prompt_type, seed, log_level):
    setup_logger(level=log_level)
    random.seed(seed)

    eval_path = Path(eval_path)
    eval_exs, eval_label_exs = load_examples(eval_path)

    if train_path is None:
        logger.info('train_path is not specified. Make few-shot examples from eval dataset')
        train_exs, train_label_exs = eval_exs, eval_label_exs
    else:
        train_path = Path(train_path)
        train_exs, train_label_exs = load_examples(train_path)

    n_shot_per_label = int(n_shot / len(train_label_exs))
    fewshot_exs: List[DeductionExample] = []
    for _, label_exs in train_label_exs.items():
        fewshot_exs.extend(random.sample(label_exs, n_shot_per_label))
    random.shuffle(fewshot_exs)

    # remove few-shot examples from eval dataset
    fewshot_exs_set = set(id(ex) for ex in fewshot_exs)
    eval_exs = [ex for ex in eval_exs if id(ex) not in fewshot_exs_set]
    eval_label_exs = {
        label: [ex for ex in label_exs if id(ex) not in fewshot_exs_set]
        for label, label_exs in eval_label_exs.items()
    }

    def dump_serial(serial: SerializedDeductionStep,
                    no_label=False) -> str:
        input_text = f'**** input ****\n{serial.input}'
        if no_label:
            output_text = '**** output ****\n'
        else:
            output_text = f'**** output ****\n{serial.next_step}'
        return '\n\n'.join([input_text, output_text])

    def join_serial_dumps(serial_dumps: List[str]) -> str:
        return '\n\n\n'.join(serial_dumps)

    def _serialize(ex: DeductionExample) -> SerializedDeductionStep:
        return serialize(ex, stepwise=False, newlines=True, proof_indicator=False)

    fewshot_serials = [_serialize(fewshot_ex)
                       for fewshot_ex in fewshot_exs]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'prompts.jsonl', 'w') as f_jsonl,\
            open(output_dir / 'prompts.txt', 'w') as f_txt:

        for eval_ex in eval_exs:

            eval_serial = _serialize(eval_ex)

            prompt = ''
            if prompt_type == 'in_context_examples.COT':
                prompt += '==== 1. First, we show some examples of deductive reasoning tasks as follows. In each example, the sentences after the "context" show a set of facts. Based on these facts, you have to either prove the "hypothesis", disprove it, or declare it as unknown if the facts are insufficient. You have to write the step-by-step thought after the "output".\n\n'
            prompt += join_serial_dumps([dump_serial(serial) for serial in fewshot_serials]) + '\n\n'
            if prompt_type == 'in_context_examples.COT':
                prompt += '==== 2. Now, solve the following example. Write the step-by-step thought after the "output" using the same format demonstrated in the above examples.\n\n'
            prompt += dump_serial(eval_serial, no_label=True)

            instance = {
                'example': eval_ex.dict(),
                'fewshot_examples': [fewshot_ex.dict()
                                     for fewshot_ex in fewshot_exs],
                'serial': eval_serial.dict(),
                'fewshot_serials': [fewshot_serial.dict()
                                    for fewshot_serial in fewshot_serials],
                'prompt': prompt,
            }

            f_jsonl.write(json.dumps(instance) + '\n')

            f_txt.write('\n\n\n======================== example ========================\n')
            f_txt.write('\n\n\n------------ prompt ------------\n\n')
            f_txt.write(prompt)
            f_txt.write('\n\n\n------------ gold ------------\n\n')
            f_txt.write(eval_serial.next_step)


if __name__ == '__main__':
    main()
