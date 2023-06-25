#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional
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


_INTROS = {

    'v0': '==== 1. First, we show some examples of deductive reasoning tasks as follows. In each example, the sentences after the "context" show a set of facts. Based on these facts, you have to either prove the "hypothesis", disprove it, or declare it as unknown if the facts are insufficient. You have to write the step-by-step thought after the "output".',

    'v1': """\
******** First, we show some examples of deductive reasoning tasks below. ********

An example consists of an input part and an output part, shown after "---- input ----" and "---- output ----," respectively.

In the input part, we have a set of facts shown after "$context$." Based on these facts, we want to verify a hypothesis written after "$hypothesis."

The output part shows the step-by-step thought to verify the hypothesis.

Each line of the output part shows a fine-grained reasoning step. In each step, the left side of the arrow "->"  shows the set of premises to be used, such as "sent2 & int3" if the set includes the fact numbered as two and the intermediate conclusion numbered as three. The right side of the arrow "->" shows the conclusion that logically follows from the premises. Note that this conclusion should be new, i.e., not match any of the facts or the previously derived conclusions.

After these steps, we conclude either the hypothesis can be proved (__PROVED__), disproved (__DISPROVED__), or neither (__UNKNOWN__) because the facts are insufficient.
""",


    'v2': """\
******** First, we show some examples of deductive reasoning tasks below. ********

An example consists of an input part and an output part, shown after "---- input ----" and "---- output ----," respectively.

In the input part, we have a set of facts shown after "$context$." Based on these facts, we want to verify a hypothesis written after "$hypothesis."

The output part shows the step-by-step thought to verify the hypothesis.

Each line of the output part shows a fine-grained reasoning step. In each step, the left side of the arrow "->"  shows the set of premises to be used, such as "sent2 & int3" if the set includes the fact numbered as two and the intermediate conclusion numbered as three. The right side of the arrow "->" shows the conclusion that logically follows from the premises. Note that this conclusion should be new, i.e., not match any of the facts or the previously derived conclusions. For example, the following is not allowed: "sent3 -> int2: this is a sentence" where the content of "sent3" is "this is a sentence".

After these steps, we conclude either the hypothesis can be proved (__PROVED__), disproved (__DISPROVED__), or neither (__UNKNOWN__) because the facts are insufficient.
"""


}

_QUESTIONS = {

    'v0': '==== 2. Now, solve the following example. Write the step-by-step thought after the "output" using the same format demonstrated in the above examples.',

    'v1': '******** 2. Now, solve the following example, i.e., write a step-by-step thought to verify the hypothesis after the "output", using exactly the same format demonstrated in the above examples.'
}


def _make_intro(prompt_type: str) -> Optional[str]:
    if prompt_type == 'in_context_examples.COT':
        return _INTROS['v0']
    elif prompt_type == 'in_context_examples.COT.v1':
        return _INTROS['v1']
    elif prompt_type == 'in_context_examples.COT.v2':
        return _INTROS['v2']

    else:
        return None


def _make_question(prompt_type: str) -> Optional[str]:
    if prompt_type == 'in_context_examples.COT':
        return _QUESTIONS['v0']
    elif prompt_type == 'in_context_examples.COT.v1':
        return _QUESTIONS['v1']
    elif prompt_type == 'in_context_examples.COT.v2':
        return _QUESTIONS['v1']
    else:
        return None


@click.command()
@click.argument('eval_path')
@click.argument('output_dir')
@click.option('--train-path')
@click.option('--prompt-type',
              type=click.Choice([
                  'in_context_examples',
                  'in_context_examples.COT',
                  'in_context_examples.COT.v1',
                  'in_context_examples.COT.v2',
              ]),
              default='in_context_examples.COT')
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
    fweshot_label_exs: Dict[Union[bool, str], List[DeductionExample]] = defaultdict(list)
    for label, label_exs in train_label_exs.items():
        fweshot_label_exs[label].extend(random.sample(label_exs, n_shot_per_label))

    # add examples following the label order: labelA, labelB, labelC, labelA, ...
    fewshot_exs: List[DeductionExample] = []
    for i_ex in range(n_shot_per_label):
        for _, label_exs in sorted(fweshot_label_exs.items()):
            fewshot_exs.append(label_exs[i_ex])

    # remove few-shot examples from eval dataset
    fewshot_exs_set = set(id(ex) for ex in fewshot_exs)
    eval_exs = [ex for ex in eval_exs if id(ex) not in fewshot_exs_set]
    eval_label_exs = {
        label: [ex for ex in label_exs if id(ex) not in fewshot_exs_set]
        for label, label_exs in eval_label_exs.items()
    }

    def dump_serial(serial: SerializedDeductionStep,
                    no_label=False) -> str:
        example_text = '============ an example ============'
        input_text = f'---- input ----\n{serial.input}'
        if no_label:
            output_text = '---- output ----\n'
        else:
            output_text = f'---- output ----\n{serial.next_step}'
        return '\n\n'.join([example_text, input_text, output_text])

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

        for i_ex, eval_ex in enumerate(eval_exs):
            # if i_ex not in [7, 8]:
            #     continue

            eval_serial = _serialize(eval_ex)

            prompt = ''

            intro = _make_intro(prompt_type)
            if intro is not None:
                prompt += intro + '\n\n\n'

            prompt += join_serial_dumps([dump_serial(serial) for serial in fewshot_serials]) + '\n\n'

            question = _make_question(prompt_type)
            if question is not None:
                prompt += '\n\n\n' + question + '\n\n\n'

            prompt += dump_serial(eval_serial, no_label=True)

            if len(eval_serial.gold_proofs) > 1:
                raise ValueError()
            instance = {
                'example': eval_ex.dict(),
                'fewshot_examples': [fewshot_ex.dict()
                                     for fewshot_ex in fewshot_exs],

                'gold_proofs': eval_serial.gold_proofs,

                'serial': eval_serial.dict(),
                'fewshot_serials': [fewshot_serial.dict()
                                    for fewshot_serial in fewshot_serials],
                'prompt': prompt,

                'stats': {
                    'prompt': {
                        'num_words': len(prompt.split()),
                    },
                },
            }

            f_jsonl.write(json.dumps(instance) + '\n')

            f_txt.write('\n\n\n************************************************ [meta comment] this is an instance ************************************************\n')
            f_txt.write('\n\n\n************************ [meta comment] this is the prompt ************************\n\n')
            f_txt.write(prompt)
            f_txt.write('\n\n\n************************ [meta comment] this is the gold output ************************\n\n')
            f_txt.write(eval_serial.next_step)


if __name__ == '__main__':
    main()
