#!/usr/bin/env python
import logging
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional
import json
import random
from collections import defaultdict

from logger_setup import setup as setup_logger
from datasets import load_dataset
import click
from FLD_task import (
    load_deduction,
    Deduction,
    SerializedDeduction,
    serialize,
)

# import line_profiling

logger = logging.getLogger(__name__)


# @profile
def _load_dataset(dataset_name: Optional[str] = None,
                  dataset_config_name: Optional[str] = None,
                  train_file: Optional[str] = None,
                  test_file: Optional[str] = None):
    if dataset_name is not None:
        raw_datasets = load_dataset(dataset_name, dataset_config_name)
    else:
        data_files = {}
        if train_file is not None:
            data_files["train"] = train_file
        if test_file is not None:
            data_files["test"] = test_file

        raw_datasets = load_dataset('json', data_files=data_files)

    return raw_datasets


# @profile
def load_examples(dataset_name: Optional[str] = None,
                  dataset_config_name: Optional[str] = None,
                  train_file: Optional[str] = None,
                  test_file: Optional[str] = None)\
        -> Dict[str, Dict[Union[str, bool], List[Deduction]]]:

    dataset = _load_dataset(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        train_file=train_file,
        test_file=test_file,
    )

    examples = defaultdict(lambda: defaultdict(list))
    for split, dataset in dataset.items():
        for dataset_example in dataset:
            example = load_deduction(dataset_example)
            label = example.world_assump_label
            examples[split][label].append(example)

    return examples


_INTROS = {

    'v0': '==== 1. First, we show some examples of deductive reasoning tasks as follows. In each example, the sentences after the "facts" show a set of facts. Based on these facts, you have to either prove the "hypothesis", disprove it, or declare it as unknown if the facts are insufficient. You have to write the step-by-step thought after the "output".',

    'v1': """\
******** First, we show some examples of deductive reasoning tasks below. ********

An example consists of an input part and an output part, shown after "---- input ----" and "---- output ----," respectively.

In the input part, we have a set of facts shown after "$facts$." Based on these facts, we want to verify a hypothesis written after "$hypothesis."

The output part shows the step-by-step thought to verify the hypothesis.

Each line of the output part shows a fine-grained reasoning step. In each step, the left side of the arrow "->"  shows the set of premises to be used, such as "fact2 & int3" if the set includes the fact numbered as two and the intermediate conclusion numbered as three. The right side of the arrow "->" shows the conclusion that logically follows from the premises. Note that this conclusion should be new, i.e., not match any of the facts or the previously derived conclusions.

After these steps, we conclude either the hypothesis can be proved (__PROVED__), disproved (__DISPROVED__), or neither (__UNKNOWN__) because the facts are insufficient.
""",


    'v2': """\
******** First, we show some examples of deductive reasoning tasks below. ********

An example consists of an input part and an output part, shown after "---- input ----" and "---- output ----," respectively.

In the input part, we have a set of facts shown after "$facts$." Based on these facts, we want to verify a hypothesis written after "$hypothesis."

The output part shows the step-by-step thought to verify the hypothesis.

Each line of the output part shows a fine-grained reasoning step. In each step, the left side of the arrow "->"  shows the set of premises to be used, such as "fact2 & int3" if the set includes the fact numbered as two and the intermediate conclusion numbered as three. The right side of the arrow "->" shows the conclusion that logically follows from the premises. Note that this conclusion should be new, i.e., not match any of the facts or the previously derived conclusions. For example, the following is not allowed: "fact3 -> int2: this is a sentence" where the content of "fact3" is "this is a sentence".

After these steps, we conclude either the hypothesis can be proved (__PROVED__), disproved (__DISPROVED__), or neither (__UNKNOWN__) because the facts are insufficient.
"""


}

_QUESTIONS = {

    'v0': '==== 2. Now, solve the following example. Write the step-by-step thought after the "output" using the same format demonstrated in the above examples.',

    'v1': '******** 2. Now, solve the following example, i.e., write a step-by-step thought to verify the hypothesis after the "output", using exactly the same format demonstrated in the above examples.',
}


def _make_intro(prompt_type: str) -> Optional[str]:
    if prompt_type == 'ICL-COT':
        return _INTROS['v0']
    elif prompt_type == 'ICL-COT.v1':
        return _INTROS['v1']
    elif prompt_type == 'ICL-COT.v2':
        return _INTROS['v2']

    else:
        return None


def _make_question(prompt_type: str) -> Optional[str]:
    if prompt_type == 'ICL-COT':
        return _QUESTIONS['v0']
    elif prompt_type == 'ICL-COT.v1':
        return _QUESTIONS['v1']
    elif prompt_type == 'ICL-COT.v2':
        return _QUESTIONS['v1']
    else:
        return None


def dump_serial(serial: SerializedDeduction,
                no_label=False) -> str:

    serial_input_text = serial.prompt
    if serial.partial_proof is not None:
        serial_input_text += serial.partial_proof

    example_text = '============ an example ============'
    input_text = f'---- input ----\n{serial_input_text}'
    if no_label:
        output_text = '---- output ----\n'
    else:
        output_text = f'---- output ----\n{serial.next_proof_step}'
    return '\n\n'.join([example_text, input_text, output_text])


def join_serial_dumps(serial_dumps: List[str]) -> str:
    return '\n\n\n'.join(serial_dumps)


def _serialize(ex: Deduction) -> SerializedDeduction:
    return serialize(ex, stepwise=False, newlines=True, proof_indicator=False)


# @profile
def _main(output_dir,
          dataset_name,
          dataset_config_name,
          train_file,
          test_file,
          n_shot,
          prompt_type,
          seed,
          icl_max_proof_by_contradiction_per_label,
          log_level):
    setup_logger(level=log_level)
    random.seed(seed)

    examples = load_examples(dataset_name=dataset_name,
                             dataset_config_name=dataset_config_name,
                             train_file=train_file,
                             test_file=test_file)

    train_labeled_examples = examples['train']
    test_labeled_examples = examples['test']

    n_shot_per_label = int(n_shot / len(train_labeled_examples) + 1)
    ICL_labeled_examples: Dict[str, List[Deduction]] = defaultdict(list)
    for label, label_exs in train_labeled_examples.items():
        ICL_labeled_examples[label].extend(random.sample(label_exs, n_shot_per_label))

    # add examples following the label order: labelA, labelB, labelC, labelA, ...
    ICL_examples: List[Deduction] = []
    proof_by_contradicition_count: Dict[str, int] = defaultdict(int)
    tot = 0
    for i_shot in range(n_shot_per_label):
        if tot >= n_shot:
            break
        for label, label_exs in sorted(ICL_labeled_examples.items()):
            if tot >= n_shot:
                break
            example = label_exs[i_shot]

            is_proof_by_contradiction = len(example.proofs) > 0 and example.proofs[0].find('contradiction') >= 0
            if is_proof_by_contradiction:
                if icl_max_proof_by_contradiction_per_label is not None\
                        and proof_by_contradicition_count[label] >= icl_max_proof_by_contradiction_per_label:
                    continue
                proof_by_contradicition_count[label] += 1

            ICL_examples.append(example)
            tot += 1

    ICL_serials = [_serialize(example)
                   for example in ICL_examples]

    all_test_examples = [example
                         for _, _examples in test_labeled_examples.items()
                         for example in _examples]
    random.shuffle(all_test_examples)
    ICL_example_ids = set(id(ex) for ex in ICL_examples)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'prompts.jsonl', 'w') as f_jsonl, \
            open(output_dir / 'prompts.txt', 'w') as f_txt:
        for test_example in all_test_examples:
            if id(test_example) in ICL_example_ids:
                # do not include few-shot examples in the test set
                continue

            test_serial = _serialize(test_example)

            prompt = ''

            intro = _make_intro(prompt_type)
            if intro is not None:
                prompt += intro + '\n\n\n'

            prompt += join_serial_dumps([dump_serial(serial) for serial in ICL_serials]) + '\n\n'

            question = _make_question(prompt_type)
            if question is not None:
                prompt += '\n\n\n' + question + '\n\n\n'

            prompt += dump_serial(test_serial, no_label=True)

            instance = {
                'example': test_example.dict(),
                'fewshot_examples': [fewshot_example.dict()
                                     for fewshot_example in ICL_examples],

                'gold_proof': test_serial.proof,

                'serial': test_serial.dict(),
                'fewshot_serials': [fewshot_serial.dict()
                                    for fewshot_serial in ICL_serials],
                'prompt': prompt,

                'stats': {
                    'prompt': {
                        'num_words': len(prompt.split()),
                    },
                },
            }

            f_jsonl.write(json.dumps(instance) + '\n')

            f_txt.write(
                '\n\n\n************************************************ [meta comment] this is an instance ************************************************\n')
            f_txt.write('\n\n\n************************ [meta comment] this is the prompt ************************\n\n')
            f_txt.write(prompt)
            f_txt.write('\n\n\n************************ [meta comment] this is the gold output ************************\n\n')
            f_txt.write(test_serial.next_proof_step)


@click.command()
@click.option('--output-dir', default=None)
@click.option('--dataset-name', default=None)
@click.option('--dataset-config-name', default='default')
@click.option('--train-file', default=None)
@click.option('--test-file', default=None)
@click.option('--prompt-type',
              type=click.Choice([
                  'ICL',
                  'ICL-COT',
                  'ICL-COT.v1',
                  'ICL-COT.v2',

                  'ICL.jpn',
                  'ICL-COT.jpn',
                  'ICL-COT.v1.jpn',
                  'ICL-COT.v2.jpn',
              ]),
              default='ICL-COT')
@click.option('--n-shot', type=int, default=10)
@click.option('--seed', type=int, default=0)
@click.option('--icl-max-proof-by-contradiction-per-label', type=int, default=None)
@click.option('--log-level', default='INFO')
def main(output_dir,
         dataset_name,
         dataset_config_name,
         train_file,
         test_file,
         n_shot,
         prompt_type,
         seed,
         icl_max_proof_by_contradiction_per_label,
         log_level):
    _main(
        output_dir,
        dataset_name,
        dataset_config_name,
        train_file,
        test_file,
        n_shot,
        prompt_type,
        seed,
        icl_max_proof_by_contradiction_per_label,
        log_level
    )


if __name__ == '__main__':
    main()
