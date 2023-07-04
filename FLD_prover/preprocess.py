from copy import deepcopy
from typing import Dict, List, Any, Optional
from typing import TypedDict

from pydantic import BaseModel
from FLD_task import Deduction, load_deduction, serialize


def preprocess_examples_train(batch_examples: Dict[str, List[Any]],
                              stepwise=False,
                              sample_negative_proof=False) -> Dict[str, List[Any]]:
    return _preprocess_examples(batch_examples, stepwise, sample_negative_proof=sample_negative_proof)


def preprocess_examples_eval(batch_examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    return _preprocess_examples(batch_examples, False)


def _preprocess_examples(batch_examples: Dict[str, List[Any]],
                         stepwise: bool,
                         sample_negative_proof=False) -> Dict[str, List[Any]]:
    keys = list(batch_examples.keys())
    n_examples = len(batch_examples[keys[0]])

    examples: List[Deduction] = []
    for i_example in range(0, n_examples):
        examples.append(
            load_deduction({
                key: batch_examples[key][i_example]
                for key in keys
            })
        )

    serialized_examples = [
        serialize(_example,
                  stepwise=stepwise,
                  sample_negative_proof=sample_negative_proof)
        for _example in examples
    ]

    preprocessed_batch_examples = deepcopy(batch_examples)
    preprocessed_batch_examples['context'] = [
        serialized_examples[i_example].input
        for i_example in range(0, len(examples))
    ]
    preprocessed_batch_examples['next_step'] = [
        serialized_examples[i_example].next_step
        for i_example in range(0, len(examples))
    ]

    for serial in serialized_examples:
        if len(serial.gold_proofs) >= 2:
            raise NotImplementedError()
    preprocessed_batch_examples['gold_proof'] = [
        serialized_examples[i_example].gold_proofs[0]
        for i_example in range(0, len(examples))
    ]

    return preprocessed_batch_examples
