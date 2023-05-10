from copy import deepcopy
from typing import Dict, List, Any, Optional
from typing import TypedDict

from pydantic import BaseModel
from FLD_prover.schema import DeductionExample
from .serialize import serialize_example


def preprocess_examples(batch_examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    keys = list(batch_examples.keys())
    n_examples = len(batch_examples[keys[0]])

    examples: List[Dict[str, Any]] = []
    for i_example in range(0, n_examples):
        examples.append({
            key: batch_examples[key][i_example]
            for key in keys
        })

    serialized_examples = [
        serialize_example(DeductionExample.parse_obj(_example)).dict()
        for _example in examples
    ]

    preprocessed_batch_examples = deepcopy(batch_examples)
    preprocessed_batch_examples.update({
        key: [serialized_examples[i_example][key]]
        for i_example in range(0, len(examples))
        for key in serialized_examples[0].keys()
    })

    return preprocessed_batch_examples
