from typing import Dict
from copy import deepcopy

from FLD_prover.schema import DeductionExample, SerializedDeductionExample


def serialize_example(example: DeductionExample) -> SerializedDeductionExample:
    serialized_example = SerializedDeductionExample(
        input_text=example.context,
        output_text=example.hypothesis,
    )
    return serialized_example
