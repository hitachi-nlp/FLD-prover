from typing import Dict
from copy import deepcopy
import re
import random

from FLD_prover.stance_indication import add_stance_markers, StanceMarker
from FLD_prover.schema import DeductionExample, SerializedDeductionStep


def serialize_example(
    example: DeductionExample,
    stepwise=True,
) -> SerializedDeductionStep:
    """

    examples)
    hypothesis = 'this is the hypothesis'
    context = 'sent1: this is sentence1 sent2: this is sentence2 sent3: this is sentence3'
    proof = 'sent1 & sent2 -> int1: the conclusion of sentence1 and sentence2; sent3 & int1 -> int2: the conclusion of int1 and sent3;'
    """

    hypothesis = example.hypothesis
    context = example.context
    stance_marker = _get_stance_marker(example.proof_stance)

    if len(example.proofs) == 0:
        partial_proof = ''
        next_step = add_stance_markers('', [stance_marker])
    else:
        proof = example.proofs[0]

        if stepwise:
            proof_steps = re.split('; *', re.sub('; *$', '', proof))
            n_next_step = random.randint(0, len(proof_steps) - 1)

            partial_proof_steps = proof_steps[:n_next_step]
            if len(partial_proof_steps) == 0:
                partial_proof = ''
            else:
                partial_proof = '; '.join(partial_proof_steps) + ';'

            next_step = proof_steps[n_next_step] + ';'
            if n_next_step == len(proof_steps) - 1:
                next_step = add_stance_markers(next_step, [stance_marker])

        else:
            partial_proof = ''
            next_step = add_stance_markers(proof, [stance_marker])

    input_text = f"$hypothesis$ = {hypothesis} ; $context$ = {context} ; $proof$ = {partial_proof}"

    serialized_example = SerializedDeductionStep(
        input=input_text,
        next_step=next_step,
    )
    return serialized_example


def _get_stance_marker(example_proof_stance: str) -> StanceMarker:
    if example_proof_stance == 'PROOF':
        return StanceMarker.PROVED
    elif example_proof_stance == 'DISPROOF':
        return StanceMarker.DISPROVED
    elif example_proof_stance == 'UNKNOWN':
        return StanceMarker.UNKNOWN
    else:
        raise ValueError()
