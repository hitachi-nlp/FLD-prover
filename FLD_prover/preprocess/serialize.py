from typing import Dict
from copy import deepcopy

from FLD_prover.stance_indication import add_stance_markers, StanceMarker
from FLD_prover.schema import DeductionExample, SerializedDeductionExample


def serialize_example(example: DeductionExample) -> SerializedDeductionExample:

    hypothesis = example.hypothesis
    context = example.context

    if example.proof_stance == 'PROOF':
        proof_text = add_stance_markers(example.proofs[0], [StanceMarker.PROVED])
    elif example.proof_stance == 'DISPROOF':
        proof_text = add_stance_markers(example.proofs[0], [StanceMarker.DISPROVED])
    elif example.proof_stance == 'UNKNOWN':
        proof_text = add_stance_markers('', [StanceMarker.UNKNOWN])
    else:
        raise ValueError()

    partial_proof = ''
    input_text = f"$hypothesis$ = {hypothesis} ; $context$ = {context} ; $proof$ = {partial_proof}"

    serialized_example = SerializedDeductionExample(
        input_text=input_text,
        output_text=proof_text,
    )
    return serialized_example
