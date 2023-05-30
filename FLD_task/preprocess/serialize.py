from typing import Dict, Tuple, Optional, List, Any, Set
import re
import random
from copy import deepcopy
import re
import random

from stance_indication import add_stance_markers, StanceMarker
from FLD_task.schema import DeductionExample, SerializedDeductionStep


def serialize(
    example: DeductionExample,
    stepwise=True,
    sample_negative_proof=False,
    newlines=False,
    proof_indicator=True,
) -> SerializedDeductionStep:
    """

    examples)
    hypothesis = 'this is the hypothesis'
    context = 'sent1: this is sentence1 sent2: this is sentence2 sent3: this is sentence3'
    proof = 'sent1 & sent2 -> int1: the conclusion of sentence1 and sentence2; sent3 & int1 -> int2: the conclusion of int1 and sent3;'
    """
    if len(example.proofs) == 0:
        partial_proof = ''
        next_step = ''
        is_final_step = True

    else:
        assert(len(example.proofs) == 1)
        proof = example.proofs[0]

        negative_proof: Optional[str] = None
        if example.negative_proofs is not None and len(example.negative_proofs) > 0:
            assert len(example.negative_proofs) == 1
            negative_proof = example.negative_proofs[0]

        if stepwise:
            sampled_proof = _sample_subproof(proof, at_least_one_step=True)
            is_final_step = _split_into_steps(sampled_proof)[-1] == _split_into_steps(proof)[-1]

            if sample_negative_proof and negative_proof is not None:
                sampled_negative_proof = _sample_subproof(negative_proof, at_least_one_step=False)
                if sampled_negative_proof is not None:
                    spliced_proof = _splice_negative_proof(sampled_proof, sampled_negative_proof)
                else:
                    spliced_proof = sampled_proof
            else:
                spliced_proof = sampled_proof

            steps = _split_into_steps(spliced_proof)
            partial_proof_steps, next_step = steps[:-1], steps[-1]
            partial_proof = _merge_steps(partial_proof_steps)
        else:
            if sample_negative_proof and negative_proof is not None:
                spliced_proof = _splice_negative_proof(proof, negative_proof, negative_then_positive=True)
            else:
                spliced_proof = proof

            partial_proof = ''
            next_step = spliced_proof
            is_final_step = True

    if is_final_step:
        next_step = add_stance_markers(next_step,
                                       [_get_stance_marker(example.proof_stance)])

    input_text = ' ; '.join([
        f'$hypothesis$ = {example.hypothesis}',
        f'$context$ = {example.context}',
    ])
    if proof_indicator:
        input_text = ' ; '.join([input_text, f'$proof$ = {partial_proof}'])
    else:
        if stepwise:
            raise ValueError('Can not add partial proof because proof_indicator=False is specified')

    if newlines:
        input_text = re.sub(' *; *', ';\n', input_text)
        input_text = re.sub('sent([0-9]*)', r'\nsent\g<1>', input_text).lstrip('\n')
        next_step = re.sub(' *; *', ';\n', next_step)

    serialized_example = SerializedDeductionStep(
        input=input_text,
        next_step=next_step,
    )
    return serialized_example


def _sample_subproof(proof: str, at_least_one_step=True) -> Optional[str]:
    steps = _split_into_steps(proof)
    upto = random.randint(1 if at_least_one_step else 0, len(steps) + 1)
    if upto == 0:
        return None
    else:
        return _merge_steps(steps[:upto])


def _splice_negative_proof(proof: str,
                           negative_proof: str,
                           negative_then_positive=False) -> str:
    negative_proof_offset = _rename_ints_with_offset(negative_proof, 10000)

    steps = _split_into_steps(proof)
    negative_steps = _split_into_steps(negative_proof_offset)

    if negative_then_positive:
        spliced_steps = negative_steps + steps
    else:
        spliced_steps = _random_splice(steps, negative_steps)

    # make sure that the last step is from positive proof
    last_positive_step = [step for step in spliced_steps
                          if step in steps][-1]
    spliced_steps.remove(last_positive_step)
    spliced_steps.append(last_positive_step)

    spliced_proof = _merge_steps(spliced_steps)
    return _rename_ints_ascending(spliced_proof)


def _random_splice(these: List[Any], those: List[Any]) -> List[Any]:
    if len(these) == 0 and len(those) == 0:
        return []
    elif len(these) == 0:
        return those
    elif len(those) == 0:
        return these

    spliced: List[Any] = []
    thise_ratio = len(these) / (len(these) + len(those))
    i_these = 0
    i_those = 0
    for _ in range(len(these) + len(those)):
        if (i_these < len(these) and random.random() < thise_ratio)\
                or i_those >= len(those):
            spliced.append(these[i_these])
            i_these += 1
        else:
            spliced.append(those[i_those])
            i_those += 1

    return spliced


def _split_into_steps(proof: str) -> List[str]:
    return [
        step + ';'
        for step in re.split('; *', re.sub('; *$', '', proof))
    ]


def _merge_steps(steps: List[str]) -> str:
    return ' '.join(steps)


def _rename_ints_with_offset(proof: str, offset: int) -> str:
    proof_offset = proof
    for int_id in _get_int_ids_by_appearance_order(proof):
        int_no = int(int_id[3:])
        int_no_after = (int_no - 1) + offset
        proof_offset = _replace_int_id(int_id, f'INT{str(int_no_after)}', proof_offset)
    return re.sub('INT([0-9]+)', 'int\g<1>', proof_offset)


def _rename_ints_ascending(proof: str) -> str:
    proof_ascending = proof
    lowest_int_no = 1
    for int_id in _get_int_ids_by_appearance_order(proof):
        proof_ascending = _replace_int_id(int_id, f'INT{str(lowest_int_no)}', proof_ascending)
        lowest_int_no += 1
    return re.sub('INT([0-9]+)', 'int\g<1>', proof_ascending)


def _replace_int_id(src_id: str, dst_id: str, proof: str) -> str:
    # avoid replacing prefix such as 'int1' in 'int14'
    return re.sub(src_id + '([^0-9])', f'{dst_id}\g<1>', proof)


def _get_int_ids_by_appearance_order(proof: str) -> List[str]:
    int_ids: List[str] = []
    done_ids: Set[str] = set([])
    for int_id in re.findall('int[0-9]+', proof):
        if int_id in done_ids:
            continue
        int_ids.append(int_id)
        done_ids.add(int_id)
    return int_ids


def _get_lowest_int_no(proof: str) -> int:
    all_int_ids = re.findall('int[0-9]+', proof)
    if len(all_int_ids) > 0:
        return min(int(int_id[3:]) for int_id in all_int_ids)
    else:
        return 1


def _get_stance_marker(example_proof_stance: str) -> StanceMarker:
    if example_proof_stance == 'PROVED':
        return StanceMarker.PROVED
    elif example_proof_stance == 'DISPROVED':
        return StanceMarker.DISPROVED
    elif example_proof_stance == 'UNKNOWN':
        return StanceMarker.UNKNOWN
    else:
        raise ValueError(f'Unknown stance label {example_proof_stance}')
