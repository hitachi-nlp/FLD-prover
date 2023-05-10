import re
from typing import List, Optional, Set
from enum import Enum
from copy import deepcopy
import logging
from common import Example, I_DONT_THINK, Answer, DatasetType
import random
from proof_common import (
    add_final_reference,
    extract_idents,
    extract_ident,
    INT_IDENT,
    ASSUMP_IDENT,
    extract_steps,
    extract_premise_concl,
)

logger = logging.getLogger(__name__)


class StanceIndicationMethod(Enum):
    NEGATED_HYPOTHESIS_BY_I_DONT_THINK = 'NEGATED_HYPOTHESIS_BY_I_DONT_THINK'
    STANCE_MARKER_IN_PROOF = 'STANCE_MARKER_IN_PROOF'


class StanceMarker(Enum):
    PROVED = '__PROVED__'
    DISPROVED = '__DISPROVED__'
    UNKNOWN = '__UNKNOWN__'

    # # # large markers
    # PROVED = '__PROVED__' * 10
    # DISPROVED = '__DISPROVED__' * 10
    # UNKNOWN = '__UNKNOWN__' * 10


def get_stance_markers(text: str) -> List[StanceMarker]:
    markers = []
    for marker in [StanceMarker.PROVED,
                   StanceMarker.DISPROVED,
                   StanceMarker.UNKNOWN]:
        if re.search(f' *{marker.value}', text):
            markers.append(marker)
    return markers


def add_stance_markers(text: str, markers: List[StanceMarker]) -> str:
    # if re.match(r'^\s*$', text):
    #     return text

    for marker in markers:
        if not re.search(f' *{marker.value}', text):
            if text == '':
                text = marker.value
            else:
                text += ' ' + marker.value

    return text


def delete_stance_markers(text: str) -> str:
    for marker in [StanceMarker.PROVED, StanceMarker.DISPROVED, StanceMarker.UNKNOWN]:
        if re.search(f' *{marker.value}', text):
            text = re.sub(f' *{marker.value}', '', text)
    return text


def marker_to_answer(marker: StanceMarker) -> Answer:
    if marker == marker.PROVED:
        return True
    elif marker == marker.DISPROVED:
        return False
    elif marker == marker.UNKNOWN:
        return 'Unknown'
    else:
        raise ValueError(f'Unknown marker {marker}')


def answer_to_marker(answer: Answer) -> StanceMarker:
    if answer is True:
        return StanceMarker.PROVED
    elif answer is False:
        return StanceMarker.DISPROVED
    elif answer == 'Unknown':
        return StanceMarker.UNKNOWN
    else:
        raise ValueError(f'Unknown answer {answer}')


def preprocess_example(component: str,
                       stance_indication_method: StanceIndicationMethod,
                       dataset_type: DatasetType,
                       is_train: bool,
                       ex: Example,
                       exclude_unknown=False,
                       reference_unknown_proof_ratio=0.0,
                       add_final_reference_to_proofs=False) -> List[Example]:
    # TODO: some of the logics are not stance marker related, thus we should transfer them to other module.
    if reference_unknown_proof_ratio > 0.0:
        raise NotImplementedError('We decided to reject reference_unknown_proof_ratio with other script. Thus, do not use this option.')

    if re.match(r'^\s*$', ex['context']):
        # Very rare case of FLNL.
        return []

    if component == 'prover':
        if stance_indication_method == StanceIndicationMethod.NEGATED_HYPOTHESIS_BY_I_DONT_THINK:
            if is_train:
                if ex["answer"] is False:
                    processed_exs = [_negate_example_by_i_dont_think(ex)]
                else:
                    processed_exs = [ex]
            else:
                processed_exs = [ex, _negate_example_by_i_dont_think(ex)]
        elif stance_indication_method == StanceIndicationMethod.STANCE_MARKER_IN_PROOF:
            processed_exs = [_preprocess_proof(ex, dataset_type,
                                               add_final_reference_to_proofs=add_final_reference_to_proofs)]
        else:
            raise ValueError(f'Unknown stance_indication_method {str(stance_indication_method)}')

    elif component == 'verifier':
        if stance_indication_method == StanceIndicationMethod.NEGATED_HYPOTHESIS_BY_I_DONT_THINK:
            if ex["answer"] is False:
                processed_exs = [_negate_example_by_i_dont_think(ex)]
            else:
                processed_exs = [ex]
        elif stance_indication_method == StanceIndicationMethod.STANCE_MARKER_IN_PROOF:
            processed_exs = [_preprocess_proof(ex, dataset_type,
                                               add_final_reference_to_proofs=add_final_reference_to_proofs)]
        else:
            raise ValueError(f'Unknown stance_indication_method {str(stance_indication_method)}')
    else:
        raise ValueError(f'Unknown component {component}')

    for processed_ex in processed_exs:
        if processed_ex['answer'] == 'Unknown':
            processed_ex['depth'] = None

    if exclude_unknown:
        processed_exs = [processed_ex for processed_ex in processed_exs
                         if processed_ex['answer'] != 'Unknown']

    if reference_unknown_proof_ratio > 0.0:
        for processed_ex in processed_exs:
            if processed_ex['answer'] != 'Unknown':
                continue

            proofs = processed_ex['proofs']
            if len(proofs) != 1:
                raise ValueError()

            proof = delete_stance_markers(proofs[0])
            num_steps = re.sub(' *; *$', '', proof).split(';')
            if len(num_steps) == 1:
                if random.random() < reference_unknown_proof_ratio:
                    logger.info('step 1 unknown proof is filtered out: %s', proof)
                    processed_ex['proofs'] = []

    return processed_exs


def _negate_example_by_i_dont_think(ex: Example) -> Example:
    negated_ex = deepcopy(ex)
    negated_ex['answer'] = not negated_ex["answer"] if negated_ex["answer"] != "Unknown" else "Unknown"
    negated_ex['hypothesis'] = f"{I_DONT_THINK} {negated_ex['hypothesis']}"
    return negated_ex


def _preprocess_proof(ex: Example, dataset_type: DatasetType, add_final_reference_to_proofs=False) -> Example:
    preprocessed_ex = deepcopy(ex)

    if dataset_type in [DatasetType.RuleTaker, DatasetType.FLNL]:
        # add negatives if not exists
        if 'negative_answer' not in ex:
            if ex.get('negative_proof_stance', None) is not None:
                negative_answer = _stance_to_answer(ex['negative_proof_stance'])
            else:
                negative_answer = None
            # for old version
            preprocessed_ex["negative_answer"] = negative_answer

        if 'negative_hypothesis' not in ex:
            preprocessed_ex["negative_hypothesis"] = None

        if 'negative_proofs' not in preprocessed_ex:
            preprocessed_ex['negative_proofs'] = []

        # filter None proofs
        preprocessed_ex['proofs'] = [proof for proof in preprocessed_ex['proofs']
                                     if proof is not None]
        preprocessed_ex['negative_proofs'] = [negative_proof for negative_proof in preprocessed_ex['negative_proofs']
                                              if negative_proof is not None]

        # filter proof for answer='Unknown' that ends with "void -> assump"
        # since the last "-> assump" may not have enough context.
        if preprocessed_ex['answer'] == 'Unknown':
            preprocessed_ex['proofs'] = [_strip_last_assump_step(proof)
                                         for proof in preprocessed_ex['proofs']
                                         if _strip_last_assump_step(proof) is not None]

        if preprocessed_ex['answer'] == 'Unknown':
            preprocessed_ex['negative_proofs'] = [_strip_last_assump_step(negative_proof)
                                                  for negative_proof in preprocessed_ex['negative_proofs']
                                                  if _strip_last_assump_step(negative_proof) is not None]

        if add_final_reference_to_proofs:
            preprocessed_ex = _add_final_reference_to_proofs(preprocessed_ex)

        preprocessed_ex = _add_random_proof_for_unknown(preprocessed_ex)

    preprocessed_ex = _add_stance_marker_to_proof(preprocessed_ex, dataset_type)

    return preprocessed_ex


def _strip_last_assump_step(proof: str) -> Optional[str]:
    steps = extract_steps(proof)
    last_step = steps[-1]
    _, concl = extract_premise_concl(last_step)
    last_ident = extract_ident(concl, allow_sentence=True)

    if last_ident.startswith(ASSUMP_IDENT):
        if len(steps) == 1:
            return None
        else:
            proof_wo_assump = '; '.join(steps[:-1]) + ';'
            return proof_wo_assump
    else:
        return proof


def _stance_to_answer(stance: str) -> Answer:
    if stance == 'PROOF':
        return True
    elif stance == 'DISPROOF':
        return False
    elif stance == 'UNKNOWN':
        return 'Unknown'
    else:
        raise ValueError('')


def _add_final_reference_to_proofs(ex: Example) -> Example:
    ex = deepcopy(ex)

    ex['proofs'] = [
        add_final_reference(
            ex['context'],
            ex['hypothesis'],
            proof,
            ex['answer'],
            dataset_depth=ex['depth'],
        )
        for proof in ex['proofs']
    ]

    ex['negative_proofs'] = [
        add_final_reference(
            ex['context'],
            ex['negative_hypothesis'],
            proof,
            ex['negative_answer'],
            dataset_depth=ex['depth'],
        )
        for proof in ex['negative_proofs']
    ]

    return ex


def _add_random_proof_for_unknown(ex: Example, step_from_special_sentence=True) -> Example:
    from src.prover.proof import SENT_IDENT  # XXX do not move to the top to avoid circular imports

    ex = deepcopy(ex)


    sent_ids = [ident for ident in extract_idents(ex['context'])
                if ident.startswith(SENT_IDENT)]

    if step_from_special_sentence:
        next_sent_id = max(int(sent_id[len(SENT_IDENT):]) for sent_id in sent_ids) + 1
        unknown_sent_id = f'{SENT_IDENT}{str(next_sent_id)}'
        unk_step = f'{unknown_sent_id} -> hypothesis;'

        # add special sentences to all the instances so that the model can not judge from whther
        # the special sentence exists or not
        ex['context'] = ex['context'] + f' {unknown_sent_id}: answer is unknown'

        def make_proofs_with_unknown_sentence(proofs: List[str]) -> List[str]:

            if len(proofs) == 0:
                proofs_with_special_sentence = [unk_step]
            else:
                proofs_with_special_sentence = []
                for proof in proofs:
                    if proof.find(' -> hypothesis') >= 0:

                        proofs_with_special_sentence.append(unk_step)

                        logger.info('hypothesis is already in the proof "%s". will be replaced as "%s"', proof, unk_step)

                    else:
                        # assume that the proof is the maximum subproof to the hypothesis
                        final_concl_int_ident: Optional[str] = None
                        for step in extract_steps(proof):
                            _, concl = extract_premise_concl(step)
                            concl_ident = extract_ident(concl, allow_sentence=True)

                            if concl_ident.startswith(INT_IDENT):
                                final_concl_int_ident = concl_ident

                        if final_concl_int_ident is None:
                            continue

                        # int_idents = [ident for ident in idents if re.match(f'^{INT_IDENT}[0-9]+$', ident)]
                        # final_concl_int_ident = int_idents[-1]
                        proof_with_unk_step = re.sub('; *$', '', proof) + '; ' + f'{final_concl_int_ident} -> hypothesis;'
                        proofs_with_special_sentence.append(proof_with_unk_step)

                        logger.info('-- Unk step will be concatenated to the proof --')
                        logger.info('orig                       : "%s"', proof)
                        logger.info('proof with unknown step    : "%s"', proof_with_unk_step)

            return proofs_with_special_sentence

        if ex['answer'] == 'Unknown':
            ex['proofs'] = make_proofs_with_unknown_sentence(ex['proofs'])

        if ex['negative_answer'] == 'Unknown':
            ex['negative_proofs'] = make_proofs_with_unknown_sentence(ex['negative_proofs'])

    else:
        # -- old version --
        proofs = ex['proofs']
        answer = ex['answer']
        if len(proofs) == 0:
            if answer != 'Unknown':
                raise ValueError()
            random_sent_id = random.choice(sent_ids)
            proofs = [f'{str(random_sent_id)} -> hypothesis;']
        ex['proofs'] = proofs

    return ex


def _add_stance_marker_to_proof(ex: Example, dataset_type: DatasetType) -> Example:
    ex = deepcopy(ex)

    if dataset_type == DatasetType.EB:
        answer = True
        ex['proof'] = add_stance_markers(ex['proof'], [answer_to_marker(answer)])

    elif dataset_type in [DatasetType.RuleTaker, DatasetType.FLNL]:
        ex['proofs'] = [
            add_stance_markers(proof, [answer_to_marker(ex['answer'])])
            for proof in ex['proofs']
        ]

        negative_answer = ex['negative_answer']
        ex['negative_proofs'] = [
            add_stance_markers(negative_proof, [answer_to_marker(negative_answer)])
            for negative_proof in ex['negative_proofs']
        ]

    else:
        raise ValueError()

    return ex


def find_step_text(text: str) -> Optional[str]:
    stance_markers = get_stance_markers(text)
    if text.find(";") >= 0:
        text = text[:text.find(";")]
    text = add_stance_markers(text, stance_markers)
    return text
