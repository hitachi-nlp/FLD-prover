import math
from typing import List
from pprint import pformat
from copy import deepcopy
from typing import Set, Tuple, Dict, Callable, Optional, Any
import logging

from common import calc_F
from proof_common import extract_assumptions, get_node_type, NodeType, extract_idents
from stance_indication import delete_stance_markers, get_stance_markers, StanceMarker
from FLD_task.proof import InvalidProof
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from proof_common import (
    get_node_type,
    NodeType,
    HYPOTHESIS_IDENT,
    VOID_IDENT,
)
import datasets
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

_LEVENSTEIN = NormalizedLevenshtein()
_BLEURT = datasets.load_metric('bleurt', 'bleurt-large-512')
_ROUGE = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL'],
    use_stemmer=True,
)

# We tuned the following threshold using "./tests/prover/test_scoring.py"
LEVENSTEIN_SIMILARITY_THRESHOLD = 0.25
ROUGE_THRESHOLD = 0.30
BLEURT_SIMILARITY_THRESHOLD = - 0.40


def _calc_levenstein_similarity(this: str, that: str) -> float:
    return 1 - _LEVENSTEIN.distance(this, that)


def calc_levenstein_similarity_batch(golds: List[str], preds: List[str]) -> List[float]:
    return [_calc_levenstein_similarity(gold, pred) for gold, pred in zip(golds, preds)]


def calc_bleurt_similarity_batch(golds: List[str], preds: List[str]) -> List[float]:
    return _BLEURT.compute(
        references=golds,
        predictions=preds,
    )['scores']


def calc_rouge_batch(golds: List[str], preds: List[str]) -> List[float]:
    scores = []
    for gold, pred in zip(golds, preds):
        if len(gold.split(' ')) <= 2 or len(pred.split(' ')) <= 2:
            score = _calc_levenstein_similarity(gold, pred)
        else:
            score = _ROUGE.score(pred, gold)['rouge2'].fmeasure
        scores.append(score)
    return scores


def calc_max_pooling_similarity_batch(golds: List[str], preds: List[str]) -> List[float]:
    lev_sims = calc_levenstein_similarity_batch(golds, preds)
    bleurt_sims = calc_bleurt_similarity_batch(golds, preds)
    rouge_sims = calc_rouge_batch(golds, preds)

    return [max(lev_sims[i], bleurt_sims[i], rouge_sims[i]) for i in range(0, len(golds))]


def calc_accuracy(proof_gold_text: str,
                  proof_pred_text: str,
                  similarity_threshold=False,
                  allowed_additional_proof_steps=0,
                  zero_one: bool = True) -> float:
    gold_labels = get_stance_markers(proof_gold_text)
    pred_labels = get_stance_markers(proof_pred_text)
    if set(gold_labels) != set(pred_labels):
        return 0.0

    if gold_labels == set([StanceMarker.UNKNOWN]):
        return 1.0
    else:
        proof_score = calc_score(
            delete_stance_markers(proof_gold_text).rstrip(' '),
            delete_stance_markers(proof_pred_text).rstrip(' '),
            similarity_threshold=similarity_threshold,
            allowed_additional_proof_steps=allowed_additional_proof_steps,
            zero_one=zero_one,
        )
        return proof_score


def calc_score(proof_gold_text: str,
               proof_pred_text: str,
               similarity_threshold=True,
               allowed_additional_proof_steps=0,
               zero_one: bool = True) -> float:
    """ Calculate the similarity score between gold and prediction.

    The score is invariant under the renaming of intermediate nodes.
    """
    logger.info('\n\n========================================= calc_score() ==================================================')
    # TODO: should score the tree by global similarity between tree like AMR.
    # TODO: should calculate  precision / recall / f rather than a single score.
    (
        gold_premise_uids_to_concl_uid,
        gold_premise_uids_to_concl_sent,
        pred_premise_uids_to_concl_uid,
        pred_premise_uids_to_concl_sent,
    ) = _get_aligned_proof_by_uids(proof_gold_text, proof_pred_text)

    logger.info('=========== gold_premise_uids_to_concl_uid ==============')
    logger.info('\n' + pformat(gold_premise_uids_to_concl_uid))
    logger.info('=========== pred_premise_uids_to_concl_uid ==============')
    logger.info('\n' + pformat(pred_premise_uids_to_concl_uid))

    if zero_one:
        if len(pred_premise_uids_to_concl_uid) < len(gold_premise_uids_to_concl_uid)\
                or len(pred_premise_uids_to_concl_uid) > len(gold_premise_uids_to_concl_uid) + allowed_additional_proof_steps:
            return 0.0

    # calculate similarities
    golds: List[str] = []
    preds: List[str] = []
    for gold_premise_uids, gold_concl_uid in gold_premise_uids_to_concl_uid.items():
        if gold_premise_uids not in pred_premise_uids_to_concl_uid:
            continue
        golds.append(gold_premise_uids_to_concl_sent[gold_premise_uids])
        preds.append(pred_premise_uids_to_concl_sent[gold_premise_uids])

    if similarity_threshold:
        levenstein_sims = calc_levenstein_similarity_batch(golds, preds)
        rouge_sims = calc_rouge_batch(golds, preds)
    else:
        levenstein_sims = [float('inf')] * len(golds)
        rouge_sims = [float('inf')] * len(golds)
        # we do not use bleurt similarity since it produce awkwardly low score for synthetic non-sense sentences.
        # And also, bleurt similarity scorer is slow.

    tot = len(gold_premise_uids_to_concl_uid)
    TP = 0
    for levenstein_sim, rouge_sim in zip(levenstein_sims, rouge_sims):
        if levenstein_sim >= LEVENSTEIN_SIMILARITY_THRESHOLD or rouge_sim >= ROUGE_THRESHOLD:
            TP += 1

    FP = 0
    FP_budget = allowed_additional_proof_steps
    for pred_premise_uids, pred_concl_uid in pred_premise_uids_to_concl_uid.items():
        if pred_premise_uids not in gold_premise_uids_to_concl_uid:
            if FP_budget == 0:
                FP += 1
            else:
                FP_budget -= 1

    precision, recall, F = calc_F(tot, TP, FP)

    if zero_one and not math.isclose(F, 1.0):
        return 0.0
    else:
        return F


def _is_uid(id_: str) -> bool:
    return id_.isupper()


def _to_uid(id_: str) -> str:
    return id_.upper()


def _get_aligned_proof_by_uids(proof_gold_text: str, proof_pred_text: str)\
        -> Tuple[Dict[Tuple[str], str], Dict[Tuple[str], str], Dict[Tuple[str], str], Dict[Tuple[str], str]]:
    logger.info('\n\n=================================== _get_aligned_proof_by_uids() ============================================')

    gold_premise_ids_to_concl_id, gold_premise_ids_to_concl_sent = _split_steps_into_id_dics(proof_gold_text)
    pred_premise_ids_to_concl_id, pred_premise_ids_to_concl_sent = _split_steps_into_id_dics(proof_pred_text)

    def premise_sort_key(premise_ids: Tuple[str]) -> Any:
        # sort key so that ('sent1', 'sent2') is faster than ('sent1', 'int1') and ('int1', 'int2')
        # not that False will be at the first of the sequence when sorted
        return (
            not all(get_node_type(premise_id) == NodeType.sent for premise_id in premise_ids),
            not any(get_node_type(premise_id) == NodeType.sent for premise_id in premise_ids),
        )

    gold_id_to_uid: Dict[str, str] = {}
    pred_id_to_uid: Dict[str, str] = {}

    # for ident in re.findall(r'(sent\d+|int\d+|assump\d+)', proof_gold_text):
    #     gold_id_to_uid[ident] = ident.upper()
    for ident in extract_idents(proof_gold_text):
        if get_node_type(ident) in [NodeType.sent, NodeType.int, NodeType.assump, NodeType.assump_deletion]:
            gold_id_to_uid[ident] = ident.upper()

    for ident in extract_idents(proof_pred_text):
        if get_node_type(ident) in [NodeType.sent]:
            pred_id_to_uid[ident] = ident.upper()

    gold_id_to_uid.update(_make_assumption_mapping(proof_gold_text, proof_pred_text)[0])
    pred_id_to_uid.update(_make_assumption_mapping(proof_gold_text, proof_pred_text)[1])

    gold_premise_uids_to_concl_uid: Dict[Tuple[str], str] = deepcopy(gold_premise_ids_to_concl_id)
    pred_premise_uids_to_concl_uid: Dict[Tuple[str], str] = deepcopy(pred_premise_ids_to_concl_id)
    done_premise_uids: Set[Tuple[str]] = set()

    def find_gold_premise_ids_from_pred_premise_uids(gold_premise_ids: Tuple[str]) -> Optional[Tuple[str]]:
        if gold_premise_ids in pred_premise_uids_to_concl_uid:
            return gold_premise_ids
        return None

    while True: 
        logger.info('\n========================= while loop ==========================')

        gold_premise_uids_to_concl_uid = {
            tuple(sorted([gold_id_to_uid.get(premise_id, premise_id) for premise_id in premise_ids])): gold_id_to_uid.get(concl_id, concl_id)
            for premise_ids, concl_id in gold_premise_uids_to_concl_uid.items()
        }
        pred_premise_uids_to_concl_uid = {
            tuple(sorted([pred_id_to_uid.get(premise_id, premise_id) for premise_id in premise_ids])): pred_id_to_uid.get(concl_id, concl_id)
            for premise_ids, concl_id in pred_premise_uids_to_concl_uid.items()
        }
        logger.info('----------- gold_id_to_uid --------------')
        logger.info('\n' + pformat(gold_id_to_uid))
        logger.info('----------- pred_id_to_uid --------------')
        logger.info('\n' + pformat(pred_id_to_uid))

        logger.info('----------- build resolution dicts --------------')
        is_resolved_any = False
        for gold_premise_ids, gold_conclusion_id in sorted(gold_premise_uids_to_concl_uid.items(),
                                                           key=lambda k_v: premise_sort_key(k_v[0])):
            logger.info('    gold_premise_ids %s: ', gold_premise_ids)
            _gold_premise_uids = tuple(_to_uid(id_) for id_ in gold_premise_ids)
            if _gold_premise_uids in done_premise_uids:
                logger.info('        skip since already done')
                continue

            _gold_conclusion_uid = _to_uid(gold_conclusion_id)

            matched_pred_premise_ids = find_gold_premise_ids_from_pred_premise_uids(gold_premise_ids)
            if matched_pred_premise_ids is not None:
                logger.info('        updating dicts by: %s', gold_premise_ids)
                done_premise_uids.add(_gold_premise_uids)
                pred_conclusion_id = pred_premise_uids_to_concl_uid[matched_pred_premise_ids]

                for gold_premise_id, matched_pred_premise_id, _gold_premise_uid in zip(gold_premise_ids, matched_pred_premise_ids, _gold_premise_uids):
                    if gold_premise_id not in gold_id_to_uid and not _is_uid(gold_premise_id):
                        gold_id_to_uid[gold_premise_id] = _gold_premise_uid
                    if matched_pred_premise_id not in pred_id_to_uid and not _is_uid(matched_pred_premise_id):
                        pred_id_to_uid[matched_pred_premise_id] = _gold_premise_uid
                if gold_conclusion_id not in gold_id_to_uid and not _is_uid(gold_conclusion_id):
                    gold_id_to_uid[gold_conclusion_id] = _gold_conclusion_uid
                if pred_conclusion_id not in pred_id_to_uid and not _is_uid(pred_conclusion_id):
                    pred_id_to_uid[pred_conclusion_id] = _gold_conclusion_uid

                is_resolved_any = True
                break
            else:
                logger.info('        skip since it it not found in prediction: %s', gold_premise_ids)
                pass

        if not is_resolved_any:
            break

    gold_premise_uids_to_concl_sent = {
        tuple(sorted([gold_id_to_uid.get(premise_id, premise_id) for premise_id in premise_ids])): sent
        for premise_ids, sent in gold_premise_ids_to_concl_sent.items()
    }
    pred_premise_uids_to_concl_sent = {
        tuple(sorted([pred_id_to_uid.get(premise_id, premise_id) for premise_id in premise_ids])): sent
        for premise_ids, sent in pred_premise_ids_to_concl_sent.items()
    }

    return (
        gold_premise_uids_to_concl_uid,
        gold_premise_uids_to_concl_sent,
        pred_premise_uids_to_concl_uid,
        pred_premise_uids_to_concl_sent,
    )


def _split_steps_into_id_dics(proof: str) -> Tuple[Dict[Tuple[str, str], str], Dict[str, str]]:
    proof = delete_stance_markers(proof)

    premise_ids_to_concl_id: Dict[Tuple[str], str] = {}
    premise_ids_to_concl_sent: Dict[Tuple[str], str] = {}
    i_void_premise = 1
    for step_text in proof.split(";"):
        step_text = step_text.strip()
        if step_text == "":
            continue
        if step_text.count(" -> ") != 1:
            raise InvalidProof()

        premises_text, conclusion_text = step_text.split(" -> ")

        premise_ids = sorted(premises_text.split(' & '))
        if premise_ids == [VOID_IDENT]:
            premise_ids = [f'{VOID_IDENT}{i_void_premise}']
            i_void_premise += 1

        if get_node_type(conclusion_text) == NodeType.hypothesis:
            conclusion_id = HYPOTHESIS_IDENT
            conclusion_sentence = HYPOTHESIS_IDENT
        else:
            if conclusion_text.find(': ') < 0:
                raise InvalidProof()
            conclusion_id = conclusion_text.split(': ')[0]
            conclusion_sentence = ': '.join(conclusion_text.split(': ')[1:])

        premise_ids_to_concl_id[tuple(premise_ids)] = conclusion_id
        premise_ids_to_concl_sent[tuple(premise_ids)] = conclusion_sentence

    return premise_ids_to_concl_id, premise_ids_to_concl_sent


def _make_assumption_mapping(proof_gold_text: str,
                             proof_pred_text: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    gold_assumptions = extract_assumptions(proof_gold_text)
    pred_assumptions = extract_assumptions(proof_pred_text)

    gold_id_to_uid: Dict[str, str] = {}
    pred_id_to_uid: Dict[str, str] = {}

    consumed_pred_ids = set()
    for gold_idx, (gold_id, gold_sentence) in enumerate(gold_assumptions.items()):
        max_similarity = - float('inf')
        max_pred_id = None
        max_pred_idx = None

        for pred_void_idx, (pred_id, pred_sentence) in enumerate(pred_assumptions.items()):
            if pred_id in consumed_pred_ids:
                continue

            similarity = _calc_levenstein_similarity(gold_sentence, pred_sentence)
            if similarity > max_similarity and similarity >= LEVENSTEIN_SIMILARITY_THRESHOLD:
                max_pred_id = pred_id
                max_similarity = similarity
                max_pred_idx = pred_void_idx

        if max_pred_id is not None:
            consumed_pred_ids.add(max_pred_id)
            gold_id_to_uid[gold_id] = _to_uid(gold_id)
            pred_id_to_uid[max_pred_id] = _to_uid(gold_id)

            gold_id_to_uid[f'{VOID_IDENT}{gold_idx + 1}'] = _to_uid(f'{VOID_IDENT}{gold_idx + 1}')
            pred_id_to_uid[f'{VOID_IDENT}{max_pred_idx + 1}'] = _to_uid(f'{VOID_IDENT}{gold_idx + 1}')

    for gold_id, gold_uid in list(gold_id_to_uid.items()):
        gold_id_to_uid[f'[{gold_id}]'] = f'[{gold_uid}]'

    for pred_id, pred_uid in list(pred_id_to_uid.items()):
        pred_id_to_uid[f'[{pred_id}]'] = f'[{pred_uid}]'

    return gold_id_to_uid, pred_id_to_uid
