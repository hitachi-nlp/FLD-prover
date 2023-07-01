import math
from typing import List
from pprint import pformat
from copy import deepcopy
from typing import Set, Tuple, Dict, Callable, Optional, Any
import logging
import re

import evaluate
import nltk
from common import calc_F
from proof_common import extract_assumptions, get_node_type, NodeType, extract_idents, extract_context
from stance_indication import delete_stance_markers, get_stance_markers, StanceMarker
from FLD_task.proof import InvalidProof, InvalidProofStep
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from proof_common import (
    get_node_type,
    NodeType,
    HYPOTHESIS_IDENT,
    VOID_IDENT,
    INT_IDENT,
    SENT_IDENT,
    normalize_proof,
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
_HF_ROUGE_METRIC = evaluate.load("rouge")

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


def _hf_rouge_postprocess_text(preds: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def _hf_compute_rouges(decoded_labels: List[str], decoded_preds: List[str]) -> Dict[str, Any]:
    # Some simple post-processing
    _decoded_preds, _decoded_labels = _hf_rouge_postprocess_text(decoded_preds, decoded_labels)
    result = _HF_ROUGE_METRIC.compute(predictions=_decoded_preds, references=_decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    return result


def calc_metrics(proof_gold_texts: List[str],
                 proof_pred_text: str,
                 allow_reference_step=False,
                 context: Optional[str] = None,
                 similarity_threshold=False,
                 allowed_additional_proof_steps=0,
                 allow_any_proof_for_unknown=False,
                 zero_one: bool = True) -> Dict[str, Any]:
    if len(proof_gold_texts) >= 2:
        raise NotImplementedError()
    proof_gold_text = normalize_proof(proof_gold_texts[0])
    proof_pred_text = normalize_proof(proof_pred_text)

    metrics: Dict[str, Any] = {}

    gold_labels = set(get_stance_markers(proof_gold_text))
    pred_labels = set(get_stance_markers(proof_pred_text))

    metrics['answer_accuracy'] = float(gold_labels == pred_labels)

    zero_one_acc = calc_accuracy(
        proof_gold_text,
        proof_pred_text,
        allow_reference_step=allow_reference_step,
        context=context,
        similarity_threshold=similarity_threshold,
        allowed_additional_proof_steps=allowed_additional_proof_steps,
        allow_any_proof_for_unknown=allow_any_proof_for_unknown,
        zero_one=zero_one,
    )
    metrics['proof_accuracy.zero_one'] = zero_one_acc

    rouges = _hf_compute_rouges([proof_gold_text], [proof_pred_text])
    metrics.update(rouges)

    return metrics


def calc_accuracy(proof_gold_text: str,
                  proof_pred_text: str,
                  allow_reference_step=False,
                  context: Optional[str] = None,
                  similarity_threshold=False,
                  allowed_additional_proof_steps=0,
                  allow_any_proof_for_unknown=False,
                  zero_one: bool = True) -> float:
    proof_gold_text = normalize_proof(proof_gold_text)
    proof_pred_text = normalize_proof(proof_pred_text)

    gold_labels = get_stance_markers(proof_gold_text)
    pred_labels = get_stance_markers(proof_pred_text)

    if set(gold_labels) != set(pred_labels):
        return 0.0

    if allow_any_proof_for_unknown and set(gold_labels) == set([StanceMarker.UNKNOWN]):
        return 1.0
    else:
        try:
            proof_score = calc_score(
                delete_stance_markers(proof_gold_text).rstrip(' '),
                delete_stance_markers(proof_pred_text).rstrip(' '),
                allow_reference_step=allow_reference_step,
                context=context,
                similarity_threshold=similarity_threshold,
                allowed_additional_proof_steps=allowed_additional_proof_steps,
                zero_one=zero_one,
            )
        except (InvalidProof, InvalidProofStep) as e:
            proof_score = 0.0
        return proof_score


def calc_score(proof_gold_text: str,
               proof_pred_text: str,
               allow_reference_step=False,
               context: Optional[str] = None,
               similarity_threshold=True,
               allowed_additional_proof_steps=0,
               zero_one: bool = True) -> float:
    """ Calculate the similarity score between gold and prediction.

    The score is invariant under the renaming of intermediate nodes.
    """
    logger.debug('\n\n========================================= calc_score() ==================================================')
    (
        gold_premise_uids_to_concl_uid,
        gold_premise_uids_to_concl_sent,
        pred_premise_uids_to_concl_uid,
        pred_premise_uids_to_concl_sent,
    ) = _get_aligned_proof_by_uids(proof_gold_text,
                                   proof_pred_text,
                                   allow_reference_step=allow_reference_step,
                                   context=context)


    logger.debug('=========== gold_premise_uids_to_concl_uid ==============')
    logger.debug('\n' + pformat(gold_premise_uids_to_concl_uid))
    logger.debug('=========== pred_premise_uids_to_concl_uid ==============')
    logger.debug('\n' + pformat(pred_premise_uids_to_concl_uid))

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


def _get_aligned_proof_by_uids(proof_gold_text: str,
                               proof_pred_text: str,
                               allow_reference_step=False,
                               context: Optional[str] = None)\
        -> Tuple[Dict[Tuple[str], str], Dict[Tuple[str], str], Dict[Tuple[str], str], Dict[Tuple[str], str]]:
    logger.debug('\n\n=================================== _get_aligned_proof_by_uids() ============================================')

    gold_premise_ids_to_concl_id, gold_premise_ids_to_concl_sent = _split_steps_into_id_dics(proof_gold_text)
    pred_premise_ids_to_concl_id, pred_premise_ids_to_concl_sent = _split_steps_into_id_dics(proof_pred_text)

    if allow_reference_step:
        # truncate reference steps which is something like "sent4 (hoge) -> int1: hoge"

        if context is None:
            raise ValueError('can not judge reference step because the context is not specified')
        context_sents = extract_context(context)
        context_sents = {id_: sent for id_, sent in context_sents.items()}

        def is_reference(gold_sent: str, pred_sent: str) -> bool:
            # LLMs generates reference sentences slightly different from the original one.
            # Thus, we allow similarity less than 1.0
            lev_sim = calc_levenstein_similarity_batch([gold_sent], [pred_sent])[0]
            return lev_sim >= 0.7

        reference_sent_id_to_int_id: Dict[str, str] = {}
        reference_int_id_to_sent_id: Dict[str, str] = {}
        for premise_ids, concl_id in pred_premise_ids_to_concl_id.items():
            concl_sent = pred_premise_ids_to_concl_sent[premise_ids]
            if len(premise_ids) == 1:
                premise_id = premise_ids[0]
                if premise_id.startswith(SENT_IDENT):
                    premise_sent = context_sents[premise_id]
                    if is_reference(premise_sent, concl_sent):
                        reference_sent_id_to_int_id[premise_id] = concl_id
                        reference_int_id_to_sent_id[concl_id] = premise_id

        pred_premise_ids_to_concl_id_reference_exluded = {}
        for premise_ids, concl_id in pred_premise_ids_to_concl_id.items():
            if len(premise_ids) == 1 and premise_ids[0] in reference_sent_id_to_int_id:
                # exclude reference step
                continue
            premise_ids_replaced = tuple(
                sorted(reference_int_id_to_sent_id.get(premise_id, premise_id)
                       for premise_id in premise_ids)
            )
            pred_premise_ids_to_concl_id_reference_exluded[premise_ids_replaced] = concl_id

        pred_premise_ids_to_concl_sent_reference_exluded = {}
        for premise_ids, concl_sent in pred_premise_ids_to_concl_sent.items():
            if len(premise_ids) == 1 and premise_ids[0] in reference_sent_id_to_int_id:
                # exclude reference step
                continue
            premise_ids_replaced = tuple(
                sorted(reference_int_id_to_sent_id.get(premise_id, premise_id)
                       for premise_id in premise_ids)
            )
            pred_premise_ids_to_concl_sent_reference_exluded[premise_ids_replaced] = concl_sent

        # fill vacant int ids
        all_pred_int_ids: Set[str] = set()
        for premise_ids, concl_id in pred_premise_ids_to_concl_id_reference_exluded.items():
            for id_ in list(premise_ids) + [concl_id]:
                if id_.startswith(INT_IDENT):
                    all_pred_int_ids.add(id_)

        int_ids_map: Dict[str, str] = {}
        i = 1
        for int_id in sorted(all_pred_int_ids, key = lambda int_id: int(int_id[3:])):
            int_idx = int(int_id[3:])
            if int_idx != i:
                int_ids_map[int_id] = f'{INT_IDENT}{i}'
            i += 1

        pred_premise_ids_to_concl_id_reference_exluded_int_remaped = {}
        for premise_ids, concl_id in pred_premise_ids_to_concl_id_reference_exluded.items():
            premise_ids_replaced = tuple(
                sorted(int_ids_map.get(premise_id, premise_id)
                       for premise_id in premise_ids)
            )
            concl_id_replaced = int_ids_map.get(concl_id, concl_id)
            pred_premise_ids_to_concl_id_reference_exluded_int_remaped[premise_ids_replaced] = concl_id_replaced

        pred_premise_ids_to_concl_sent_reference_exluded_int_remaped = {}
        for premise_ids, concl_sent in pred_premise_ids_to_concl_sent_reference_exluded.items():
            premise_ids_replaced = tuple(
                sorted(int_ids_map.get(premise_id, premise_id)
                       for premise_id in premise_ids)
            )
            pred_premise_ids_to_concl_sent_reference_exluded_int_remaped[premise_ids_replaced] = concl_sent

        pred_premise_ids_to_concl_id = pred_premise_ids_to_concl_id_reference_exluded_int_remaped
        pred_premise_ids_to_concl_sent = pred_premise_ids_to_concl_sent_reference_exluded_int_remaped

    def premise_sort_key(premise_ids: Tuple[str]) -> Any:
        # sort key so that ('sent1', 'sent2') is faster than ('sent1', 'int1') and ('int1', 'int2')
        # not that False will be at the first of the sequence when sorted
        return (
            not all(get_node_type(premise_id) == NodeType.sent for premise_id in premise_ids),
            not any(get_node_type(premise_id) == NodeType.sent for premise_id in premise_ids),
        )

    gold_id_to_uid: Dict[str, str] = {}
    pred_id_to_uid: Dict[str, str] = {}

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
        logger.debug('\n========================= while loop ==========================')

        gold_premise_uids_to_concl_uid = {
            tuple(sorted([gold_id_to_uid.get(premise_id, premise_id) for premise_id in premise_ids])): gold_id_to_uid.get(concl_id, concl_id)
            for premise_ids, concl_id in gold_premise_uids_to_concl_uid.items()
        }
        pred_premise_uids_to_concl_uid = {
            tuple(sorted([pred_id_to_uid.get(premise_id, premise_id) for premise_id in premise_ids])): pred_id_to_uid.get(concl_id, concl_id)
            for premise_ids, concl_id in pred_premise_uids_to_concl_uid.items()
        }
        logger.debug('----------- gold_id_to_uid --------------')
        logger.debug('\n' + pformat(gold_id_to_uid))
        logger.debug('----------- pred_id_to_uid --------------')
        logger.debug('\n' + pformat(pred_id_to_uid))

        logger.debug('----------- build resolution dicts --------------')
        is_resolved_any = False
        for gold_premise_ids, gold_conclusion_id in sorted(gold_premise_uids_to_concl_uid.items(),
                                                           key=lambda k_v: premise_sort_key(k_v[0])):
            logger.debug('    gold_premise_ids %s: ', gold_premise_ids)
            _gold_premise_uids = tuple(_to_uid(id_) for id_ in gold_premise_ids)
            if _gold_premise_uids in done_premise_uids:
                logger.debug('        skip since already done')
                continue

            _gold_conclusion_uid = _to_uid(gold_conclusion_id)

            matched_pred_premise_ids = find_gold_premise_ids_from_pred_premise_uids(gold_premise_ids)
            if matched_pred_premise_ids is not None:
                logger.debug('        updating dicts by: %s', gold_premise_ids)
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
                logger.debug('        skip since it it not found in prediction: %s', gold_premise_ids)
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
            concl_id = HYPOTHESIS_IDENT
            concl_sent = HYPOTHESIS_IDENT
        else:
            if conclusion_text.find(': ') < 0:
                raise InvalidProof()
            concl_id = conclusion_text.split(': ')[0]
            concl_sent = ': '.join(conclusion_text.split(': ')[1:])

        premise_ids_to_concl_id[tuple(premise_ids)] = concl_id
        premise_ids_to_concl_sent[tuple(premise_ids)] = concl_sent

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


def build_metrics(type_: str) -> Callable[[List[str], str], Dict[str, float]]:
    """ A thin wrapper to bind the settings"""
    if type_ == 'strict':
        def calc(gold_proofs: List[str], pred_proof: str) -> Dict[str, float]:
            return calc_metrics(
                gold_proofs,
                pred_proof,
                similarity_threshold=False,
                allowed_additional_proof_steps=0,
                allow_any_proof_for_unknown=False,
            )
    elif type_ == 'allow_extra_steps':
        def calc(gold_proofs: List[str], pred_proof: str) -> Dict[str, float]:
            return calc_metrics(
                gold_proofs,
                pred_proof,
                similarity_threshold=False,
                allowed_additional_proof_steps=5,
                allow_any_proof_for_unknown=True,
            )
    else:
        raise ValueError(f'Unknown metrics type {type_}')
    return calc
