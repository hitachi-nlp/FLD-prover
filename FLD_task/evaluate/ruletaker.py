"""
Utilities for evaluation.
"""
import math
import logging
import itertools
from collections import defaultdict
from typing import Set, Dict, Any, List, Optional

from tqdm import tqdm
# from ete3 import TextFace, TreeStyle, NodeStyle
from PyPDF3.pdf import PageObject
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer
from FLD_task.proof import prettify_proof_text, InvalidProof
from common import (
    int2answer,
    answer2int,
    Example,
    I_DONT_THINK,
)
from proof_common import (
    extract_context,
    normalize,
    TreeNode,
    serialize,
    SENT_IDENT,
)

from .scoring import calc_score
from stance_indication import (
    StanceIndicationMethod,
    delete_stance_markers,
    marker_to_answer,
    get_stance_markers,
)


logger = logging.getLogger(__name__)


def _split_steps(proof: str) -> Set[Any]:
    proof = delete_stance_markers(proof)

    steps = set()
    for s in proof.split(";"):
        s = s.strip()
        if s == "":
            continue
        if s.count(" -> ") != 1:
            raise InvalidProof()
        premises, conclusion = s.split(" -> ")
        steps.add((tuple(sorted(premises.split(" & "))), conclusion == "hypothesis" ))
    return steps


def _check_ruletaker_proof(proof_pred: str,
                           proofs_gt: List[str],
                           allowed_additional_proof_steps=0,
                           no_similarity_threshold=False) -> bool:
    """
    Check whether two RukeTaker proofs are equivalent.
    """
    if proof_pred == 'INVALID_PROOF':
        return False

    proof_pred = delete_stance_markers(proof_pred)
    proofs_gt = [delete_stance_markers(p) for p in proofs_gt]

    ORIGINAL_IMPL = False
    if ORIGINAL_IMPL:
        # The original scoring below is inappropriate in two points:
        # (i) The score is not invariant under the renaming of intermediate nodes. This underestimates the performance.
        # (ii) It only checks the combination of premises are correct. This overestimates the performances.
        try:
            proof_steps_pred = _split_steps(proof_pred)
            proofs_steps_gt = [_split_steps(_) for _ in proofs_gt]
            is_correct = proof_steps_pred in proofs_steps_gt
        except InvalidProof as e:
            is_correct = False
    else:
        score_max = 0.0
        is_correct = False
        for proof_gt in proofs_gt:
            try:
                score = calc_score(proof_gt,
                                   proof_pred,
                                   no_similarity_threshold=no_similarity_threshold,
                                   allowed_additional_proof_steps=allowed_additional_proof_steps,
                                   zero_one=True)
                if score > score_max:
                    score_max = score
                if math.isclose(score_max, 1.0):
                    is_correct = True
                    break
            except InvalidProof as e:
                is_correct = False
    return is_correct


def _calc_ruletaker_rouge_score(proof_pred: str, proofs_gt: List[str]) -> Dict[str, float]:
    proof_pred = delete_stance_markers(proof_pred)
    proofs_gt = [delete_stance_markers(p) for p in proofs_gt]

    rouge_names = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL']
    sub_metrics = ['precision', 'recall', 'fmeasure']

    # This log-scale weighting leads to that
    # "If rouge-N score with higher N is nono-zero, then only consider it for taking alignment."
    # rouge_weights_for_align = [0.001, 0.01, 0.1, 1.0, 0.0]
    # assert len(rouge_names) == len(rouge_weights_for_align)

    rouge_name_for_gold_alignment = 'rouge2'

    if proof_pred == 'INVALID_PROOF':
        # logger.info('return 0.0 since proof is invalid')
        return {f'{rouge_name}.{sub_metric}': 0.0
                for rouge_name in rouge_names
                for sub_metric in sub_metrics}
    else:
        scorer = rouge_scorer.RougeScorer(rouge_names, use_stemmer=True)
        ret_scores = None

        # take the gold proof with maximum score, which we assume the aligned proof.
        max_score = - float('inf')
        aligned_gold_proof = None
        for gold_proof in proofs_gt:
            scores = scorer.score(proof_pred, gold_proof)
            alignment_score = scores[rouge_name_for_gold_alignment].fmeasure
            # alignment_score = sum([
            #     weight * scores[f'{rouge_name}'].fmeasure
            #     for rouge_name, weight in zip(rouge_names, rouge_weights_for_align)
            # ])
            if alignment_score > max_score:
                ret_scores = {
                    f'{rouge_name}.{sub_metric}': getattr(scores[rouge_name], sub_metric)
                    for rouge_name in rouge_names
                    for sub_metric in sub_metrics
                }
                aligned_gold_proof = gold_proof
                max_score = alignment_score

        return ret_scores


def process_ruletaker_results(results: List[Any],
                              stance_indication_method: StanceIndicationMethod) -> Any:
    scores = []
    labels = []
    depths = []
    all_proofs = []

    def line_head(head_text: str) -> str:
        _head_text = head_text
        return f'{_head_text:<30}: '

    def get_single_result_log_lines(result: Dict) -> List[str]:
        log_lines = []

        log_lines.append(f'{line_head("hypothesis")}{result["hypothesis"]}')

        log_lines.append('')
        log_lines.append(f'{line_head("context")}')
        for sent_id, sent in sorted(result['context'].items(), lambda sent_id_sent: int(sent_id_sent[0][len(SENT_IDENT):])):
            log_lines.append(f'    {sent_id}: {sent}')

        for i_gold, gold_proof in enumerate(result["all_proofs"]):
            log_lines.append('')
            log_lines.append(f'{line_head("gold_proof-" + str(i_gold))}')
            log_lines.append(prettify_proof_text(gold_proof, indent_level=1))

        log_lines.append('')
        log_lines.append(f'{line_head("proof_gt")}')
        log_lines.append(prettify_proof_text(result["proof_gt"], indent_level=1))

        log_lines.append('')
        log_lines.append(f'{line_head("proof_pred")}')
        log_lines.append(prettify_proof_text(result["proof_pred"], indent_level=1))

        log_lines.append('')
        log_lines.append(f'{line_head("depth")}{result["depth"]}')
        log_lines.append(f'{line_head("generation score")}{result["score"]}')
        log_lines.append(f'{line_head("gold_answer")}{result["answer"]}')

        return log_lines

    if stance_indication_method == StanceIndicationMethod.NEGATED_HYPOTHESIS_BY_I_DONT_THINK:
        assert len(results) % 2 == 0
        n = len(results) // 2

        for i in range(n):
            r_orig = results[2 * i]
            r_neg = results[2 * i + 1]
            assert r_orig["proof_gt"] == r_neg["proof_gt"]
            assert r_orig["depth"] == r_neg["depth"]
            assert r_orig["all_proofs"] == r_neg["all_proofs"]
            assert r_neg["hypothesis"] == f"{I_DONT_THINK} {r_orig['hypothesis']}"
            scores.append([r_orig["score"], r_neg["score"]])


            labels.append(answer2int(r_orig["answer"]))

            depths.append(r_orig["depth"])
            all_proofs.append(r_orig["all_proofs"])

    elif stance_indication_method == StanceIndicationMethod.STANCE_MARKER_IN_PROOF:
        for r_orig in results:
            stance_markers = get_stance_markers(r_orig['proof_pred'])
            predicted_answers = [marker_to_answer(marker) for marker in stance_markers]

            if len(predicted_answers) != 1:
                score = -1   # unused label so that the answer and proof is evaluated as incorrect
            else:
                score = answer2int(predicted_answers[0])
            scores.append([score])

            labels.append(answer2int(r_orig["answer"]))
            depths.append(r_orig["depth"])
            all_proofs.append(r_orig["all_proofs"])

    else:
        raise ValueError()

    return scores, labels, depths, all_proofs


def _calculate_ruletaker_metrics(
    y_pred: Any, y: Any, depths: Any, all_proofs: Any, results: Any, stance_indication_method: StanceIndicationMethod,
    allowed_additional_proof_steps=0,
    no_similarity_threshold=False,
) -> Any:
    """
    Calculate the answer accuracy and proof accuracy for RuleTaker.
    """
    answer_is_correct = defaultdict(list)
    n = len(y_pred)
    for i in range(n):
        depth = depths[i]
        answer_is_correct[depth].append(y[i] == y_pred[i])
    answer_accuracies_agg = {depth: np.mean(v) for depth, v in answer_is_correct.items()}
    answer_accuracies_agg["overall"] = accuracy_score(y, y_pred)
    answer_accuracies_agg["overall_wo_None"] = accuracy_score(
        [_y for _y in y if _y != 2],
        [_y_pred for _y_pred, _y in zip(y_pred, y) if _y != 2],
    )
    answer_accuracies_agg["overall_macro"] = np.mean([score for depth, score in answer_accuracies_agg.items()
                                                      if not str(depth).startswith('overall')])
    answer_accuracies_agg["overall_macro"] = np.mean([score for depth, score in answer_accuracies_agg.items()
                                                      if not str(depth).startswith('overall') and str(depth) != 'None'])

    proof_is_correct: Dict[int, List[bool]] = defaultdict(list)
    proof_scores: Dict[int, Dict[str, List[float]]] = defaultdict(lambda : defaultdict(list))
    proof_is_correct_all = []
    for i in range(n):
        depth = depths[i]

        if stance_indication_method == StanceIndicationMethod.NEGATED_HYPOTHESIS_BY_I_DONT_THINK:
            if y_pred[i] == 0:
                proof_pred = results[2 * i]["proof_pred"]
            else:
                proof_pred = results[2 * i + 1]["proof_pred"]
        elif stance_indication_method == StanceIndicationMethod.STANCE_MARKER_IN_PROOF:
                proof_pred = results[i]["proof_pred"]

        if y[i] == 2:  # UNKNOWN
            _proof_is_correct = y_pred[i] == 2

            for score_name, score in _calc_ruletaker_rouge_score(proof_pred, ['dummy']).items():
                proof_scores[depth][score_name].append(1.0)
        else:
            gold_proofs = all_proofs[i]

            _proof_is_correct = _check_ruletaker_proof(proof_pred,
                                                       gold_proofs,
                                                       allowed_additional_proof_steps=allowed_additional_proof_steps,
                                                       no_similarity_threshold=no_similarity_threshold)

            for score_name, score in _calc_ruletaker_rouge_score(proof_pred, gold_proofs).items():
                proof_scores[depth][score_name].append(score)

        proof_is_correct[depth].append(_proof_is_correct)
        proof_is_correct_all.append(_proof_is_correct)

    proof_accuracies_agg = {depth: np.mean(correct_or_nots) for depth, correct_or_nots in proof_is_correct.items()}
    proof_accuracies_agg["overall"] = np.mean(
        list(itertools.chain.from_iterable(proof_is_correct.values()))
    )
    proof_accuracies_agg["overall_wo_None"] = np.mean(
        list(itertools.chain.from_iterable([val for key, val in proof_is_correct.items() if key is not None]))
    )
    proof_accuracies_agg["overall_macro"] = np.mean([score for depth, score in proof_accuracies_agg.items()
                                                    if not str(depth).startswith('overall')])
    proof_accuracies_agg["overall_macro_wo_None"] = np.mean([score for depth, score in proof_accuracies_agg.items()
                                                             if not str(depth).startswith('overall') and str(depth) != 'None'])



    score_names = list(list(proof_scores.values())[0].keys()) if len(proof_scores) > 0 else []
    proof_scores_agg = {
        depth: {
            score_name: np.mean(depth_scores[score_name])
            for score_name in score_names
        }
        for depth, depth_scores in proof_scores.items()
    }
    proof_scores_agg["overall"] = {
        score_name: np.mean(
            list(itertools.chain.from_iterable(proof_scores[depth][score_name] for depth in proof_scores.keys()))
        )
        for score_name in score_names
    }
    proof_scores_agg["overall_wo_None"] = {
        score_name: np.mean(
            list(itertools.chain.from_iterable(proof_scores[depth][score_name] for depth in proof_scores.keys() if depth is not None))
        )
        for score_name in score_names
    }
    proof_scores_agg["overall_macro"] = {
        score_name: np.mean([proof_scores_agg[depth][score_name] for depth in proof_scores_agg.keys()
                             if not str(depth).startswith('overall')])
        for score_name in score_names
    }
    proof_scores_agg["overall_macro_wo_None"] = {
        score_name: np.mean([proof_scores_agg[depth][score_name] for depth in proof_scores_agg.keys()
                             if not str(depth).startswith('overall') and str(depth) != 'None'])
        for score_name in score_names
    }



    return answer_accuracies_agg, proof_accuracies_agg, proof_scores_agg, proof_is_correct_all


def evaluate_ruletaker(
    results_val: List[Any],
    stance_indication_method: StanceIndicationMethod,
    allowed_additional_proof_steps=0,
    no_similarity_threshold=False,
    results_test: Optional[List[Any]] = None
) -> Any:
    """
    Evaluate on RuleTaker.
    """
    def line_head(head_text: str) -> str:
        _head_text = head_text
        return f'{_head_text:<30}: '

    def get_single_result_log_lines(y_pred: int, y: int, proof_is_correct: bool, result: Dict) -> List[str]:
        log_lines = []

        log_lines.append(f'{line_head("hypothesis")}{result["hypothesis"]}')

        log_lines.append('')
        log_lines.append(f'{line_head("context")}')
        for sent_id, sent in sorted(result['context'].items(), key = lambda id_sent: int(id_sent[0][len(SENT_IDENT):])):
            log_lines.append(f'    {sent_id}: {sent}')

        for i_gold, gold_proof in enumerate(result["all_proofs"]):
            log_lines.append('')
            log_lines.append(f'{line_head("gold_proof-" + str(i_gold))}')
            log_lines.append(prettify_proof_text(gold_proof, indent_level=1))

        log_lines.append('')
        log_lines.append(f'{line_head("proof_gt")}')
        log_lines.append(prettify_proof_text(result["proof_gt"], indent_level=1))

        log_lines.append('')
        log_lines.append(f'{line_head("proof_pred")}')
        log_lines.append(prettify_proof_text(result["proof_pred"], indent_level=1))

        log_lines.append('')
        log_lines.append(f'{line_head("depth")}{result["depth"]}')
        log_lines.append(f'{line_head("generation score")}{result["score"]}')

        answer = int2answer(y_pred) if y_pred != -1 else "INVALID_ANSWER(-1)"
        log_lines.append('')
        log_lines.append(f'{line_head("gold_answer")}{int2answer(y)}')
        log_lines.append(f'{line_head("pred_answer")}{answer}')
        log_lines.append(f'{line_head("answer is correct")}{answer == result["answer"]}')
        log_lines.append(f'{line_head("proof is correct")}{proof_is_correct}')

        return log_lines

    scores_val, labels_val, depths_val, all_proofs_val = process_ruletaker_results(results_val, stance_indication_method)

    y_val = np.array(labels_val, dtype=np.int64)
    if stance_indication_method == StanceIndicationMethod.NEGATED_HYPOTHESIS_BY_I_DONT_THINK:
        scaler = StandardScaler()
        X_val = scaler.fit_transform(np.array(scores_val))

        if len(set(y_val)) >= 2:
            clf = LogisticRegression()
            clf.fit(X_val, y_val)

            def predict(X):
                return clf.predict(X)
        else:
            unique_label = y_val[0]
            def predict(X):
                return np.array([unique_label] * len(X))
        y_val_pred = predict(X_val)
    elif stance_indication_method == StanceIndicationMethod.STANCE_MARKER_IN_PROOF:
        y_val_pred = np.array([_scores[0] for _scores in scores_val], dtype=np.int)
    else:
        raise ValueError()

    answer_accuracies_val, proof_accuracies_val, proof_scores_val, proof_is_correct_all_val = _calculate_ruletaker_metrics(
        y_val_pred, y_val, depths_val, all_proofs_val, results_val, stance_indication_method,
        allowed_additional_proof_steps=allowed_additional_proof_steps,
        no_similarity_threshold=no_similarity_threshold,
    )

    for y_pred, y, result, proof_is_correct in zip(y_val_pred, y_val, results_val, proof_is_correct_all_val):
        log_lines = []
        log_lines.append('\n\n\n------ evaluate_ruletaker (valid) --------')
        log_lines.extend(get_single_result_log_lines(y_pred, y, proof_is_correct, result))
        logger.info('\n'.join(log_lines))

    if results_test is None:
        return answer_accuracies_val, proof_accuracies_val, proof_scores_val

    scores_test, labels_test, depths_test, all_proofs_test = process_ruletaker_results(
        results_test,
        stance_indication_method,
    )
    X_test = scaler.transform(np.array(scores_test))
    y_test = np.array(labels_test, dtype=np.int64)
    y_test_pred = predict(X_test)
    answer_accuracies_test, proof_accuracies_test, proof_scores_test, proof_is_correct_all_test = _calculate_ruletaker_metrics(
        y_test_pred, y_test, depths_test, all_proofs_test, results_test, stance_indication_method,
        allowed_additional_proof_steps=allowed_additional_proof_steps,
        no_similarity_threshold=no_similarity_threshold,
    )

    for y_pred, y, result, proof_is_correct in zip(y_test_pred, y_test, results_test, proof_is_correct_all_test):
        log_lines = []
        log_lines.append('\n\n\n------ evaluate_ruletaker (test) --------')
        log_lines.extend(get_single_result_log_lines(y_pred, y, proof_is_correct, result))
        logger.info('\n'.join(log_lines))

    return (
        answer_accuracies_val,
        proof_accuracies_val,
        proof_scores_val,
        answer_accuracies_test,
        proof_accuracies_test,
        proof_scores_test,
    )
