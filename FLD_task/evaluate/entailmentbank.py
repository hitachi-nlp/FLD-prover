"""
Utilities for evaluation.
"""
import logging
import argparse
import itertools
from collections import defaultdict
import json
import tempfile
import os
from pprint import pprint
from typing import DefaultDict, Set, Dict, Tuple, Any, Union, List, Optional

from tqdm import tqdm
from ete3 import TextFace, TreeStyle, NodeStyle
import datasets
from PyPDF3 import PdfFileWriter, PdfFileReader
from PyPDF3.pdf import PageObject
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer
from src.prover.proof import prettify_proof_text, InvalidProof
from common import (
    Example,
    I_DONT_THINK,
    answer2int,
    int2answer,
)
from proof_common import (
    extract_context,
    normalize,
    TreeNode,
    serialize,
    deserialize,
    get_node_type,
    NodeType,
)
from stance_indication import (
    StanceIndicationMethod,
    delete_stance_markers,
    marker_to_answer,
    get_stance_markers,
)


logger = logging.getLogger(__name__)


def _gather_descendants(tree: TreeNode) -> DefaultDict[str, Set[str]]:
    descendants = defaultdict(set)

    for node in tree.traverse("postorder"):
        if get_node_type(node.name) == NodeType.sent:
            descendants[node.name].add(node.name)
        for child in node.children:
            descendants[node.name].update(descendants[child.name])

    return descendants


def _intersection_over_union(a: Set[Any], b: Set[Any]) -> float:
    return len(a.intersection(b)) / len(a.union(b))


def _align(tree_pred: TreeNode, tree_gt: TreeNode) -> Dict[str, str]:
    """
    Align nodes in the predicted tree to nodes in the ground truth tree.
    """
    alignment: Dict[str, str] = {}
    if tree_pred is None:
        return alignment

    descendants_pred = _gather_descendants(tree_pred)
    descendants_gt = _gather_descendants(tree_gt)

    for node in tree_pred.traverse():
        if get_node_type(node.name) == NodeType.sent:
            alignment[node.name] = (
                node.name if tree_gt.get_leaves_by_name(node.name) != [] else "dummy"
            )
        else:
            max_iou = 0.0
            max_node = None
            for node_gt in tree_gt.traverse():
                if get_node_type(node.name) == NodeType.sent:
                    continue
                iou = _intersection_over_union(
                    descendants_gt[node_gt.name], descendants_pred[node.name]
                )
                if iou > max_iou:
                    max_iou = iou
                    max_node = node_gt
            alignment[node.name] = max_node.name if max_node is not None else "dummy"

    return alignment


def _calculate_f1(tp: float, fp: float, p: float) -> float:
    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if p == 0 else tp / p
    return 0.0 if prec + rec == 0.0 else 2 * prec * rec / (prec + rec)


def evaluate_leaves(tree_pred: TreeNode, tree_gt: TreeNode) -> Tuple[float, float]:
    if tree_pred is None:
        return 0.0, 0.0

    sents_gt = {node.name for node in tree_gt.get_leaves()}

    sents_pred = set()
    for node in tree_pred.get_leaves():
        if node.name not in sents_gt:
            node.add_feature("error", "leaf")
        sents_pred.add(node.name)

    tp = len(sents_pred.intersection(sents_gt))
    fp = len(sents_pred.difference(sents_gt))
    fn = len(sents_gt.difference(sents_pred))
    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0.0 else 0.0
    em = float(f1 == 1.0)

    return em, f1


def _evaluate_steps(
    tree_pred: TreeNode, tree_gt: TreeNode, alignment: Dict[str, str]
) -> Tuple[float, float]:
    if tree_pred is None:
        return 0.0, 0.0

    steps_gt = set()
    for node in tree_gt.traverse():
        if node.is_leaf():
            continue
        steps_gt.add(
            (tuple(sorted([child.name for child in node.children])), node.name)
        )

    steps_pred = set()
    for node in tree_pred.traverse():
        if node.is_leaf():
            continue
        step = (
            tuple(sorted([alignment[child.name] for child in node.children])),
            alignment[node.name],
        )
        steps_pred.add(step)
        if step not in steps_gt:
            node.add_feature("error", "step")

    tp = len(steps_pred.intersection(steps_gt))
    fp = len(steps_pred.difference(steps_gt))
    fn = len(steps_gt.difference(steps_pred))
    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0.0 else 0.0
    em = float(f1 == 1.0)

    return em, f1


def _evaluate_intermediates(
    tree_pred: TreeNode, tree_gt: TreeNode, alignment: Dict[str, str], bleurt: Any
) -> Tuple[float, float]:
    if tree_pred is None:
        return 0.0, 0.0

    ints_gt = {node.name for node in tree_gt.traverse() if not node.is_leaf()}

    tp = fp = 0
    nodes_pred = []
    sents_pred = []
    nodes_gt = []
    sents_gt = []

    for node in tree_pred.traverse():
        if node.is_leaf():
            continue
        if alignment[node.name] == "dummy":
            fp += 1
            node.add_feature("error", "intermediate")
        else:
            nodes_pred.append(node)
            nodes_gt.append(tree_gt.search_nodes(name=alignment[node.name])[0])
            sents_pred.append(node.sent)
            sents_gt.append(nodes_gt[-1].sent)

    similarities = bleurt.compute(predictions=sents_pred, references=sents_gt)["scores"]

    ints = set()

    for k, s in enumerate(similarities):
        if s >= 0.28:
            tp += 1
            ints.add(nodes_gt[k].name)
        else:
            fp += 1
            nodes_pred[k].add_feature("error", "intermediate")

    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if len(ints) == 0 else len(ints) / len(ints_gt)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0.0 else 0.0
    em = float(f1 == 1.0)

    return em, f1


def _highlight_errors(tree: TreeNode) -> None:
    "Highlight errors in the tree."
    for node in tree.traverse():
        name = TextFace(node.name)
        sent = TextFace(node.sent)
        if hasattr(node, "error"):
            if node.error == "step":
                node.set_style(NodeStyle(vt_line_width=10, vt_line_color="LightSalmon"))
                for child in node.children:
                    child.set_style(
                        NodeStyle(hz_line_width=10, hz_line_color="LightSalmon")
                    )
            else:
                sent.background.color = name.background.color = "LightSalmon"
        node.add_face(name, column=0)
        node.add_face(sent, column=0)
        if hasattr(node, "score"):
            node.add_face(TextFace(node.score), column=0)


def _get_tree_style(proof: str, score: Optional[float], is_gt: bool) -> TreeStyle:
    style = TreeStyle()
    style.branch_vertical_margin = 100
    style.show_leaf_name = False
    style.show_scale = False
    style.title.add_face(TextFace("Ground truth" if is_gt else "Predicted"), column=0)
    style.title.add_face(TextFace(proof), column=0)
    if score is not None:
        style.title.add_face(TextFace(f"Verifier score: {score}"), column=0)
    return style


def _evaluate_entailmentbank_example(ex: Example, bleurt: Any, output_pdf: bool) -> Any:
    hypothesis = normalize(ex["hypothesis"]).strip()
    context = ex["context"]

    proof_gt = normalize(delete_stance_markers(ex["proof_gt"])).strip()
    tree_gt = deserialize(hypothesis, context, proof_gt, strict=False)
    proof_gt = serialize(tree_gt)

    proof_pred = delete_stance_markers(ex["proof_pred"])

    assert tree_gt is not None
    em = {}
    f1 = {}

    with tempfile.TemporaryDirectory() as dir_path:
        if output_pdf:
            _highlight_errors(tree_gt)
            file_path = os.path.join(dir_path, "tree_gt.pdf")
            style = _get_tree_style(
                proof_gt, ex.get("verifier_score_gt", None), is_gt=True
            )
            tree_gt.render(file_path, tree_style=style)
            pdf_pages = [PdfFileReader(open(file_path, "rb")).getPage(0)]

        tree_pred = deserialize(hypothesis, context, proof_pred)
        proof_pred = serialize(tree_pred)
        alignment = _align(tree_pred, tree_gt)

        em_leaves, f1_leaves = evaluate_leaves(tree_pred, tree_gt)
        em["leaves"] = em_leaves
        f1["leaves"] = f1_leaves

        em_steps, f1_steps = _evaluate_steps(tree_pred, tree_gt, alignment)
        em["steps"] = em_steps
        f1["steps"] = f1_steps

        if bleurt is not None:
            em_intermediates, f1_intermediates = _evaluate_intermediates(
                tree_pred, tree_gt, alignment, bleurt
            )
            em["intermediates"] = em_intermediates
            f1["intermediates"] = f1_intermediates

        correct = (em["leaves"] == 1.0) and (em["steps"] == 1.0)
        if bleurt is not None:
            correct = correct and (em["intermediates"] == 1.0)
        em["proof"] = f1["proof"] = 1.0 if correct else 0.0

        if output_pdf and tree_pred is not None:
            _highlight_errors(tree_pred)
            file_path = os.path.join(dir_path, f"tree_pred.pdf")
            style = _get_tree_style(
                proof_pred,
                ex["verifier_scores_pred"] if "verifier_scores_pred" in ex else None,
                is_gt=False,
            )
            tree_pred.render(file_path, tree_style=style)
            pdf_pages.append(PdfFileReader(file_path).getPage(0))

    tree_depth = int(tree_gt.get_farthest_leaf()[1])
    tree_size = 1 + len(tree_gt.get_descendants())

    if output_pdf:
        margin = 50
        total_width = np.max([page.mediaBox.upperRight[0] for page in pdf_pages])
        total_height = np.sum(
            [page.mediaBox.upperRight[1] + margin for page in pdf_pages]
        )
        combined_page = PageObject.createBlankPage(None, total_width, total_height)
        offset = 0
        for page in pdf_pages[::-1]:
            combined_page.mergeTranslatedPage(page, 0, offset)
            offset += page.mediaBox.upperRight[1] + margin
        return em, f1, tree_depth, tree_size, combined_page
    else:
        return em, f1, tree_depth, tree_size, None


def evaluate_entailmentbank(
    results: List[Any],
    eval_intermediates: bool = True,
    output_pdf: Optional[str] = None,
) -> Any:
    """
    Evaluate predicted proof trees on EntailmentBank.

    The implementation has minor differences from EntailmentBank's official evaluation code.
    DO NOT use it for reporting results in papers. Use the official code instead.
    """
    # eval_intermediates = False may results in the overestimates of the score.
    measures = ["leaves", "steps", "proof"]
    if eval_intermediates:
        measures.append("intermediates")
    em = defaultdict(list)
    f1 = defaultdict(list)
    depth = []
    size = []
    bleurt = (
        datasets.load_metric("bleurt", "bleurt-large-512")
        if eval_intermediates
        else None
    )
    if output_pdf is not None:
        pdf_oup = PdfFileWriter()

    for ex in tqdm(results):
        em_i, f1_i, depth_i, size_i, page = _evaluate_entailmentbank_example(
            ex, bleurt, output_pdf is not None,
        )

        depth.append(depth_i)
        size.append(size_i)
        for m in measures:
            em[m].append(em_i[m])
            f1[m].append(f1_i[m])

        if output_pdf:
            pdf_oup.addPage(page)

    if output_pdf is not None:
        pdf_oup.write(open(output_pdf, "wb"))

    depth_arr = np.array(depth)
    print("Performance by depth:")
    for d in np.unique(depth_arr):
        mask = depth_arr == d
        n = mask.sum()
        print(f"{n} trees have depth {d}")
        for m in measures:
            print(
                f"\t{m}: {(np.array(em[m]) * mask).sum() / n}\t{(np.array(f1[m]) * mask).sum() / n}"
            )

    size_arr = np.array(size)
    print("Performance by size:")
    for s in np.unique(size_arr):
        mask = size_arr == s
        n = mask.sum()
        print(f"{n} trees have size {s}")
        for m in measures:
            print(
                f"\t{m}: {(np.array(em[m]) * mask).sum() / n}\t{(np.array(f1[m]) * mask).sum() / n}"
            )

    return (
        {m: np.mean(values) for m, values in em.items()},
        {m: np.mean(values) for m, values in f1.items()},
    )
