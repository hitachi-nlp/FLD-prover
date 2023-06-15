from copy import deepcopy
from enum import Enum
import re
import random
import unicodedata
from typing import Optional, List, Dict, OrderedDict, Tuple, Optional, Set
from common import Answer
import logging

from ete3 import Tree, TreeNode


SENT_IDENT = 'sent'
VOID_IDENT = 'void'
INT_IDENT = 'int'
HYPOTHESIS_IDENT = 'hypothesis'
ASSUMP_IDENT = 'assump'


logger = logging.getLogger(__name__)


class NodeType(Enum):
    sent = 'sentence'
    int = 'internal'
    assump = 'assumption'
    assump_deletion = 'assumption_deletion'
    void = 'void'
    hypothesis = 'hypothesis'


def get_node_type(rep: str) -> Optional[NodeType]:
    if rep.strip().startswith('sent'):
        return NodeType.sent
    elif rep.strip().startswith('int'):
        return NodeType.int
    elif rep.strip().startswith('assump'):
        return NodeType.assump
    elif rep.strip().startswith('[assump'):
        return NodeType.assump_deletion
    elif rep.strip().startswith('void'):
        return NodeType.void
    elif rep.strip().startswith('hypothesis'):
        return NodeType.hypothesis
    else:
        return None


def extract_ident(rep: str, allow_sentence=False) -> Optional[str]:
    if allow_sentence:
        m = re.match(r"(?P<ident>(sent\d+[: ]|int\d+[: ]|assump\d+[: ]|\[assump\d+\][: ]|void[: ]|hypothesis[; ]))", rep)
    else:
        m = re.fullmatch(r"(?P<ident>(sent\d+[: ]|int\d+[: ]|assump\d+[: ]|\[assump\d+\][: ]|void[: ]|hypothesis[; ]))", rep)
    if m is None:
        return None
    return m["ident"][:-1]


def extract_idents(rep: str) -> List[str]:
    idents = []
    for ident in re.findall(r'(sent\d+[: ]|int\d+[: ]|assump\d+[: ]|\[assump\d+\][: ]|void[: ]|hypothesis[; ])', rep):
        idents.append(ident[:-1])
    return idents


def extract_ident_sent(rep: str) -> Optional[Tuple[str, str]]:
    ident_sents = extract_ident_sents(rep)
    if len(ident_sents) == 0:
        return None
    elif len(ident_sents) == 1:
        return list(ident_sents.items())[0]
    else:
        raise ValueError(f'Multiple idents / sentence found in "{rep}"')


def extract_ident_sents(rep: str) -> Dict[str, str]:
    ident_sents: Dict[str, str] = {}

    ident_matches = [m for m in re.finditer(r"(sent\d*: |int\d*: |assump\d*: )", rep)]
    is_proof_rep = rep.find(' -> ') >= 0

    for i_match, match in enumerate(ident_matches):
        if is_proof_rep:
            end = match.end() + rep[match.end():].find(';')
        else:
            if len(ident_matches) > i_match + 1:
                next_match = ident_matches[i_match + 1]
                end = next_match.start()
            else:
                end = len(rep)

        ident = match.group()[:-2]  # strip ":"
        sent = rep[match.end():end]
        # ident_sents[ident] = re.sub(' *;* *$', '', re.sub('^ *', '', sent))
        ident_sents[ident] = sent.rstrip(' ')

    return ident_sents


def extract_steps(proof: str) -> List[str]:
    steps = re.sub(' *; *$', '', proof).split(';')
    return [step.strip(' ') for step in steps]


def extract_premise_concl(step: str) -> Tuple[str, str]:
    return step.split(' -> ')
            

def extract_premises_concl(step: str) -> Tuple[List[str], str]:
    premise, concl = extract_premise_concl(step)
    premises = premise.split(' & ')
    return premises, concl


def get_lowest_vacant_int_id(text: str) -> Optional[int]:
    idents = extract_idents(text)
    int_idents = [ident for ident in idents
                  if re.match(f'^{INT_IDENT}[0-9]+$', ident)]
    int_idxs = [int(int_id[len(INT_IDENT):]) for int_id in int_idents]

    return max(int_idxs) + 1 if len(int_idxs) > 0 else 1


def is_valid_premise(rep: str) -> bool:
    return re.fullmatch(r"(sent\d+|int\d+|assump\d+|\[assump\d+\]|void)", rep)


def normalize(text: str) -> str:
    """
    Deal with unicode-related artifacts.
    """
    return unicodedata.normalize("NFD", text)


def normalize_sentence(text: str, no_lower=True) -> str:
    """
    Convert sentences to lowercase and remove the trailing period.
    """
    text = normalize(text)
    if not no_lower:
        text = text.lower()
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def extract_context(ctx: str, no_lower=True) -> OrderedDict[str, str]:
    """
    Extract supporting facts from string to dict.
    """
    return OrderedDict(
        {
            ident.strip(): normalize_sentence(sent, no_lower=no_lower)
            for ident, sent in re.findall(
                r"(?P<ident>sent\d+): (?P<sent>.+?) (?=sent\d+)", ctx + " sent999"
            )
        }
    )


def extract_assumptions(proof_text: str, no_lower=True) -> OrderedDict[str, str]:
    assumptions = OrderedDict()
    for m_begin in re.finditer(r'-> *assump\d+', proof_text):
        begin = m_begin.span()[0]
        assump_text = re.sub(r' *-> *', '', proof_text[begin:].split(';')[0])
        ident, sent = re.split(r':  *', assump_text)
        assumptions[ident.strip()] = normalize_sentence(sent, no_lower=no_lower)
    return assumptions


def deserialize(
    hypothesis: str,
    context: OrderedDict[str, str],
    proof: str,
    assumptions: Optional[OrderedDict[str, str]] = None,
    strict: bool = True
) -> Tree:
    """
    Construct a tree from a text sequence.
    """
    from stance_indication import delete_stance_markers
    proof = delete_stance_markers(proof)

    context_and_assumptions = deepcopy(context)
    if assumptions is not None:
        context_and_assumptions.update(assumptions)

    nodes = {}

    for proof_step in proof.split(";"):
        proof_step = proof_step.strip()
        if proof_step == "":
            continue

        if proof_step.count(" -> ") != 1:
            return None
        premises_txt, conclusion_txt = proof_step.split(" -> ")
        m = re.fullmatch(r"\((?P<score>.+?)\) (?P<concl>.+)", conclusion_txt)
        score: Optional[str]
        if m is not None:
            score = m["score"]
            conclusion_txt = m["concl"]
        else:
            score = None

        if conclusion_txt == "hypothesis":
            conclusion_ident = "hypothesis"
            conclusion_sent = hypothesis
        else:
            m = re.match(r"(?P<ident>(int|assump)\d+): (?P<sent>.+)", conclusion_txt)
            if m is None:
                return None
            conclusion_ident = m["ident"]
            conclusion_sent = m["sent"]
        conclusion_node = TreeNode(name=conclusion_ident)

        nodes[conclusion_ident] = conclusion_node
        nodes[conclusion_ident].add_feature("sent", conclusion_sent)
        nodes[conclusion_ident].add_feature("assump_children", [])
        if score is not None:
            nodes[conclusion_ident].add_feature("score", score)

        if get_node_type(premises_txt) == NodeType.void:
            pass
        else:
            for premise_ident in premises_txt.split(" & "):
                if premise_ident == conclusion_ident:
                    return None

                if premise_ident in nodes:
                    premise_node = nodes[premise_ident]
                else:
                    if get_node_type(premise_ident) not in [NodeType.sent,
                                                 NodeType.assump,
                                                 NodeType.assump_deletion]:
                        # non-leaf nodes should have been already added to nodes at the conclusion code block above
                        return None
                    premise_node = TreeNode(name=premise_ident)
                    premise_sent = context_and_assumptions.get(premise_ident.lstrip('[').rstrip(']'), None)
                    if premise_sent is None and strict:
                        return None
                    premise_node.add_feature("sent", premise_sent,)
                    nodes[premise_ident] = premise_node

                if get_node_type(premise_ident) == NodeType.assump_deletion:
                    conclusion_node.assump_children.append(premise_node)
                else:
                    if premise_node not in conclusion_node.children:
                        conclusion_node.add_child(premise_node)

    return nodes.get("hypothesis", None)


def serialize(tree: Tree) -> str:
    """
    Serialize a proof tree as a text sequence.
    """
    if tree is None:
        return "INVALID"

    elif tree.is_leaf():

        if get_node_type(tree.name) == NodeType.assump:
            return f'void -> {tree.name}'
        else:
            return tree.name  # type: ignore

    else:
        prev_steps = [
            serialize(child)
            for child in tree.children if not child.is_leaf()
        ]
        random.shuffle(prev_steps)

        if len(prev_steps) == 0:
            premise_seq = " & ".join([child.name for child in tree.children] + [child.name for child in tree.assump_children])
        else:
            premise_seq = (
                " ".join(prev_steps)
                + " "
                + " & ".join([child.name for child in tree.children] + [child.name for child in tree.assump_children])
            )

        if get_node_type(tree.name) == NodeType.hypothesis:
            conclusion_seq = "hypothesis;"
        else:
            conclusion_seq = f"{tree.name}: {tree.sent};"

        return " -> ".join([premise_seq, conclusion_seq])



def rename_idents(proof: str, assert_on_duplicated_ident=True) -> str:
    """
    Rename the `int\d+` and `assump\d+` identifiers in a proof so that they increase from 1.
    """
    renamed_proof = proof
    renamed_proof = _rename_idents(renamed_proof, 'int', assert_on_duplicated_ident=assert_on_duplicated_ident)
    renamed_proof = _rename_idents(renamed_proof, 'assump', assert_on_duplicated_ident=assert_on_duplicated_ident)
    return renamed_proof


def _rename_idents(proof: str, ident_prefix: str, assert_on_duplicated_ident=True) -> str:
    from src.prover.proof import InvalidProof
    proof_org = proof

    # print('=============== _rename_idents() ==============')
    # print('proof:', proof)
    # print('ident_prefix:', ident_prefix)

    assert "HOGEFUGAPIYO" not in proof
    mapping: Dict[str, str] = dict()

    while True:
        # print('------------- while ------------')
        m = re.search(f"{ident_prefix}\d+", proof)
        if m is None:
            break
        s = m.group()
        if assert_on_duplicated_ident:
            assert s not in mapping
        else:
            if s in mapping:
                raise InvalidProof(f'Identifier "{s}" might be duplicated in "{proof_org}"')
        # try:
        #     assert s not in mapping
        # except:
        #     from pprint import pprint
        #     print('proof:', proof)
        #     print('s:', s)
        #     print('mapping:')
        #     pprint(mapping)
        #     raise
        dst = f"HOGEFUGAPIYO{1 + len(mapping)}"
        mapping[s] = dst
        proof = proof.replace(f"{s}:", f"{dst}:").replace(f"{s} ", f"{dst} ")

    return proof.replace("HOGEFUGAPIYO", ident_prefix)


def collapse_proof(context: str,
                   proof: str,
                   add_hypothesis_at_end=False,
                   conclude_hypothesis_from_random_sent_id_if_no_proof=False) -> Tuple[str, str, List[str]]:

    steps = [
        step.strip(' ')
        for step in proof.split(';')
        if step.strip(' ') != ''
    ]

    premise_to_step: Dict[str, str] = {}
    premises: List[str] = []
    for step in steps:
        step_premises = step.split(' -> ')[0].split(' & ')
        for step_premise in step_premises:
            premise_to_step[step_premise] = step
            if step_premise not in premises:
                premises.append(step_premise)

    premise_sent_ids = [
        premise for premise in list(premises)
        if premise.startswith(SENT_IDENT)
    ]
    # dead_ids = random.sample(premise_sent_ids, 1)
    dead_ids = [premise_sent_ids[-1]]  # to avoid not short proofs to dominate, we use the longest subproof
    dead_steps: List[str] = []
    final_concl_id: Optional[str] = None
    for step in steps:
        step_premises = step.split(' -> ')[0].split(' & ')

        step_concl = step.split(' -> ')[1]
        if step_concl == 'hypothesis':
            step_concl_id = 'hypothesis'
        else:
            step_concl_id = re.findall(INT_IDENT + '[0-9]+', step_concl)[0]

        if any(step_premise in dead_ids for step_premise in step_premises):
            dead_steps.append(step)
            dead_ids.append(step_concl_id)
        else:
            final_concl_id = step_concl_id

    alive_steps = [step for step in steps if step not in dead_steps]
    if len(alive_steps) == 0:
        collapsed_proof = None
    else:
        collapsed_proof = '; '.join(alive_steps) + ';'

    collapsed_context = context
    for dead_id in dead_ids:
        if not dead_id.startswith(SENT_IDENT):
            continue
        collapsed_context = re.sub(f'{dead_id}:((?!sent[0-9]+).)*', '', collapsed_context)

    # -- fill the vacant sentence ids --
    context_sent_ids = sorted(set(re.findall(f'{SENT_IDENT}[0-9]+', context)))
    sent_id_map: Dict[str, str] = {}
    idx = 1
    for sent_id in context_sent_ids:
        if sent_id in dead_ids:
            continue
        sent_id_map[sent_id] = f'{SENT_IDENT}{idx}'
        idx += 1

    for src_sent_id, tgt_sent_id in sent_id_map.items():
        if collapsed_proof is not None:
            collapsed_proof = re.sub(f'{src_sent_id}([^0-9])', f'{tgt_sent_id.upper()}\g<1>', collapsed_proof)
        collapsed_context = re.sub(f'{src_sent_id}([^0-9])', f'{tgt_sent_id.upper()}\g<1>', collapsed_context)

    for tgt_sent_id in sent_id_map.values():
        if collapsed_proof is not None:
            collapsed_proof = re.sub(f'{tgt_sent_id.upper()}([^0-9])', f'{tgt_sent_id.lower()}\g<1>', collapsed_proof)
        collapsed_context = re.sub(f'{tgt_sent_id.upper()}([^0-9])', f'{tgt_sent_id.lower()}\g<1>', collapsed_context)

    # -- add hypothesis at end --
    if add_hypothesis_at_end:
        if collapsed_proof is None:
            if conclude_hypothesis_from_random_sent_id_if_no_proof:
                collapsed_proof = ''
                sent_ids = set(re.findall(f'{SENT_IDENT}[0-9]+', context))
                alive_send_ids = [sent_id for sent_id in list(sent_ids)
                                  if sent_id not in dead_ids]
                random_sent_id = random.choice(alive_send_ids)
                collapsed_proof = f'{random_sent_id} -> hypothesis;'
        else:
            final_step = collapsed_proof.rstrip('; ').split(';')[-1].strip(' ')
            final_concl = final_step.split(' -> ')[-1]

            final_concl_id = re.findall(f'{INT_IDENT}[0-9]+', final_concl)[0]
            collapsed_proof += f' {final_concl_id} -> hypothesis;'

    return collapsed_context, collapsed_proof, dead_ids


_NOT_INTRO = 'the following is not true: '


def add_final_reference(context: str,
                        hypothesis: str,
                        proof: str,
                        answer: Answer,
                        skip_if_already_has_final_reference=True,
                        dataset_depth: Optional[int] = None) -> str:
    from stance_indication import delete_stance_markers, get_stance_markers, add_stance_markers

    if proof.find(' -> hypothesis') < 0:  # proof is incomplete
        return proof

    markers = get_stance_markers(proof)
    proof = delete_stance_markers(proof)

    if skip_if_already_has_final_reference\
            and _is_proof_with_reference(context, hypothesis, proof, dataset_depth=dataset_depth):
        return add_stance_markers(proof, markers)

    int_ids = set(re.findall(f'{INT_IDENT}[0-9][0-9]*', proof))
    if len(int_ids) > 0:
        int_indexes = [int(int_id[len(INT_IDENT):]) for int_id in list(int_ids)]
        next_int_index = max(int_indexes) + 1
    else:
        next_int_index = 1
    next_int_id = f'{INT_IDENT}{str(next_int_index)}'

    reference_sentence = _make_reference_sentence(hypothesis, answer)
    proof_with_reference = proof.replace('-> hypothesis', f'-> {next_int_id}: {reference_sentence}')
    proof_with_reference = re.sub('; *$', '', proof_with_reference) + '; ' + f'{next_int_id} -> hypothesis;'

    proof_with_reference = add_stance_markers(proof_with_reference, markers)

    # for debugging
    # print('\n\n=================================     add_final_reference       =========================')
    # print('---------------- context ------------------')
    # print(context)

    # print('---------------- hypothesis ------------------')
    # print(hypothesis)

    # print('---------------- proof ------------------')
    # print(proof)

    # print('---------------- proof with reference------------------')
    # print(proof_with_reference)

    logger.info('-- Add reference step to proof --')
    logger.info('orig                  :   "%s"', proof)
    logger.info('with reference step   :   "%s"', proof_with_reference)

    return proof_with_reference


def _is_proof_with_reference(context: str,
                             hypothesis: str,
                             proof: str,
                             dataset_depth: Optional[int] = None):
    """ proof like sent1 -> hypothesis """
    if proof.find(' -> hypothesis') < 0:
        raise ValueError('The proof is incomplete: "%s"', proof)

    steps = re.sub('; *$', '', proof).split(';')

    if len(steps) == 1:
        if dataset_depth is None:
            raise ValueError(f"We can not determine whether the proof is with reference or not without dataset_depth, for step 1 proof '{proof}'")
        return dataset_depth == 0

    else:
        final_step = steps[-1]
        premise_str, _ = final_step.split(' -> ')
        premises = [rep.strip(' ') for rep in premise_str.split(' & ')]
        if len(premises) != 1:
            return False

        context_with_proof = context + '; ' + proof

        premise = premises[0]
        ident_to_sent = extract_ident_sents(context_with_proof)
        if premise not in ident_to_sent:
            raise ValueError(f'We can not determine the premise "{premise}" since we could not find partial proof from context_with_proof = "{context_with_proof}"')
        sent = ident_to_sent[premise]

        sent = sent.strip(' ')

        possible_reference_sents = [_make_reference_sentence(hypothesis, True),
                                    _make_reference_sentence(hypothesis, False)]

        return sent in possible_reference_sents


def _make_reference_sentence(hypothesis: str, answer: Answer) -> str:
    if answer is True:
        return hypothesis
    elif answer is False:
        return f'{_NOT_INTRO}{hypothesis}'
    elif answer == "Unknown":
        raise ValueError()
    else:
        raise ValueError()


def normalize_proof(proof_text: str) -> str:
    proof_text = re.sub('\n+', ' ', proof_text)
    proof_text = re.sub(r'\s+', ' ', proof_text)
    proof_text = re.sub(r'\s+$', '', re.sub(r'^\s+', '', proof_text))
    return proof_text
