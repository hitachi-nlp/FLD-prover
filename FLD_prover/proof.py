"""
Proof steps and proof trees.
"""

from typing import List, Union, OrderedDict, Optional, Tuple, Dict
import logging
import itertools
import re
import random


from proof_common import(
    extract_context,
    NodeType, 
    normalize,
    Tree,
    extract_assumptions,
    is_valid_premise,
    extract_ident_sent,
    deserialize,
    get_node_type,
    SENT_IDENT,
    INT_IDENT,
    ASSUMP_IDENT,
)
from stance_indication import delete_stance_markers, get_stance_markers, StanceMarker

logger = logging.getLogger(__name__)


class InvalidProof(Exception):
    pass


class InvalidProofStep(Exception):
    pass


class ProofStep:

    def __init__(self,
                 proof: "Proof",
                 step_text: str,
                 stance_markers: List[StanceMarker],
                 strict: bool) -> None:

        if len(get_stance_markers(step_text)) > 0:
            # we require callers deleting the markers for explicitness.
            raise ValueError(f'Text with stance markers ("{step_text}") is not allowed for ProofStep().'
                             ' Delete the markers, and input the resulting text and the markers into the constructor as ProofStep(proof, text_wo_markers, markers).')
        self.stance_markers = stance_markers

        self.proof = proof
        if step_text.count(" -> ") != 1:
            raise InvalidProofStep(step_text)
        premises, conclusion = step_text.split(" -> ")

        self.premise_idents = []
        self.premise_sents = []
        premises = premises.split(" & ")
        if premises == ['void']:
            pass
        else:
            for p in premises:
                if not is_valid_premise(p):
                    raise InvalidProofStep(step_text)
                self.premise_idents.append(p)
                try:
                    sent = proof.ident2sent(p)
                    self.premise_sents.append(sent)
                except KeyError:
                    # Unsatisfied premises.
                    raise InvalidProofStep(step_text)

        if conclusion == "hypothesis":
            self.conclusion_ident = "hypothesis"
            self.conclusion_sent = proof.hypothesis
        else:
            try:
                results = extract_ident_sent(conclusion)
            except ValueError:
                raise InvalidProofStep(step_text)

            if not results:
                raise InvalidProofStep(step_text)

            ident, sent = results

            if get_node_type(ident) not in [NodeType.int, NodeType.assump]:
                raise InvalidProofStep(step_text)

            self.conclusion_ident = ident
            self.conclusion_sent = sent
            if self.conclusion_sent == self.proof.hypothesis:
                # Intermediate conclusion identical with the hypothesis.
                pass
                # raise InvalidProofStep(step_text)  # HONOKA
            if strict and (
                self.conclusion_sent in self.proof.context.values()
                or any(
                    self.conclusion_sent == step.conclusion_sent
                    for step in self.proof.proof_steps
                )
            ):
                # Intermediate conclusion identical with premises or an existing intermediate conclusion.
                raise InvalidProofStep(step_text)

        if self.conclusion_ident in self.premise_idents:
            raise InvalidProofStep(step_text)

    @property
    def text(self) -> str:
        if self.conclusion_ident == "hypothesis":
            return f"{' & '.join(self.premise_idents)} -> hypothesis"
        else:
            return f"{' & '.join(self.premise_idents)} -> {self.conclusion_ident}: {self.conclusion_sent}"

    def __str__(self) -> str:
        if self.conclusion_ident == "hypothesis":
            return self.text + f" (with stance markers {str([marker.value for marker in self.stance_markers])})"
        else:
            return self.text

    def __repr__(self) -> str:
        return self.__str__()

    def is_final(self) -> bool:
        return self.conclusion_ident == "hypothesis"


class Proof:
    def __init__(
        self,
        context: Union[str, OrderedDict[str, str]],
        hypothesis: str,
        proof_text: str,
        stance_markers: List[StanceMarker],
        strict: bool,
        requires_complete: bool = False,
    ) -> None:
        if len(get_stance_markers(proof_text)) > 0:
            # we require callers deleting the markers for explicitness.
            raise ValueError(f'Text with stance markers ("{proof_text}") is not allowed for Proof().'
                             'Delete the markers, and input the resulting text and the markers into the constructor as ProofStep(proof, text_wo_markers, markers).')
        stance_markers = stance_markers

        self._context_text = context
        if isinstance(context, str):
            context = extract_context(context)
        self.context = context

        self.assumptions = extract_assumptions(proof_text)

        self.hypothesis = hypothesis
        self.strict = strict
        self.requires_complete = requires_complete

        proof_text = proof_text.strip()
        if proof_text.endswith(";"):
            proof_text = proof_text[:-1]
        self.proof_text = proof_text

        self.proof_steps = []
        step_texts = proof_text.split(";")
        for i_step, step_text in enumerate(step_texts):
            step_text = step_text.strip()
            if step_text == "":
                continue
            if i_step == len(step_texts) - 1:
                _stance_markers = stance_markers
            else:
                _stance_markers = []
            self.proof_steps.append(ProofStep(self,
                                              step_text,
                                              _stance_markers,
                                              strict))

        if requires_complete:
            assert self.is_complete()

    @property
    def stance_markers(self) -> List[StanceMarker]:
        # We decided not to check whether more than two steps have stance markers
        # since it is possible when the prover is under-trained.

        # find stance markers from the final step.
        # stance_markers = None
        # for step in self.proof_steps:
        #     if len(step.stance_markers) > 0:
        #         if stance_markers is not None:
        #             raise Exception('Multiple step have stance_markers')
        #         stance_markers = step.stance_markers
        # if stance_markers is None:
        #     return []
        # else:
        #     return stance_markers

        if len(self.proof_steps) > 0:
            return self.proof_steps[-1].stance_markers
        else:
            return []

    def __str__(self) -> str:
        content = self.proof_text + f' (with stance markers {str([marker.value for marker in self.stance_markers])})'
        return f'Proof<{content}>'

    def __repr__(self) -> str:
        return self.__str__()

    def __contains__(self, text: str) -> bool:
        text = delete_stance_markers(text)
        return text in self.context.values() or any(
            text == step.conclusion_sent for step in self.proof_steps
        )

    def is_empty(self) -> bool:
        return self.proof_text == ""

    def serialize_context(self) -> str:
        return normalize(" ".join(f"{k}: {v}" for k, v in self.context.items()))

    def execute(self, step: ProofStep) -> None:
        assert not self.is_complete()
        assert step.proof is self
        if len(self.proof_steps) > 0:
            self.proof_text += "; "
        self.proof_text += step.text
        self.proof_steps.append(step)

    def to_tree(self) -> Tree:
        return deserialize(self.hypothesis,
                           self.context,
                           self.proof_text,
                           assumptions=self.assumptions)

    def ident2sent(self, ident: str) -> str:
        node_type = get_node_type(ident)
        if node_type == NodeType.hypothesis:
            return self.hypothesis
        elif node_type == NodeType.sent:
            return self.context[ident]
        elif node_type == NodeType.assump:
            return self.assumptions[ident]
        elif node_type == NodeType.assump_deletion:
            return self.assumptions[ident.lstrip('[').rstrip(']')]
        elif node_type == NodeType.int:
            for step in self.proof_steps:
                if step.conclusion_ident == ident:
                    return step.conclusion_sent
            raise KeyError()
        else:
            raise KeyError()

    def shuffle_context(self, other_proofs: Optional[List["Proof"]] = None) -> Union["Proof", Tuple["Proof", List["Proof"]]]:
        """
        Randomly shuffle the identifiers of the supporting facts.
        """
        if other_proofs is not None:
            for other_proof in other_proofs:
                if len(other_proof.context) != len(self.context):
                    raise ValueError('Context does not match: "%s" and "%s"', str(self.context), str(other_proof.context))
                for key, val in self.context.items():
                    if val != other_proof.context[key]:
                        raise ValueError('Context does not match: "%s" and "%s"', str(self.context), str(other_proof.context))

        num_sents = len(self.context)
        permutation = list(range(num_sents))
        random.shuffle(permutation)
        inv_permutation = [permutation.index(i) for i in range(num_sents)]

        shuffled_context = " ".join(
            f"{SENT_IDENT}{i+1}: {self.context[f'{SENT_IDENT}{permutation[i]+1}']}"
            for i in range(num_sents)
        )

        def make_renamed_proof(proof: "Proof") -> str:
            tokens = []
            for t in proof.proof_text.split():
                if re.fullmatch(f"{SENT_IDENT}\d+", t):
                    i = int(t[4:])
                    tokens.append(f"{SENT_IDENT}{inv_permutation[i-1]+1}")
                else:
                    tokens.append(t)
            renamed_proof_text = " ".join(tokens)
            return Proof(
                shuffled_context,
                proof.hypothesis,
                renamed_proof_text,
                proof.stance_markers,
                proof.strict,
                proof.requires_complete,
            )

        renamed_proof = make_renamed_proof(self)

        if other_proofs is not None:
            renamed_other_proofs = [make_renamed_proof(other_proof) for other_proof in other_proofs]
            return renamed_proof, renamed_other_proofs
        else:
            return renamed_proof

    def _next_int(self) -> str:
        existing_ints = {step.conclusion_ident for step in self.proof_steps
                         if get_node_type(step.conclusion_ident) == NodeType.int}
        for n in itertools.count(1):
            ident = f"{INT_IDENT}{n}"
            if ident not in existing_ints:
                break
        return ident

    def _next_assump(self) -> str:
        existing_assumps = {step.conclusion_ident for step in self.proof_steps
                            if get_node_type(step.conclusion_ident) == NodeType.assump}
        for n in itertools.count(1):
            ident = f"{ASSUMP_IDENT}{n}"
            if ident not in existing_assumps:
                break
        return ident

    def add_number_to_step_conclusion(self, step: str) -> str:
        if f"-> {INT_IDENT}:" in step:
            step = step.replace(f"-> {INT_IDENT}:", f"-> {self._next_int()}:").strip()
        if f"-> {ASSUMP_IDENT}:" in step:
            step = step.replace(f"-> {ASSUMP_IDENT}:", f"-> {self._next_assump()}:").strip()
        return step

    def is_complete(self) -> bool:
        return self.proof_text.endswith("-> hypothesis")


def prettify_proof_text(proof_text: str, indent_level=0) -> str:
    stance_markers = get_stance_markers(proof_text)
    proof_text = delete_stance_markers(proof_text)

    pretty_lines = []
    proof_lines = proof_text.split('; ')
    for line in proof_lines:
        if line.find(' -> ') < 0:
            logger.info('Could not prettify the proof since the following line have no " -> ": "%s"', line)
            return proof_text

        implication_fields = line.split(' -> ')
        if len(implication_fields) > 2:
            logger.info('Could not prettify the proof since the following line have more than two " -> ": "%s"', line)
            return proof_text

        premises_text, concl_text = line.split(' -> ')

        premises = premises_text.split(' & ')
        pretty_premise_texts = []
        for premise_text in sorted(premises):
            pretty_premise_texts.append(f'{premise_text:>8}')
        pretty_premises_text = ''.join(pretty_premise_texts)

        if concl_text.find(': ') >= 0:
            concl_fields = concl_text.split(': ')
            if len(concl_fields) > 2:
                logger.info('Could not prettify the proof since the following line have more than two ": ": "%s"', line)
                return proof_text
            concl_sent_id, concl_sentence = concl_fields
            pretty_concl_text = f'{concl_sent_id:>10}:       {concl_sentence}'
        else:
            pretty_concl_text = f'{concl_text:>10}'

        pretty_line = ' ' * indent_level + f'{pretty_premises_text:<25} ->     {pretty_concl_text}'
        pretty_lines.append(pretty_line)

    stance_markers_text = f'=>    stance markers = {str([mk.value for mk in stance_markers])}'
    pretty_lines.append(' ' * indent_level + f'{stance_markers_text:>81}')

    return '\n'.join(pretty_lines)
