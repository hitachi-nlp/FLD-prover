from enum import Enum
from copy import deepcopy
from typing import Dict, List, Any, Optional
from typing import TypedDict

from pydantic import BaseModel


class DeductionExample(BaseModel):
    hypothesis: str
    context: str

    proofs: Optional[List[str]] = None
    proof_stance: Optional[str] = None
    answer: Optional[str] = None
    original_tree_depth: Optional[int] = None
    depth: Optional[int] = None
    num_formula_distractors: Optional[int] = None
    num_translation_distractors: Optional[int] = None
    num_all_distractors: Optional[int] = None

    negative_hypothesis: Optional[str] = None
    negative_proofs: Optional[List[str]] = None
    negative_proof_stance: Optional[str] = None
    negative_answer: Optional[str] = None


class SerializedDeductionStep(BaseModel):
    input: str
    next_step: Optional[str] = None


class AnswerLabel(Enum):
    PROVED = 'PROVED'
    DISPROVED = 'DISPROVED'
    UNKNOWN = 'UNKNOWN'


class ProofStanceLabel(Enum):
    PROVED = 'PROVED'
    DISPROVED = 'DISPROVED'
    UNKNOWN = 'UNKNOWN'
