from typing import Any, Optional, Tuple
import logging

from collections import defaultdict
from typing import List, Dict
from FLD_prover.tokenization import unmask_by_pad_token
from FLD_task.proof import StanceMarker, get_stance_markers

from .base import Metrics


logger = logging.getLogger()


class ProofWriterMetrics(Metrics):

    def _compute_metrics_from_example(self, example, pred_proof: str) -> Dict[str, List[Any]]:

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        facts, hypothesis, gold_proof = self._get_logic(example)

        pred_markers = get_stance_markers(pred_proof)
        gold_markers = get_stance_markers(gold_proof)

        answer_accuracy = 1.0 if set(pred_markers) == set(gold_markers) else 0.0
        if gold_markers == [StanceMarker.UNKNOWN] and answer_accuracy == 1.0:
            proof_accuracy = 1.0
        else:
            proof_accuracy = 1.0 if pred_proof == gold_proof else 0.0

        metrics = {
            'answer_accuracy': answer_accuracy,
            'proof_accuracy': proof_accuracy,
        }

        return metrics

    def _get_logic(self, example) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return (
            example['context'],
            example['question'],
            example['gold_proof'],
        )
