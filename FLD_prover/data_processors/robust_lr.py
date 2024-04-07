from typing import Optional, Any, Tuple, Dict
import logging

from FLD_task.proof import StanceMarker, add_stance_markers
from FLD_task.evaluation import compute_answer_accuracy
from FLD_prover.tokenization import unmask_by_pad_token

from .base import Processor

logger = logging.getLogger()


class RobustLRProcessor(Processor):

    def _make_in_out(
        self,
        example,
        split: str,
    ) -> Tuple[str, str, str]:
 
        facts, hypothesis, gold_proof = self._get_logic(example, 'train')

        prompt = ' ; '.join([
            '$facts$ = ' + example['context'],
            '$hypothesis$ = ' + example['statement'],
            '$proof$ = '
        ])
        prompt_with_partial_proof = self._prompt_prefix + prompt
        next_proof_step = gold_proof

        return prompt_with_partial_proof, next_proof_step, gold_proof

    def _compute_metrics(self, example, pred_proof: str):

        metrics = {}

        facts, hypothesis, gold_proof = self._get_logic(example, 'eval')

        _metrics = {
            'answer_accuracy': compute_answer_accuracy(gold_proof, pred_proof),
        }
        depths = (['all', str(example['depth'])] if example.get('depth', None) is not None
                  else ['all', 'None'])
        for depth in depths:
            for metric_name, metric_val in _metrics.items():
                metrics[f"D-{depth}.{metric_name}"] = metric_val

        return metrics

    def _get_features(self, examples) -> Dict[str, Any]:
        return {}

    def _get_logic(self, example, split: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        facts = example['context']
        hypothesis = example['statement']

        label = example['label']
        if label == 'entailment':
            marker = StanceMarker.PROVED
        elif label == 'neutral':
            marker = StanceMarker.UNKNOWN
        elif label == 'contradiction':
            marker = StanceMarker.DISPROVED
        else:
            raise ValueError()
        gold_proof = add_stance_markers('', [marker])

        return facts, hypothesis, gold_proof
