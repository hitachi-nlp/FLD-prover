from typing import Any, Optional, Tuple
import logging

from collections import defaultdict
from typing import List, Dict
from FLD_prover.tokenization import unmask_by_pad_token

from .base import Metrics


logger = logging.getLogger()


class PararulePlusMetrics(Metrics):

    def _compute_metrics_from_example(self, example, pred_proof: str) -> Dict[str, List[Any]]:

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        metrics: Dict[str, List[Any]] = defaultdict(list)

        facts, hypothesis, gold_proof = self._get_logic(example)

        _metrics = {
            'accuracy': 1.0 if gold_proof == pred_proof else 0.0,
        }
        depths = (['all', str(example['depth'])] if example.get('depth', None) is not None
                  else ['all', 'None'])
        for depth in depths:
            for metric_name, metric_val in _metrics.items():
                metrics[f"D-{depth}.{metric_name}"].append(metric_val)

        return metrics

    def _get_logic(self, example) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return (
            example['facts'],
            example['question'],
            example['gold_proof'],
        )
