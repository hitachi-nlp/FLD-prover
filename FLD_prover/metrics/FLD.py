from typing import Any, Tuple, Optional
import logging

from collections import defaultdict
from typing import List, Dict
from FLD_prover.tokenization import unmask_by_pad_token
from FLD_task import build_metrics

from .base import Metrics

logger = logging.getLogger()


class FLDMetrics(Metrics):

    def _compute_metrics_from_example(self, example, pred_proof: str) -> Dict[str, List[Any]]:

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        metrics = {}

        metric_funcs = {
            'strct': build_metrics('strict'),
            'extr_stps': build_metrics('allow_extra_steps'),
        }

        facts, hypothesis, gold_proof = self._get_logic(example)

        for metric_type, calc_metrics in metric_funcs.items():
            try:
                _metrics = calc_metrics(
                    [gold_proof],
                    pred_proof,
                    facts=facts,
                )
            except Exception as e:
                logger.warning(
                    'calc_metrics() failed due to the following error. this sample will be skipped from metrics: %s', str(e))
                _metrics = {}
            depths = (['all', str(example['depth'])] if example.get('depth', None) is not None
                      else ['all', 'None'])
            for depth in depths:
                for metric_name, metric_val in _metrics.items():
                    metrics[f"{metric_type}.D-{depth}.{metric_name}"] = metric_val

        return metrics

    def _get_logic(self, example) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return (
            example['facts'],
            example['hypothesis'],
            example['gold_proof'],
        )
