from typing import Any, Tuple
import logging

from collections import defaultdict
from typing import List, Dict
from FLD_task import (
    load_deduction,
    serialize,
    build_metrics,
    log_example,
)
from FLD_task.proof import get_stance_markers
from FLD_prover.lm_types import LMType
from FLD_prover.tokenization import (
    prepare_tokenized_targets,
    unmask_by_pad_token,
)

from .base import Metrics

logger = logging.getLogger()


class FLDMetrics(Metrics):

    def _compute_metrics_from_example(self, example, pred_proof: str) -> Dict[str, List[Any]]:

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        metrics: Dict[str, List[Any]] = defaultdict(list)

        metric_funcs = {
            'strct': build_metrics('strict'),
            'extr_stps': build_metrics('allow_extra_steps'),
        }

        gold_proof = example['gold_proof']
        input_ids = example['input_ids']
        input_ids = _unmask_by_pad_token(input_ids)

        facts = example['facts']
        hypothesis = example['hypothesis']

        log_example(
            facts=facts,
            hypothesis=hypothesis,
            gold_proofs=[gold_proof],
            pred_proof=pred_proof,
            logger=logger,
        )

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
                    metrics[f"{metric_type}.D-{depth}.{metric_name}"].append(metric_val)

            log_texts, log_args = [], []
            for metric_name, metric_val in sorted(_metrics.items()):
                log_texts.append('%-20s: %5.2f')
                log_args.extend([f"{metric_type}.{metric_name}", metric_val])
            logger.info('------------   metrics  ------------\n' + '\n'.join(log_texts), *log_args)

        return metrics
