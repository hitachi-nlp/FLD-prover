from typing import Optional, Any, Tuple, Dict
import logging

from FLD_task.proof import StanceMarker, add_stance_markers
from FLD_prover.tokenization import unmask_by_pad_token

from .base import Processor

logger = logging.getLogger()


class RuleTakerProcessor(Processor):

    def _make_in_out(
        self,
        example,
        split: str,
        padding='longest',
    ) -> Tuple[str, str, str]:
 

        facts, hypothesis, gold_proof = self._get_logic(example, 'train')

        prompt = ' ; '.join([
            '$facts$ = ' + facts,
            '$hypothesis$ = ' + hypothesis,
            '$proof$ = '
        ])
        prompt_with_partial_proof = self._prompt_prefix + prompt
        next_proof_step = gold_proof

        return prompt_with_partial_proof, next_proof_step, gold_proof

    def _compute_metrics(self, example, pred_proof: str):

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        metrics = {}

        facts, hypothesis, gold_proof = self._get_logic(example, 'eval')

        _metrics = {
            'accuracy': 1.0 if gold_proof == pred_proof else 0.0,
        }
        depths = (['all', str(example['depth'])] if example.get('depth', None) is not None
                  else ['all', 'None'])
        for depth in depths:
            for metric_name, metric_val in _metrics.items():
                metrics[f"D-{depth}.{metric_name}"] = metric_val

        return metrics

    def _get_features(self, examples) -> Dict[str, Any]:
        return {
            'depth': [int(depth_str.lstrip('depth-'))
                      for depth_str in examples['config']]
        }

    def _get_logic(self, example, split: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        facts = example['context']
        hypothesis = example['question']

        label = example['label']
        if label == 'entailment':
            marker = StanceMarker.PROVED
        elif label == 'not entailment':
            marker = StanceMarker.UNKNOWN
        else:
            raise ValueError()
        gold_proof = add_stance_markers('', [marker])

        return facts, hypothesis, gold_proof
