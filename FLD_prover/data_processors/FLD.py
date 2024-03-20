from typing import Optional, Any, Tuple, Dict
import logging

from FLD_task import (
    load_deduction,
    serialize,
    build_metrics,
    SerializedDeduction,
)
from FLD_task.proof import get_stance_markers
from FLD_prover.lm_types import LMType
from FLD_prover.tokenization import (
    prepare_tokenized_targets,
    unmask_by_pad_token,
)
from .base import Processor

logger = logging.getLogger()


class FLDProcessor(Processor):

    _warn_on_example_prettify_failure = True

    def _make_in_out(
        self,
        example,
        split: str,
        padding='longest',
    ) -> Tuple[str, str, str]:

        def _prepare_tokenized_targets(targets, max_length, **kwargs):
            return prepare_tokenized_targets(targets, self._tokenizer, padding, max_length, **kwargs)

        whole_proof_max_length = (
            self._max_target_length * 200 if self._proof_sampling == 'stepwise' and self._lm_type == LMType.SEQ_2_SEQ
            else self._max_target_length
        )

        serial = self._get_serial(example, split)

        prompt_with_partial_proof = self._prompt_prefix + serial.prompt + (serial.partial_proof or '')
        next_proof_step = serial.next_proof_step
        gold_proof = serial.proof
        if gold_proof is not None:
            # check whther the tokenizer can recognize stance markers
            gold_proof_dec = self._tokenizer.decode(_prepare_tokenized_targets([gold_proof],
                                                                         whole_proof_max_length)["input_ids"][0])
            if len(get_stance_markers(gold_proof_dec)) == 0:
                logger.warning(
                    '\n'.join([
                        'The tokenizer could not recognized the stance markers.',
                        f'The original proof: "{gold_proof}"',
                        f'The tokenized proof: "{gold_proof_dec}"',
                    ])
                )
        return prompt_with_partial_proof, next_proof_step, gold_proof

    def _compute_metrics(self, example, pred_proof: str):

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        metrics = {}

        metric_funcs = {
            'strct': build_metrics('strict'),
            'extr_stps': build_metrics('allow_extra_steps'),
        }

        facts, hypothesis, gold_proof = self._get_logic(example, 'eval')

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

    def _get_features(self, examples) -> Dict[str, Any]:
        return {
            add_feature: examples[add_feature]
            for add_feature in ['depth', 'hypothesis', 'facts']
            if add_feature in examples
        }

    def _get_logic(self, example, split: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        gold_proof = self._get_serial(example, split).proof
        return (
            example['facts'],
            example['hypothesis'],
            gold_proof,
        )

    def _get_serial(self, example, split: str) -> SerializedDeduction:
        return serialize(
            load_deduction(example),
            stepwise=(self._proof_sampling == 'stepwise'),
            sample_negative_proof=self._sample_negative_proof if split == 'train' else False,
            include_max_subproof_for_unknown=not self._no_subproof_for_unknown,
            instruction=self._instruction,
        )
