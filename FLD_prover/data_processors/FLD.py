from typing import Optional, Any, Tuple, Dict
import logging
import random

from FLD_task import (
    load_deduction,
    Serializor,
    build_metrics,
    SerializedDeduction,
    make_instruction,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._serializor = Serializor(
            proof_intermediate_steps_prob=self._proof_intermediate_steps_prob,
            proof_sampling=self._proof_sampling,
            sample_negative_proof=self._sample_negative_proof,
            no_subproof_for_unknown=self._no_subproof_for_unknown,
            no_instruction=self._no_instruction,
            prompt_indicate_theorems=self._prompt_indicate_theorems,
            prompt_emphasize_theorems=self._prompt_emphasize_theorems,
            augmentation=self._augmentation,
            augmentation_prob=self._augmentation_prob,
            paraphrase_contradiction=self._paraphrase_contradiction,
            formula_prob=self._formula_prob,
        )

    def _make_in_out(
        self,
        example,
        split: str,
    ) -> Tuple[str, str, str]:

        serial = self._get_serial(example, split)
        prompt_with_partial_proof = serial.prompt + (serial.partial_proof or '')
        next_proof_step = serial.next_proof_step
        gold_proof = serial.proof

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
        deduction = load_deduction(example)
        if self._use_original_serial:
            serial = SerializedDeduction(
                prompt=deduction.prompt_serial,
                proof=deduction.proof_serial,
                partial_proof=None,
                next_proof_step=deduction.proof_serial,
            )
        else:
            serial = self._serializor(deduction, split)
        return serial
