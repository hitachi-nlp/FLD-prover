from typing import Any, Tuple
import logging

from typing import Dict
from FLD_task import (
    load_deduction,
    serialize,
)
from FLD_task.proof import get_stance_markers
from FLD_prover.lm_types import LMType
from FLD_prover.tokenization import prepare_tokenized_targets

from .base import Preprocessor

logger = logging.getLogger()


class FLDPreprocessor(Preprocessor):

    def _get_logic(
        self,
        example,
        split: str,
    ) -> Tuple[str, str, str]:

        def _prepare_tokenized_targets(targets, max_length, **kwargs):
            return prepare_tokenized_targets(targets, self._tokenizer, self._padding, max_length, **kwargs)

        whole_proof_max_length = (
            self._max_target_length * 200 if self._proof_sampling == 'stepwise' and self._lm_type == LMType.SEQ_2_SEQ
            else self._max_target_length
        )

        deduction = load_deduction(example)
        serial = serialize(
            deduction,
            stepwise=(self._proof_sampling == 'stepwise'),
            sample_negative_proof=self._sample_negative_proof if split == 'train' else False,
            include_max_subproof_for_unknown=not self._no_subproof_for_unknown,
            instruction=self._instruction,
        )

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

    def _get_features(self, examples) -> Dict[str, Any]:
        return {
            add_feature: examples[add_feature]
            for add_feature in ['depth', 'hypothesis', 'facts']
            if add_feature in examples
        }
