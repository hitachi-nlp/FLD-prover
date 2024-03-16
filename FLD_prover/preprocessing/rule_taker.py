from typing import Any, Tuple
import logging

from typing import Dict
from FLD_prover.lm_types import LMType
from FLD_prover.tokenization import prepare_tokenized_targets

from .base import Preprocessor

logger = logging.getLogger()


class RuleTakerPreprocessor(Preprocessor):

    def _get_logic(
        self,
        example,
        split: str,
    ) -> Tuple[str, str, str]:

        context = 'Context: ' + example['context'] + '\n Question: ' + example['question']
        prompt_with_partial_proof = self._prompt_prefix + context
        next_proof_step = example['label']
        gold_proof = example['label']
        return prompt_with_partial_proof, next_proof_step, gold_proof

    def _get_features(self, examples) -> Dict[str, Any]:
        return {
            'depth': [int(depth_str.lstrip('depth-'))
                      for depth_str in examples['config']]
        }
