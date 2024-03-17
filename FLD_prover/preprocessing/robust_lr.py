from typing import Any, Tuple
import logging

from typing import Dict

from FLD_task.proof import StanceMarker, add_stance_markers

from .base import Preprocessor

logger = logging.getLogger()


class RobustLRPreprocessor(Preprocessor):

    def _get_logic(
        self,
        example,
        split: str,
    ) -> Tuple[str, str, str]:

        prompt = ' ; '.join([
            '$facts$ = ' + example['context'],
            '$hypothesis$ = ' + example['statement'],
            '$proof$ = '
        ])
        prompt_with_partial_proof = self._prompt_prefix + prompt

        label = example['label']
        if label == 'entailment':
            marker = StanceMarker.PROVED
        elif label == 'neutral':
            marker = StanceMarker.UNKNOWN
        elif label == 'contradiction':
            marker = StanceMarker.DISPROVED
        else:
            raise ValueError()

        next_proof_step = add_stance_markers('', [marker])
        gold_proof = next_proof_step

        return prompt_with_partial_proof, next_proof_step, gold_proof

    def _get_features(self, examples) -> Dict[str, Any]:
        return {}
