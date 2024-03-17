from typing import Any, Tuple
import logging
import re
import random
from typing import Dict

from FLD_task.proof import StanceMarker, add_stance_markers

from .base import Preprocessor

logger = logging.getLogger()


class ProofWriterPreprocessor(Preprocessor):

    def _get_logic(
        self,
        example,
        split: str,
    ) -> Tuple[str, str, str]:

        facts = example['theory']

        question = random.choice(example['questions'].values())
        hypothesis = question['question']

        if 'proofsWithIntermediates' in question:
            proof_dic = random.choice(question['proofsWithIntermediates'])
            proof = proof_dic['representation']
            for step_id, step_dic in proof_dic['intermediates'].items():
                step_text = step_dic['text']

                proof_org = proof

                proof = re.sub(f'{step_id}\)', f'{step_id}: {step_text})', proof, 1)
                is_replaced = proof != proof_org
                if is_replaced:
                    continue

                proof = re.sub(f'{step_id} ', f'{step_id}: {step_text} ', proof, 1)
                is_replaced = proof != proof_org
                if is_replaced:
                    continue

                proof = re.sub(f'{step_id}$', f'{step_id}: {step_text}', proof, 1)
                is_replaced = proof != proof_org
                if is_replaced:
                    continue
        else:
            proof = ''

        prompt = ' ; '.join([
            '$facts$ = ' + facts,
            '$hypothesis$ = ' + hypothesis,
            '$proof$ = '
        ])
        prompt_with_partial_proof = self._prompt_prefix + prompt

        label = str(question['answer'])
        if label == 'True':
            marker = StanceMarker.PROVED
        elif label == 'Unknown':
            marker = StanceMarker.DISPROVED
        elif label == 'False':
            marker = StanceMarker.DISPROVED
        else:
            raise ValueError()

        next_proof_step = add_stance_markers(proof, marker)
        gold_proof = next_proof_step

        return prompt_with_partial_proof, next_proof_step, gold_proof

    def _get_features(self, examples) -> Dict[str, Any]:
        # TODO: implement
        # return {
        #     'depth': [int(depth_str.lstrip('depth-'))
        #               for depth_str in examples['config']]
        # }
        return {}
