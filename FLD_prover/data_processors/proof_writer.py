from typing import Optional, Any, Tuple, Dict
import logging
import random
import re

from FLD_task.proof import StanceMarker, add_stance_markers, get_stance_markers
from FLD_task.evaluation import compute_answer_accuracy
from FLD_prover.tokenization import unmask_by_pad_token

from .base import Processor

logger = logging.getLogger()


class ProofWriterProcessor(Processor):

    _warn_on_example_prettify_failure = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._question_cache: Dict[str, int] = {}
        self._proof_cache: Dict[str, int] = {}

    def _make_in_out(
        self,
        example,
        split: str,
    ) -> Tuple[str, str, str]:

        if random.random() > self._proof_intermediate_steps_prob:
            include_proof = False
        else:
            include_proof = True

        facts, hypothesis, gold_proof = self._get_logic(example, 'train', include_proof=include_proof)
        prompt = ' ; '.join([
            '$facts$ = ' + facts,
            '$hypothesis$ = ' + hypothesis,
            '$proof$ = '
        ])
        prompt_with_partial_proof = self._prompt_prefix + prompt
        next_proof_step = gold_proof
        return prompt_with_partial_proof, next_proof_step, gold_proof

    def _compute_metrics(self, example, pred_proof: str):

        metrics = {}

        facts, hypothesis, gold_proof = self._get_logic(example, 'eval')

        gold_markers = get_stance_markers(gold_proof)

        answer_accuracy = compute_answer_accuracy(gold_proof, pred_proof)
        if gold_markers == [StanceMarker.UNKNOWN] and answer_accuracy == 1.0:
            proof_accuracy = 1.0
        else:
            proof_accuracy = 1.0 if pred_proof.strip(' ') == gold_proof.strip(' ') else 0.0

        _metrics = {
            'answer_accuracy': answer_accuracy,
            'proof_accuracy': proof_accuracy,
        }
        depths = (['all', str(example['depth'])] if example.get('depth', None) is not None
                  else ['all', 'None'])
        for depth in depths:
            for metric_name, metric_val in _metrics.items():
                metrics[f"D-{depth}.{metric_name}"] = metric_val

        return metrics

    def _get_features(self, examples) -> Dict[str, Any]:
        # TODO: implement
        # return {
        #     'depth': [int(depth_str.lstrip('depth-'))
        #               for depth_str in examples['config']]
        # }
        return {}

    def _get_logic(self,
                   example,
                   split: str,
                   include_proof=True) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        id_ = example['id']

        facts = example['theory']

        questions = list([q for q in example['questions'].values() if q is not None])
        if id_ in self._question_cache:
            question_idx = self._question_cache[id_]
        else:
            question_idx = random.choice(range(len(questions)))
            self._question_cache[id_] = question_idx
        question = questions[question_idx]

        hypothesis = question['question']

        if include_proof\
                and question.get('proofsWithIntermediates', None) and len(question['proofsWithIntermediates']) > 0:
            proofs = question['proofsWithIntermediates']
            if id_ in self._proof_cache:
                proof_idx = self._proof_cache[id_]
            else:
                proof_idx = random.choice(range(len(proofs)))
                self._proof_cache[id_] = proof_idx
            proof_dic = proofs[proof_idx]
            proof = proof_dic['representation']

            for step_dic in proof_dic['intermediates']:
                step_id = step_dic['id']
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

        label = str(question['answer'])
        if label == 'True':
            marker = StanceMarker.PROVED
        elif label == 'Unknown':
            marker = StanceMarker.DISPROVED
        elif label == 'False':
            marker = StanceMarker.DISPROVED
        else:
            raise ValueError()
        gold_proof = add_stance_markers(proof, [marker])

        return facts, hypothesis, gold_proof
