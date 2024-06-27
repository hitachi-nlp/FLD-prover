from typing import Optional, Dict

from transformers import DataCollatorForSeq2Seq, default_data_collator
from trl import DataCollatorForCompletionOnlyLM

# taken from data_processing.preprocess_function
_REMOVE_NAMES = [
    # ---- FLD ----
    'depth',
    'facts',
    'hypothesis',
    'prompts_w_partial_proof',
    'proof_step',
    'gold_proof',


    # 'input_ids',
    # 'attention_mask',
    # 'labels',
    'version',
    'hypothesis_formula',
    'facts_formula',
    'proofs',
    'proofs_formula',
    'negative_hypothesis',
    'negative_hypothesis_formula',
    'negative_proofs',
    'negative_original_tree_depth',
    'original_tree_depth',
    'num_formula_distractors',
    'num_translation_distractors',
    'num_all_distractors',
    'proof_label',
    'negative_proof_label',
    'world_assump_label',
    'negative_world_assump_label',
    'prompt_serial',
    'proof_serial',

    # ---- ruletaker ----
    'context',
    'label',
    'question',
    'config',

    # ------------- proof writer -------------
    'id',
    'maxD',
    'NFact',
    'NRule',
    'theory',
    'triples',
    'rules',
    'questions',
    'allProofs',
    'proofDetails',

    # ------------- pararule plus -------------
    'meta',

    # ------------- robust lr -------------
    'statement',
]


def _remove_features(features: Dict) -> Dict:
    features_removed = features.copy()
    for feature in features_removed:
        for remove_name in _REMOVE_NAMES:
            if remove_name in feature:
                feature.pop(remove_name, None)
    return features_removed


class RemoveUnusedColumnsCollator:

    def __init__(self,
                 return_tensors: Optional[str] = None):
        if return_tensors is None:
            raise ValueError()
        self.return_tensors = return_tensors

    def __call__(self, features, return_tensors=None):
        return default_data_collator(_remove_features(features),
                                     return_tensors=return_tensors or self.return_tensors)


class RemoveUnusedColumnsCollatorForCompletionOnlyLM(RemoveUnusedColumnsCollator):

    def __init__(self,
                 response_template,
                 tokenizer=None,
                 return_tensors: Optional[str] = None):
        super().__init__(return_tensors=return_tensors)
        self._collator_for_completion = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    def __call__(self, features, return_tensors=None):
        return self._collator_for_completion(_remove_features(features))
