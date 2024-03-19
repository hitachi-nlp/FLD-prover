from typing import Optional
from transformers import DataCollatorForSeq2Seq, default_data_collator

# taken from data_processing.preprocess_function
_REMOVE_NAMES = [
    # ---- ruletaker ----
    'context',
    'label',
    'question',
    'config',

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
]


class RemoveUnusedColumnsCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        for feature in features:
            for remove_name in _REMOVE_NAMES:
                if remove_name in feature:
                    feature.pop(remove_name, None)
        return super().__call__(features, return_tensors=return_tensors)


class RemoveUnusedColumnsCollator:

    def __init__(self,
                 return_tensors: Optional[str] = None):
        if return_tensors is None:
            raise ValueError()
        self.return_tensors = return_tensors

    def __call__(self, features, return_tensors=None):
        for feature in features:
            for remove_name in _REMOVE_NAMES:
                if remove_name in feature:
                    feature.pop(remove_name, None)
        return default_data_collator(features,
                                     return_tensors=return_tensors or self.return_tensors)
