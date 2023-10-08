from typing import Optional
from transformers import DataCollatorForSeq2Seq, default_data_collator

# taken from data_processing.preprocess_function
_REMOVE_NAMES = [
    'depth',
    'facts',
    'hypothesis',
    'gold_proofs',
]


class RemoveUnusedColumnsCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        for feature in features:
            for remove_name in _REMOVE_NAMES:
                if remove_name in feature:
                    feature.pop(remove_name, None)
        return super().__call__(features, return_tensors=return_tensors)


class RemoveUnusedColumnsCollator:

    def __init__(self, return_tensors: Optional[str] = None):
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
