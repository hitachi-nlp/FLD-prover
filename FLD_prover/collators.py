from typing import Optional
from transformers import DataCollatorForSeq2Seq, default_data_collator

class RemoveUnusedColumnsCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        for feature in features:
            if "depth" in feature:
                feature.pop("depth", None)
        return super().__call__(features, return_tensors=return_tensors)


class RemoveUnusedColumnsCollator:

    def __init__(self, return_tensors: Optional[str] = None):
        if return_tensors is None:
            raise ValueError()
        self.return_tensors = return_tensors

    def __call__(self, features, return_tensors=None):
        for feature in features:
            if "depth" in feature:
                feature.pop("depth", None)
        return default_data_collator(features,
                                     return_tensors=return_tensors or self.return_tensors)
