from typing import Optional
import logging
import time
from copy import copy

import torch
from transformers.generation.stopping_criteria import StoppingCriteria

from .lm_types import LMType

logger = logging.getLogger(__name__)


class MaxTimeCriteriaWithWarning(StoppingCriteria):
    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        elapsed = time.time() - self.initial_timestamp
        do_timeout = elapsed > self.max_time
        if do_timeout:
            logger.warning('generation timeout with %d sec', self.max_time)
            return True
        else:
            return False


def generation_handled(func,
                       lm_type: LMType,
                       tokenizer,
                       model,
                       timeout: Optional[int] = None,
                       top_k=1,
                       num_return_sequences=1):
    if lm_type == LMType.CAUSAL:
        """
            model.generate() with batch size >= 2 require hacks on special tokens.
            this preparation must be exactly when entering/exiting evaluate()/predict(),
            as other functions such as train() should not respect this hack.
            See [here](https://discuss.huggingface.co/t/batch-generation-with-gpt2/1517/2)
        """

        padding_side_org = tokenizer.padding_side
        pad_token_org = tokenizer.pad_token
        pad_token_id_org = model.config.eos_token_id

        def generation_init_special_tokens():
            logger.info('generation_init_special_tokens() called!')
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        def generation_exit_special_tokens():
            logger.info('generation_exit_special_tokens() called!')
            tokenizer.padding_side = padding_side_org
            tokenizer.pad_token = pad_token_org
            model.config.pad_token_id = pad_token_id_org
    else:
        def generation_init_special_tokens():
            pass

        def generation_exit_special_tokens():
            pass

    def make_gen_kwargs():
        """
            As generation_init_special_tokens()/generation_exit_special_tokens() dynamically change
            tokenizer special tokens, we also have to generate gen_kwargs dynamically.
        """
        stopping_criteria = MaxTimeCriteriaWithWarning(timeout)
        return {
            'top_k': top_k,
            'stopping_criteria': [stopping_criteria],
            'num_return_sequences': num_return_sequences,
            'pad_token_id': tokenizer.pad_token_id,
        }

    def handled(self, *args, **kwargs):
        generation_init_special_tokens()
        results = func(self, *args, **kwargs, **make_gen_kwargs())
        generation_exit_special_tokens()
        return results

    return handled
