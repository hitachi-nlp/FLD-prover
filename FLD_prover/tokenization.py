from typing import Optional, Union, Any
import logging

import numpy as np
import torch
from typing import List, Dict

logger = logging.getLogger()


CAUSAL_LM_END_OF_PROMPT = '::'


def prepare_tokenized_inputs(inputs: List[str],
                             tokenizer,
                             padding: str,
                             max_length: int,
                             **kwargs) -> Dict[str, torch.Tensor]:
    _padding = kwargs.pop('padding', padding)
    tokenized = tokenize_with_log(tokenizer,
                                  text=inputs,
                                  max_length=max_length,
                                  padding=_padding,
                                  truncation=True,
                                  return_tensors='pt',
                                  add_special_tokens=False,
                                  **kwargs)
    return tokenized


def prepare_tokenized_targets(targets: List[str],
                              tokenizer,
                              padding: str,
                              max_length: int,
                              **kwargs) -> Dict[str, torch.Tensor]:
    tokenized = tokenize_with_log(tokenizer,
                                  text_target=targets,
                                  max_length=max_length,
                                  padding=padding,
                                  truncation=True,
                                  return_tensors='pt',
                                  add_special_tokens=False,
                                  **kwargs)
    return tokenized


def mask_labels_by_ignore_index(labels,
                                pad_token_id,
                                mask_id=-100,
                                mask_lengths: Optional[List[int]] = None,
                                mask_pad_tokens=True):
    """
    [OpenCALM-7BをLoRAでinstruction tuningするための実装解説](https://qiita.com/m__k/items/173ade78990b7d6a4be4)
    """

    if mask_lengths is not None:
        for i_label, mask_length in enumerate(mask_lengths):
            non_pad_first_token = 0  # for the case of "left" padding
            for i_token, token_id in enumerate(labels[i_label].numpy().tolist()):
                if token_id != pad_token_id:
                    non_pad_first_token = i_token
                    break
            labels[i_label][non_pad_first_token : non_pad_first_token + mask_length] = mask_id

    if mask_pad_tokens:
        labels = torch.where(labels != pad_token_id, labels, mask_id)

    return labels


def unmask_by_pad_token(tensor: Union[np.ndarray, torch.Tensor],
                        pad_token_id,
                        mask_id=-100) -> np.ndarray:
    if isinstance(tensor, list):
        tensor = torch.tensor(tensor)
    if not isinstance(tensor, (np.ndarray, torch.Tensor)):
        raise ValueError()
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return np.where(tensor != mask_id, tensor, pad_token_id)


def tokenize_with_log(tokenizer, **kwargs):
    if kwargs.get('truncation', False) is False:
        return tokenizer(**kwargs)

    sub_kwargs = kwargs.copy()
    truncation = sub_kwargs.pop('truncation', False)
    padding = sub_kwargs.pop('padding', False)

    tokens_wo_truncation = tokenizer(
        truncation=False,
        padding='longest',
        **sub_kwargs,
    )

    tokens_with_truncation = tokenizer(
        truncation=truncation,
        padding=padding,
        **sub_kwargs,
    )

    if 'text' in kwargs:
        if 'text_target' in kwargs:
            raise NotImplementedError()
        texts = kwargs['text']
    elif 'text_target' in kwargs:
        texts = kwargs['text_target']
    else:
        raise NotImplementedError()

    for _text, _tokens_with_truncation, _tokens_wo_truncation in zip(texts, tokens_with_truncation['input_ids'], tokens_wo_truncation['input_ids']):
        if len(_tokens_with_truncation) < len(_tokens_wo_truncation):
            logger.warning('The input text has %d token ids, but they are truncated into %d ids.',
                           len(_tokens_wo_truncation),
                           len(_tokens_with_truncation))
            logger.warning('The input text is: "%s"', _text)
            # logger.warning('tokniezer() options are: %s', str(kwargs))
        elif len(_tokens_with_truncation) == len(_tokens_wo_truncation):
            pass
        elif len(_tokens_with_truncation) > len(_tokens_wo_truncation):
            logger.debug(
                'The input text has %d token ids, but they are up-converted into %d ids. This is no problem for learning but memory inefficient.',
                len(_tokens_wo_truncation),
                len(_tokens_with_truncation),
            )
            # logger.debug('The input text is: "%s"', _text)
            # logger.debug('tokniezer() options are: %s', str(kwargs))
    return tokens_with_truncation
