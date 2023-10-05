from typing import Optional, Union, Any
import logging
from enum import Enum

import numpy as np
import torch
from typing import List, Dict
from FLD_task import (
    load_deduction,
    serialize,
)
from FLD_task.proof import get_stance_markers


class LMType(Enum):
    SEQ_2_SEQ = 'seq2seq'
    CAUSAL = 'causal'


logger = logging.getLogger()


def preprocess_function(examples: Dict[str, List[Any]],
                        split: str,
                        lm_type: LMType,
                        tokenizer,
                        prompt_prefix='',
                        padding=False,
                        max_source_length=1024,
                        max_target_length=1024,
                        ignore_index=-100,
                        proof_sampling=False,
                        sample_negative_proof=False,
                        no_subproof_for_unknown=False,
                        ignore_prompt_for_causal_lm_loss=False,
                        log_examples=False) -> Dict[str, List[Any]]:

    def _prepare_tokenized_targets(targets, max_length, **kwargs):
        return prepare_tokenized_targets(targets, tokenizer, padding, max_length, **kwargs)

    def _prepare_tokenized_inputs(inputs, max_length, **kwargs):
        return prepare_tokenized_inputs(inputs, tokenizer, padding, max_length, **kwargs)

    def _mask_labels_by_ignore_index(labels, mask_lengths: Optional[List[int]] = None):
        return mask_labels_by_ignore_index(labels,
                                           tokenizer.pad_token_id,
                                           mask_id=ignore_index,
                                           mask_lengths=mask_lengths,
                                           mask_pad_tokens = padding == "max_length" and data_args.ignore_pad_token_for_loss)

    def _unmask_by_pad_token(tensor):
        return unmask_by_pad_token(tensor, tokenizer.pad_token_id, mask_id=ignore_index)

    whole_proof_max_length = (
        max_target_length * 200 if proof_sampling == 'stepwise' and lm_type == LMType.SEQ_2_SEQ
        else max_target_length
    )

    batch_size = len(list(examples.values())[0])
    unbatched_examples = [{key: examples[key][i] for key in examples.keys()}
                          for i in range(batch_size)]

    prompts_w_partial_proof: List[str] = []
    proof_steps: List[str] = []
    gold_proofs: List[str] = []
    for i_example, example in enumerate(unbatched_examples):
        deduction = load_deduction(example)
        serial = serialize(
            deduction,
            stepwise = (proof_sampling == 'stepwise'),
            sample_negative_proof = sample_negative_proof if split == 'train' else False,
            include_max_subproof_for_unknown = not no_subproof_for_unknown,
        )

        prompt_with_partial_proof = prompt_prefix + serial.prompt + (serial.partial_proof or '')

        gold_proof = serial.proof
        # check whther the tokenizer can recognize stance markers
        gold_proof_dec = tokenizer.decode(_prepare_tokenized_targets([gold_proof],
                                                                     whole_proof_max_length)["input_ids"][0])
        if len(get_stance_markers(gold_proof_dec)) == 0:
            logger.warning(
                '\n'.join([
                    'The tokenizer could not recognized the stance markers.',
                    f'The original proof: "{gold_proof}"',
                    f'The tokenized proof: "{gold_proof_dec}"',
                ])
            )

        prompts_w_partial_proof.append(prompt_with_partial_proof)
        proof_steps.append(serial.next_proof_step)
        gold_proofs.append(gold_proof)

        if log_examples:
            logger.info('------------------------------ preprocess_function [example=%d] ------------------------------', i_example)
            logger.info('prompt             : "%s"', prompt_with_partial_proof)
            logger.info('next proof step    : "%s"', serial.next_proof_step)
            logger.info('gold proof         : "%s"', gold_proof)

    _proof_steps_w_eos = [step + f' {tokenizer.eos_token}' for step in proof_steps]

    # without this additional token, we can not accurately calculate the prompt length
    causal_lm_sep_token = '::'
    forward_inputs: Dict[str, Any] = {}
    if split == 'train':
        if any(_targets is None for _targets in proof_steps):
            raise ValueError()

        if lm_type == LMType.SEQ_2_SEQ:
            forward_inputs.update(_prepare_tokenized_inputs(prompts_w_partial_proof, max_source_length))
            forward_inputs["labels"] = _prepare_tokenized_targets(_proof_steps_w_eos,
                                                                  max_target_length)["input_ids"]
            forward_inputs["labels"] = _mask_labels_by_ignore_index(forward_inputs["labels"])

        elif lm_type == LMType.CAUSAL:
            # just for getting length
            _prompts = [prompt + causal_lm_sep_token for prompt in prompts_w_partial_proof]

            if ignore_prompt_for_causal_lm_loss:
                prompt_lengths = None
            else:
                prompt_ids = [
                    _prepare_tokenized_inputs(
                        prompt,
                        max_source_length,
                        padding='longest',
                        return_length=True,
                        # add_special_tokens=False,
                    )
                    for prompt in _prompts
                ]
                prompt_lengths = [_promt_ids['length'][0] for _promt_ids in prompt_ids]

            inputs_with_targets = [f'{prompt}{proof_step}'
                                   for prompt, proof_step in zip(_prompts, _proof_steps_w_eos)]
            forward_inputs.update(_prepare_tokenized_inputs(inputs_with_targets, max_source_length))
            forward_inputs["labels"] = forward_inputs['input_ids'].detach().clone()

            forward_inputs["labels"] = _mask_labels_by_ignore_index(
                forward_inputs["labels"],
                mask_lengths=prompt_lengths,
            )
        else:
            raise NotImplementedError()
    else:
        if any(_gold_proofs is None for _gold_proofs in gold_proofs):
            raise Exception('Why pass here? might be a bug?')

        proof_col = 'gold_proofs'

        if lm_type == LMType.SEQ_2_SEQ:
            forward_inputs.update(_prepare_tokenized_inputs(prompts_w_partial_proof, max_source_length))
            forward_inputs[proof_col] = _prepare_tokenized_targets(gold_proofs,
                                                                   whole_proof_max_length)["input_ids"]
            forward_inputs[proof_col] = _mask_labels_by_ignore_index(forward_inputs[proof_col])

        elif lm_type == LMType.CAUSAL:
            _prompts = [prompt + causal_lm_sep_token for prompt in prompts_w_partial_proof]

            forward_inputs.update(
                _prepare_tokenized_inputs(
                    _prompts,
                    max_source_length,
                    # add_special_tokens=False
                ))

            # the padding is arbitrary
            forward_inputs[proof_col] = _prepare_tokenized_targets(gold_proofs,
                                                                   whole_proof_max_length)["input_ids"]
            forward_inputs[proof_col] = _mask_labels_by_ignore_index(forward_inputs[proof_col])
        else:
            raise NotImplementedError()

    # some models do not accept 'token_type_ids' as inputs
    if 'token_type_ids' in forward_inputs:
        forward_inputs.pop('token_type_ids', None)
    if "depth" in examples:
        forward_inputs["depth"] = examples["depth"]

    inputs_decoded = tokenizer.batch_decode(_unmask_by_pad_token(forward_inputs['input_ids']))
    if 'labels' in forward_inputs:
        labels_decoded = tokenizer.batch_decode(_unmask_by_pad_token(forward_inputs['labels']))
    else:
        labels_decoded = [None] * len(inputs_decoded)
    for i_example, (input_decoded, label_decoded) in enumerate(zip(inputs_decoded, labels_decoded)):
        logger.info('------------ [example=%d] tokenized inputs ----------------', i_example)
        logger.info(input_decoded)
        if label_decoded is not None:
            logger.info('------------ [example=%d] tokenized labels ----------------', i_example)
            logger.info(label_decoded)

    return forward_inputs


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
