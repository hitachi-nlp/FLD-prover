
from typing import Optional, Union, Any, Tuple
import logging
import re
from pprint import pformat
from abc import ABC, abstractmethod

from collections import defaultdict
import numpy as np
import torch
from typing import List, Dict
from FLD_task import (
    load_deduction,
    serialize,
    build_metrics,
    log_example,
)
from FLD_task.proof import get_stance_markers
from FLD_prover.lm_types import LMType
from FLD_prover.tokenization import (
    CAUSAL_LM_END_OF_PROMPT,
    prepare_tokenized_inputs,
    prepare_tokenized_targets,
    mask_labels_by_ignore_index,
    unmask_by_pad_token,
)

logger = logging.getLogger()


class Preprocessor(ABC):

    def __init__(self,
                 lm_type: LMType,
                 tokenizer,
                 prompt_prefix='',
                 padding=False,
                 max_source_length=1024,
                 max_target_length=1024,
                 ignore_index=-100,
                 proof_sampling='stepwise',
                 sample_negative_proof=False,
                 no_subproof_for_unknown=False,
                 ignore_pad_token_for_loss=True,
                 include_prompt_for_causal_lm_loss=False,
                 instruction=False,
                 log_examples=False):
        self._lm_type = lm_type
        self._tokenizer = tokenizer
        self._prompt_prefix = prompt_prefix
        self._padding = padding
        self._max_source_length = max_source_length
        self._max_target_length = max_target_length
        self._ignore_index = ignore_index
            
        self._proof_sampling = proof_sampling
        self._sample_negative_proof = sample_negative_proof
        self._no_subproof_for_unknown = no_subproof_for_unknown
        self._ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self._include_prompt_for_causal_lm_loss = include_prompt_for_causal_lm_loss
        self._instruction = instruction
        self._log_examples = log_examples

    def preprocess_examples(
        self,
        examples,
        split: str,
    )  -> Dict[str, List[Any]]:

        def _prepare_tokenized_targets(targets, max_length, **kwargs):
            return prepare_tokenized_targets(targets, self._tokenizer, self._padding, max_length, **kwargs)

        def _prepare_tokenized_inputs(inputs, max_length, padding=self._padding, **kwargs):
            return prepare_tokenized_inputs(inputs, self._tokenizer, padding, max_length, **kwargs)

        def _mask_labels_by_ignore_index(labels, mask_lengths: Optional[List[int]] = None):
            return mask_labels_by_ignore_index(labels,
                                               self._tokenizer.pad_token_id,
                                               mask_id=self._ignore_index,
                                               mask_lengths=mask_lengths,
                                               # mask_pad_tokens = padding == "max_length" and ignore_pad_token_for_loss,
                                               mask_pad_tokens=self._ignore_pad_token_for_loss)

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        batch_size = len(list(examples.values())[0])
        unbatched_examples = [{key: examples[key][i] for key in examples.keys()}
                              for i in range(batch_size)]

        prompts_w_partial_proof: List[str] = []
        proof_steps: List[str] = []
        gold_proofs: List[str] = []
        for i_example, example in enumerate(unbatched_examples):
            (
                prompt_w_partial_proof,
                next_proof_step,
                gold_proof,
            ) = self._get_logic(example, split)

            prompts_w_partial_proof.append(prompt_w_partial_proof)
            proof_steps.append(next_proof_step)
            gold_proofs.append(gold_proof)

            if self._log_examples:
                logger.info(
                    '------------------------------ preprocess_function [example=%d] ------------------------------', i_example)
                logger.info('prompt             : "%s"', prompt_w_partial_proof)
                logger.info('next proof step    : "%s"', next_proof_step)
                logger.info('gold proof         : "%s"', gold_proof)

        # without this additional token, we can not accurately calculate the prompt length
        # as the token
        forward_inputs: Dict[str, Any] = {}
        if split == 'train':
            _proof_steps_w_eos = [step + f' {self._tokenizer.eos_token}' for step in proof_steps]

            if any(_targets is None for _targets in proof_steps):
                raise ValueError()

            if self._lm_type == LMType.SEQ_2_SEQ:
                forward_inputs.update(_prepare_tokenized_inputs(prompts_w_partial_proof, self._max_source_length))
                forward_inputs["labels"] = _prepare_tokenized_targets(_proof_steps_w_eos,
                                                                      self._max_target_length)["input_ids"]
                forward_inputs["labels"] = _mask_labels_by_ignore_index(forward_inputs["labels"])

            elif self._lm_type == LMType.CAUSAL:
                # just for getting length
                _prompts = [prompt + CAUSAL_LM_END_OF_PROMPT for prompt in prompts_w_partial_proof]

                if self._include_prompt_for_causal_lm_loss:
                    prompt_lengths = None
                else:
                    prompt_ids = [
                        _prepare_tokenized_inputs(
                            [prompt],
                            self._max_source_length,
                            padding='longest',
                            return_length=True,
                            # add_special_tokens=False,
                        )
                        for prompt in _prompts
                    ]
                    prompt_lengths = [_promt_ids['length'][0] for _promt_ids in prompt_ids]

                inputs_with_targets = [f'{prompt}{proof_step}'
                                       for prompt, proof_step in zip(_prompts, _proof_steps_w_eos)]
                forward_inputs.update(_prepare_tokenized_inputs(inputs_with_targets, self._max_source_length))
                forward_inputs["labels"] = forward_inputs['input_ids'].detach().clone()

                forward_inputs["labels"] = _mask_labels_by_ignore_index(
                    forward_inputs["labels"],
                    mask_lengths=prompt_lengths,
                )
            else:
                raise NotImplementedError()

        elif split == 'eval':
            if self._lm_type == LMType.SEQ_2_SEQ:
                forward_inputs.update(_prepare_tokenized_inputs(prompts_w_partial_proof, self._max_source_length))

            elif self._lm_type == LMType.CAUSAL:
                _prompts = [prompt + CAUSAL_LM_END_OF_PROMPT for prompt in prompts_w_partial_proof]

                forward_inputs.update(
                    _prepare_tokenized_inputs(
                        _prompts,
                        self._max_source_length,
                        # add_special_tokens=False
                    ))

            else:
                raise NotImplementedError()

        else:
            raise ValueError()

        forward_inputs['prompts_w_partial_proof'] = prompts_w_partial_proof
        forward_inputs['proof_step'] = proof_steps
        forward_inputs['gold_proof'] = gold_proofs

        # some models do not accept 'token_type_ids' as inputs
        if 'token_type_ids' in forward_inputs:
            forward_inputs.pop('token_type_ids', None)

        if self._log_examples:
            inputs_decoded = self._tokenizer.batch_decode(_unmask_by_pad_token(forward_inputs['input_ids']))
            if 'labels' in forward_inputs:
                labels_decoded = self._tokenizer.batch_decode(_unmask_by_pad_token(forward_inputs['labels']))
            else:
                labels_decoded = [None] * len(inputs_decoded)

            for i_example, (input_decoded, label_decoded) in enumerate(zip(inputs_decoded, labels_decoded)):
                logger.info('------------ [example=%d] tokenized inputs ----------------', i_example)
                logger.info(input_decoded)
                if label_decoded is not None:
                    logger.info('------------ [example=%d] tokenized labels ----------------', i_example)
                    logger.info(label_decoded)

        forward_inputs.update(self._get_features(examples))

        return forward_inputs

    @abstractmethod
    def _get_logic(
        self,
        example,
        split: str,
    ) -> Tuple[str, str, str]:
        pass

    @abstractmethod
    def _get_features(self, examples) -> Dict[str, Any]:
        pass
