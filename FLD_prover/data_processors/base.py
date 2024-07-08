from typing import Optional, Any, Tuple, List, Dict
import logging
from pprint import pformat
from abc import ABC, abstractmethod

from collections import defaultdict
import numpy as np

from FLD_task import (
    log_example,
    log_metrics,
)
from FLD_prover.lm_types import LMType
from FLD_prover.tokenization import (
    CAUSAL_LM_END_OF_PROMPT,
    prepare_tokenized_inputs,
    prepare_tokenized_targets,
    mask_labels_by_ignore_index,
    unmask_by_pad_token,
)

logger = logging.getLogger()


class Processor(ABC):

    _warn_on_example_prettify_failure = False

    def __init__(self,
                 lm_type: LMType,
                 tokenizer,
                 surface_is_formula=False,
                 prompt_prefix='',
                 max_length=1024,
                 max_prompt_length=1024,
                 ignore_index=-100,
                 proof_intermediate_steps='include',
                 proof_sampling='stepwise',
                 sample_negative_proof=False,
                 no_subproof_for_unknown=False,
                 ignore_pad_token_for_loss=True,
                 include_prompt_for_causal_lm_loss=False,
                 instruction=False,
                 augmentation=False,
                 eval_dataset=None,
                 log_examples=False,
                 log_only_first_example=True):
        self._lm_type = lm_type
        self._tokenizer = tokenizer
        self._surface_is_formula = surface_is_formula
        self._prompt_prefix = prompt_prefix
        self._max_length = max_length
        self._max_prompt_length = max_prompt_length
        self._ignore_index = ignore_index
            
        if proof_intermediate_steps not in ['include', 'exclude', 'randomly_include']:
            raise ValueError(f"proof_intermediate_steps must be one of ['include', 'exclude', 'randomly_include'], but got {proof_intermediate_steps}")
        self._proof_intermediate_steps = proof_intermediate_steps
        self._proof_sampling = proof_sampling
        self._sample_negative_proof = sample_negative_proof
        self._no_subproof_for_unknown = no_subproof_for_unknown
        self._ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self._include_prompt_for_causal_lm_loss = include_prompt_for_causal_lm_loss
        self._instruction = instruction
        self._augmentation = augmentation
        self.eval_dataset = eval_dataset
        self.log_examples = log_examples
        self._log_only_first_example = log_only_first_example

    def preprocess(
        self,
        examples,
        mode: str,
        padding='longest',
    )  -> Dict[str, List[Any]]:


        def _prepare_tokenized_targets(targets, max_length, **kwargs):
            return prepare_tokenized_targets(targets, self._tokenizer, padding, max_length, **kwargs)

        def _prepare_tokenized_inputs(inputs, max_length, padding=padding, **kwargs):
            return prepare_tokenized_inputs(inputs, self._tokenizer, padding, max_length, **kwargs)

        def _mask_labels_by_ignore_index(labels,
                                         attention_mask=None,
                                         mask_lengths: Optional[List[int]] = None):
            return mask_labels_by_ignore_index(labels,
                                               self._tokenizer.pad_token_id,
                                               mask_id=self._ignore_index,
                                               mask_lengths=mask_lengths,
                                               # mask_pad_tokens = padding == "max_length" and ignore_pad_token_for_loss,
                                               mask_pad_tokens=self._ignore_pad_token_for_loss,
                                               attention_mask=attention_mask)

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        batch_size = len(list(examples.values())[0])
        unbatched_examples = [{key: examples[key][i] for key in examples.keys()}
                              for i in range(batch_size)]

        prompts_w_partial_proof: List[str] = []
        proof_steps: List[str] = []
        gold_proofs: List[str] = []
        for i_example, example in enumerate(unbatched_examples):
            prompt_w_partial_proof, next_proof_step, gold_proof = self._make_in_out(example, mode)

            prompts_w_partial_proof.append(prompt_w_partial_proof)
            proof_steps.append(next_proof_step)
            gold_proofs.append(gold_proof)

            if self.log_examples\
                    and not (self._log_only_first_example and i_example > 0):
                logger.info(
                    '------------------------------ preprocess_function [example=%d] ------------------------------', i_example)
                logger.info('prompt             : "%s"', prompt_w_partial_proof)
                logger.info('next proof step    : "%s"', next_proof_step)
                logger.info('gold proof         : "%s"', gold_proof)

        # without this additional token, we can not accurately calculate the prompt length
        # as the token
        forward_inputs: Dict[str, Any] = {}
        if mode == 'auto_regression':
            _proof_steps_w_eos = [step + f' {self._tokenizer.eos_token}' for step in proof_steps]

            if any(_targets is None for _targets in proof_steps):
                raise ValueError()

            if self._lm_type == LMType.SEQ_2_SEQ:
                forward_inputs.update(_prepare_tokenized_inputs(prompts_w_partial_proof, self._max_prompt_length))
                forward_inputs["labels"] = _prepare_tokenized_targets(_proof_steps_w_eos, self._max_length - self._max_prompt_length)["input_ids"]
                forward_inputs["labels"] = _mask_labels_by_ignore_index(forward_inputs["labels"],
                                                                        forward_inputs["attention_mask"])

            elif self._lm_type == LMType.CAUSAL:
                # just for getting length
                _prompts = [prompt + CAUSAL_LM_END_OF_PROMPT for prompt in prompts_w_partial_proof]

                if self._include_prompt_for_causal_lm_loss:
                    prompt_lengths = None
                else:
                    prompt_ids = [
                        _prepare_tokenized_inputs(
                            [prompt],
                            self._max_length,
                            padding='longest',
                            return_length=True,
                            # add_special_tokens=False,
                        )
                        for prompt in _prompts
                    ]
                    prompt_lengths = [_promt_ids['length'][0] for _promt_ids in prompt_ids]

                inputs_with_targets = [f'{prompt}{proof_step}'
                                       for prompt, proof_step in zip(_prompts, _proof_steps_w_eos)]
                forward_inputs.update(_prepare_tokenized_inputs(inputs_with_targets, self._max_length))
                forward_inputs["labels"] = forward_inputs['input_ids'].detach().clone()

                forward_inputs["labels"] = _mask_labels_by_ignore_index(
                    forward_inputs["labels"],
                    mask_lengths=prompt_lengths,
                    attention_mask=forward_inputs["attention_mask"],
                )
            else:
                raise NotImplementedError()

        elif mode == 'generation':
            if self._lm_type == LMType.SEQ_2_SEQ:
                forward_inputs.update(_prepare_tokenized_inputs(prompts_w_partial_proof, self._max_prompt_length))

            elif self._lm_type == LMType.CAUSAL:
                _prompts = [prompt + CAUSAL_LM_END_OF_PROMPT for prompt in prompts_w_partial_proof]

                # def _prepare_tokenized_inputs(inputs, max_length, padding=padding, **kwargs):
                forward_inputs.update(
                    _prepare_tokenized_inputs(
                        _prompts,
                        self._max_prompt_length,
                    )
                )

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

        if self.log_examples:
            inputs_decoded = self._tokenizer.batch_decode(_unmask_by_pad_token(forward_inputs['input_ids']))
            if 'labels' in forward_inputs:
                labels_decoded = self._tokenizer.batch_decode(_unmask_by_pad_token(forward_inputs['labels']))
            else:
                labels_decoded = [None] * len(inputs_decoded)

            for i_example, (input_decoded, label_decoded) in enumerate(zip(inputs_decoded, labels_decoded)):
                if self._log_only_first_example and i_example > 0:
                    break
                logger.info('------------ [example=%d] tokenized inputs ----------------', i_example)
                logger.info(input_decoded)
                if label_decoded is not None:
                    logger.info('------------ [example=%d] tokenized labels ----------------', i_example)
                    logger.info(label_decoded)

        forward_inputs.update(self._get_features(examples))

        return forward_inputs

    def compute_metrics(self, eval_preds) -> Dict[str, Any]:

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        examples = self.eval_dataset

        # Replace ignore_indexs used for padding as we can't decode them
        preds = _unmask_by_pad_token(preds)
        decoded_preds = self._tokenizer.batch_decode(preds, skip_special_tokens=True)

        results = {}

        prediction_lens = [np.count_nonzero(pred != self._tokenizer.pad_token_id) for pred in preds]
        results["gen_len"] = np.mean(prediction_lens)

        metrics: Dict[str, List[Any]] = defaultdict(list)
        for i_example, (pred_proof, example) in enumerate(zip(decoded_preds, examples)):

            logger.info('')
            logger.info('')
            logger.info('================ compute_metrics() example=[%d] ================\n', i_example)

            facts, hypothesis, gold_proof = self._get_logic(example, 'eval')

            if self._lm_type == LMType.CAUSAL:
                # the results from model generation include also the prompt
                prompt = self._tokenizer.decode(_unmask_by_pad_token(example["input_ids"]),
                                                skip_special_tokens=True)
                if prompt in pred_proof:
                    pred_proof = pred_proof[len(prompt):]

            log_example(
                facts=facts,
                hypothesis=hypothesis,
                gold_proofs=[gold_proof],
                pred_proof=pred_proof,
                logger=logger,
                warn_on_prettify_failure=self._warn_on_example_prettify_failure,
            )

            if example is not None:
                _metrics = self._compute_metrics(example, pred_proof)
                log_metrics(_metrics, logger=logger)
                for metric_name, metric_val in _metrics.items():
                    metrics[metric_name].append(metric_val)

        for metric_name, metric_vals in metrics.items():
            results[f"{metric_name}"] = np.mean(metric_vals)

        logger.info('-------- compute_metrics() done! ------------------')
        logger.info('\n' + pformat(results))

        return results

    @abstractmethod
    def _make_in_out(
        self,
        example,
        split: str,
    ) -> Tuple[str, str, str]:
        pass

    @abstractmethod
    def _compute_metrics(self, example, pred_proof: str):
        pass

    @abstractmethod
    def _get_features(self, examples) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_logic(self, example, split: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        pass
