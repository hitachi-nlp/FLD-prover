from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import Seq2SeqTrainer
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.utils import (
    is_torch_tpu_available,
)
from transformers.trainer_utils import (
    EvalPrediction,
    EvalLoopOutput,
    has_length,
    denumpify_detensorize,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl


logger = logging.getLogger(__name__)


class ForceCallMetricsSeq2SeqTrainer(Seq2SeqTrainer):
    """ calll self.compute_metrics() even if labels are None """

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:

        eval_loop_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        args = self.args
        all_preds = eval_loop_output.predictions
        all_labels = eval_loop_output.label_ids

        # if all_labels is None, compute_metrics is not called in the super class
        if self.compute_metrics is not None and all_preds is not None and all_labels is None:
            if args.include_inputs_for_metrics:
                # metrics = self.compute_metrics(
                #     EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                # )
                raise ValueError()
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        metrics = denumpify_detensorize(metrics)

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                eval_loop_output.metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return eval_loop_output


class StepWiseGenerationTrainer(ForceCallMetricsSeq2SeqTrainer):

    def __init__(
        self,
        *args,
        max_steps: Optional[int] = 20,
        texts_to_inputs_func: Optional[Callable[[List[str]], Dict[str, Union[torch.Tensor, Any]]]] = None,
        is_finished_func: Optional[Callable[[str], bool]] = None,
        # drop_labels_for_eval=False,
        log_generation=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_steps = max_steps
        self._texts_to_inputs_func = texts_to_inputs_func
        self._is_finished_func = is_finished_func
        self._log_generation = log_generation

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        # **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        labels = None
        n_examples = len(list(inputs.values())[0])
        input_seqs = self._tensor_to_seqs(inputs['input_ids'])
        extended_input_seqs = input_seqs
        extended_pred_seqs: List[str] = [''] * n_examples
        are_finished: Dict[int, bool] = {i_example: False for i_example in range(0, n_examples)}

        for i_step in range(0, self._max_steps):
            # shoud used inputs for i_step=0 to get labels
            extended_inputs = inputs.copy() if i_step == 0 else self._seqs_to_tensors(extended_input_seqs)

            # drop custom fields so that super() do not raise exception
            gold_proofs = extended_inputs.pop('gold_proofs', None)

            _, _preds, _labels = super().prediction_step(
                model,
                extended_inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

            if gold_proofs is not None:
                extended_inputs['gold_proofs'] = gold_proofs

            # we need these option to get the scores. They are allowed only in the higher version of transformers.
            # However, if we use higher version of transformers, we need to re-implemente this class.
            # self._gen_kwargs.update({
            #     'return_dict_in_generate': True,
            #     'output_scores': True,
            # })
            # hoge = super().prediction_step(
            #     model,
            #     extended_inputs,
            #     prediction_loss_only,
            #     ignore_keys=ignore_keys,
            # )
            # self._gen_kwargs.pop('return_dict_in_generate', None)
            # self._gen_kwargs.pop('output_scores', None)

            if isinstance(_preds, tuple):
                _preds = _preds[0]

            # update
            if i_step == 0 and 'labels' in inputs:
                labels = inputs['labels']
                labels = labels.to(self.model.device)

            num_return_sequences = self._gen_kwargs.get('num_return_sequences', 1)
            _pred_seqs = self._tensor_to_seqs(_preds)
            if num_return_sequences == 1:
                nbest_pred_seqs = [[_pred_seqs[i_example]] for i_example in range(n_examples)]
            else:
                if n_examples == 1:
                    nbest_pred_seqs = [_pred_seqs]
                else:
                    raise NotImplementedError()

            # extend sequence
            for i_example in range(n_examples):
                extended_input_seq = extended_input_seqs[i_example]
                extended_pred_seq = extended_pred_seqs[i_example]
                _nbest_pred_seqs = nbest_pred_seqs[i_example]
                onebest_pred_seq = nbest_pred_seqs[i_example][0]

                if self._log_generation:
                    logger.info('---------    example=%d   step=%d   ----------', i_example, i_step)
                    logger.info('partial_input:')
                    logger.info('    %s', extended_input_seq)
                    logger.info('')
                    for i_nbest, nbest_pred_seq in enumerate(_nbest_pred_seqs):
                        logger.info('nbest=%d', i_nbest)
                        logger.info('    %s', nbest_pred_seq)

                if not are_finished[i_example]:
                    if i_step == 0:
                        extended_pred_seq_next = onebest_pred_seq
                    else:
                        extended_pred_seq_next = ' '.join([extended_pred_seq, onebest_pred_seq])
                    extended_pred_seqs[i_example] = extended_pred_seq_next
                    extended_input_seqs[i_example] = input_seqs[i_example] + extended_pred_seq_next
                    if self._is_finished_func(onebest_pred_seq):
                        are_finished[i_example] = True

            if all(are_finished.values()):
                break

        if not all(are_finished.values()):
            logger.warning('The step-wise generation is not finished within max_steps=%d. will return the incomplete results', self._max_steps)

        # return loss=None computing the loss is not implemented.
        # it is compliated to compare the gold text with step-wisely generated texts.
        return None, self._seqs_to_tensors(extended_pred_seqs)['input_ids'], labels

    def _seqs_to_tensors(self, extended_seqs: List[str]) -> Dict[str, torch.Tensor]:
        extended_inputs_no_tensor = self._texts_to_inputs_func(extended_seqs)
        n_examples = len(list(extended_inputs_no_tensor.values())[0])
        extended_inputs_no_tensor_itemized = [
            {key: extended_inputs_no_tensor[key][i_example]
             for key in extended_inputs_no_tensor.keys()}
            for i_example in range(0, n_examples)
        ]

        collated = self.data_collator(extended_inputs_no_tensor_itemized)

        return {key: tensor.to(self.args.device) for key, tensor in collated.items()}

    def _tensor_to_seqs(self, preds: torch.Tensor) -> List[str]:
        # Replace -100s used for padding as we can't decode them
        preds = torch.where(preds != -100, preds, self.tokenizer.pad_token_id)
        return self.tokenizer.batch_decode(preds, skip_special_tokens=True)
