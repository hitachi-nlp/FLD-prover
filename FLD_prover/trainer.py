from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable
import logging

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


class StepWiseGenerationTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        *args,
        max_steps: Optional[int] = 20,
        texts_to_inputs_func: Optional[Callable[[List[str]], Dict[str, Union[torch.Tensor, Any]]]] = None,
        is_finished_func: Optional[Callable[[str], bool]] = None,
        no_loss_when_eval=False,
        log_generation=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_steps = max_steps
        self._texts_to_inputs_func = texts_to_inputs_func
        self._is_finished_func = is_finished_func
        self._log_generation = log_generation
        self._no_loss_when_eval = no_loss_when_eval

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Taken from trainer.py of transformers.
        The code change is minimul: just passing data_loader to self.compute_metrics.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        if self.compute_metrics is not None and all_preds is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs),
                    dataloader=dataloader,
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels),
                                               dataloader=dataloader)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Taken from trainer.py of transformers.
        The code change is minimul: just passing data_loader to self.compute_metrics.
        """
        args = self.args

        if not has_length(dataloader):
            raise ValueError("dataloader must implement a working __len__")

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        inputs_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            inputs_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
                    inputs_gatherer.add_arrays(self._gather_and_numpify(inputs_host, "eval_inputs_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, inputs_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
            inputs_gatherer.add_arrays(self._gather_and_numpify(inputs_host, "eval_inputs_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None
        inputs_ids = inputs_gatherer.finalize() if not prediction_loss_only else None

        # if self.compute_metrics is not None and preds is not None and label_ids is not None:
        if self.compute_metrics is not None and preds is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=preds, label_ids=label_ids, inputs=inputs_ids),
                    dataloader=dataloader,
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=preds, label_ids=label_ids),
                    dataloader=dataloader,
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=preds, label_ids=label_ids, metrics=metrics, num_samples=num_examples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
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

            if self._no_loss_when_eval:
                extended_inputs.pop('labels', None)
            _, _preds, _labels = super().prediction_step(
                model,
                extended_inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

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
