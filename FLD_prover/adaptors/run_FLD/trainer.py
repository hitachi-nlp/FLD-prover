from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Callable

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.configuration_utils import GenerationConfig
from transformers.trainer import Trainer
from transformers.utils import logging
from transformers import Seq2SeqTrainer


if TYPE_CHECKING:
    from .data.data_collator import DataCollator
    from .modeling_utils import PreTrainedModel
    from .tokenization_utils_base import PreTrainedTokenizerBase
    from .trainer_callback import TrainerCallback
    from .trainer_utils import EvalPrediction, PredictionOutput
    from .training_args import TrainingArguments


logger = logging.get_logger(__name__)


class StepWiseGenerationTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        *args,
        max_steps: Optional[int] = 30,
        texts_to_inputs_func: Optional[Callable[[List[str]], Dict[str, Union[torch.Tensor, Any]]]] = None,
        is_finished_func: Optional[Callable[[str], bool]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._max_steps = max_steps
        self._texts_to_inputs_func = texts_to_inputs_func
        self._is_finished_func = is_finished_func

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
            extended_inputs = inputs if i_step == 0 else self._seqs_to_tensors(extended_input_seqs)

            _, _preds, _labels = super().prediction_step(
                model,
                extended_inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

            if isinstance(_preds, tuple):
                _preds = _preds[0]

            # update
            if i_step == 0:
                labels = _labels

            pred_seqs = self._tensor_to_seqs(_preds)

            # log the current sequences
            for i_example, (extended_input_seq, pred_seq) in enumerate(zip(extended_input_seqs, pred_seqs)):
                logger.info('step=%d    example=%d    %s    ----------> %s', i_step, i_example, extended_input_seq, pred_seq)

            # extend sequence
            for i_example, pred_seq in enumerate(pred_seqs):
                if not are_finished[i_example]:
                    extended_pred_seqs[i_example] = ' '.join([extended_pred_seqs[i_example], pred_seq])
                    extended_input_seqs[i_example] = input_seqs[i_example] + extended_pred_seqs[i_example]
                    if self._is_finished_func(pred_seq):
                        are_finished[i_example] = True

            if all(are_finished.values()):
                break

        if all(are_finished.values()):
            logger.warning('DONE!!!!!!!!!!!!!!!!!!! %d', i_step)
        else:
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
        return self.data_collator(extended_inputs_no_tensor_itemized)

    def _tensor_to_seqs(self, preds: torch.Tensor) -> List[str]:
        # Replace -100s used for padding as we can't decode them
        preds = torch.where(preds != -100, preds, self.tokenizer.pad_token_id)
        return self.tokenizer.batch_decode(preds, skip_special_tokens=True)
