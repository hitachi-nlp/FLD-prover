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
    log_metrics,
)
from FLD_task.proof import get_stance_markers
from FLD_prover.lm_types import LMType
from FLD_prover.tokenization import unmask_by_pad_token

logger = logging.getLogger()


class Metrics(ABC):

    def __init__(self,
                 tokenizer,
                 eval_dataset,
                 lm_type: LMType,
                 ignore_index=-100):
        self._tokenizer = tokenizer
        self._eval_dataset = eval_dataset
        self._lm_type = lm_type
        self._ignore_index = ignore_index

    def __call__(self, eval_preds) -> Dict[str, Any]:

        def _unmask_by_pad_token(tensor):
            return unmask_by_pad_token(tensor, self._tokenizer.pad_token_id, mask_id=self._ignore_index)

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        examples = self._eval_dataset

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

            facts, hypothesis, gold_proof = self._get_logic(example)

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
            )

            if example is not None:
                _metrics = self._compute_metrics_from_example(example, pred_proof)
                log_metrics(_metrics, logger=logger)
                for metric_name, metric_val in _metrics.items():
                    metrics[metric_name].append(metric_val)

        for metric_name, metric_vals in metrics.items():
            results[f"{metric_name}"] = np.mean(metric_vals)

        logger.info('-------- compute_metrics() done! ------------------')
        logger.info('\n' + pformat(results))

        return results

    @abstractmethod
    def _get_logic(self, example) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        pass

    @abstractmethod
    def _compute_metrics_from_example(self, example, pred_proof: str):
        pass
