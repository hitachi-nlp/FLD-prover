import math
from typing import Tuple, Dict, Any, Union, Iterable
from enum import Enum
import logging
from pathlib import Path
from pprint import pformat

# import torch
# from transformers import get_cosine_schedule_with_warmup

Example = Dict[str, Any]
Batch = Dict[str, Any]
Answer = Union[bool, str]

logger = logging.getLogger(__name__)

I_DONT_THINK = "i don't think"


def int2answer(int_answer: int) -> Answer:
    if int_answer == 0:
        return True
    elif int_answer == 1:
        return False
    elif int_answer == 2:
        return "Unknown"
    else:
        raise ValueError(f'Unknown int_answer {int_answer}')


def answer2int(answer: Answer) -> int:
    if answer is True:
        return 0
    elif answer is False:
        return 1
    elif answer == "Unknown":
        return 2
    else:
        raise ValueError(f'Unknown answer {answer}')


class DatasetType(Enum):
    Seq2Seq = 'seq2seq'
    RuleTaker = 'ruletaker'
    FLNL = 'FLNL'
    EB = 'entailmentbank'


# def get_optimizers(
#     parameters: Iterable[torch.nn.parameter.Parameter],
#     lr: float,
#     num_warmup_grad_steps: int,
#     num_training_steps: int,
# ) -> Dict[str, Any]:
#     """
#     Get an AdamW optimizer with linear learning rate warmup and cosine decay.
#     """
#     optimizer = torch.optim.AdamW(parameters, lr=lr)
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=num_warmup_grad_steps,
#         num_training_steps=num_training_steps,
#     )
#     return {
#         "optimizer": optimizer,
#         "lr_scheduler": {
#             "scheduler": scheduler,
#             "interval": "step",
#         },
#     }


def get_json_lines(path: str) -> Iterable[str]:
    path = Path(path)
    if path.is_dir():
        for file_path in path.glob('*.jsonl'):
            for line in open(file_path):
                yield line
    else:
        yield from open(path)


def log_example(msg: str, ex: Example) -> None:
    logger.info(msg)
    logger.info('\n' + pformat(ex))


def log_batch(msg: str, batch: Batch) -> None:
    logger.info(msg)
    stats = {}
    for key, value in batch.items():
        stats[f'{key}[0]'] = value[0]
    logger.info('The first items of batch:')
    logger.info('\n' + pformat(stats))


def tokenize_with_log(tokenizer, texts, **kwargs):
    if kwargs.get('truncation', False) is False:
        return tokenizer(texts, **kwargs)

    sub_kwargs = kwargs.copy()
    truncation = sub_kwargs.pop('truncation', False)
    padding = sub_kwargs.pop('padding', False)

    tokens_wo_truncation = tokenizer(
        texts,
        truncation=False,
        padding='longest',
        **sub_kwargs,
    )

    tokens_with_truncation = tokenizer(
        texts,
        truncation=truncation,
        padding=padding,
        **sub_kwargs,
    )

    for _text, _tokens_with_truncation, _tokens_wo_truncation in zip(texts, tokens_with_truncation['input_ids'], tokens_wo_truncation['input_ids']):
        if len(_tokens_with_truncation) < len(_tokens_wo_truncation):
            logger.warning('The input text has %d token ids, but they are truncated into %d ids.',
                           len(_tokens_wo_truncation),
                           len(_tokens_with_truncation))
            logger.warning('The input text is: "%s"', _text)
            logger.warning('tokniezer() options are: %s', str(kwargs))
        elif len(_tokens_with_truncation) == len(_tokens_wo_truncation):
            pass
        elif len(_tokens_with_truncation) > len(_tokens_wo_truncation):
            logger.debug('The input text has %d token ids, but they are up-converted into %d ids. This is no problem for learning but memory inefficient.',
                        len(_tokens_wo_truncation),
                        len(_tokens_with_truncation))
            # logger.debug('The input text is: "%s"', _text)
            # logger.debug('tokniezer() options are: %s', str(kwargs))
    return tokens_with_truncation


def calc_F(gold_tot: int, TP: int, FP: int) -> Tuple[float, float, float]:
    if TP + FP == 0:
        precision = 1.0
    else:
        precision = TP / (TP + FP)

    if gold_tot == 0:
        recall = 1.0
    else:
        recall = TP / gold_tot

    if math.isclose(precision + recall, 0.0):
        F = 0.0
    else:
        F = 2 * precision * recall / (precision + recall)

    return precision, recall, F
