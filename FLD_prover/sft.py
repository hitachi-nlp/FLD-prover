from typing import List, Optional
import logging

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from rec_adam.trainer import build_optimizer_from_trainer
from transformers.utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

logger = logging.getLogger(__name__)


class RecAdamSFTTrainer(SFTTrainer):

    def __init__(
        self,
        *args,
        rec_adam_target_task_weight=1.0,
        rec_adam_fisher_coef=3000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rec_adam_target_task_weight = rec_adam_target_task_weight
        self._rec_adam_fisher_coef = rec_adam_fisher_coef


    def create_optimizer(self):
        return build_optimizer_from_trainer(self,
                                            rec_adam_target_task_weight=self._rec_adam_target_task_weight,
                                            rec_adam_fisher_coef=self._rec_adam_fisher_coef)


def build_sft_trainer(dataset_name: str,
                      tokenizer,
                      block_size: int,
                      lang: Optional[str] = None,
                      sft_neftune_noise_alpha=None):

    if dataset_name in ['jhu-cogsci/hans', 'hpprc/janli']:
        task_type = 'two_choice_nli'
        label_maping = {
            0: 'entailment',
            1: 'not entailment',
        }

    elif dataset_name in ['RobZamp/sick', 'hpprc/jsick',
                          'stanfordnlp/snli', 'shunk031/jsnli']:
        task_type = 'three_choice_nli'

        if dataset_name in ['RobZamp/sick', 'hpprc/jsick'] :
            # TODO: we have to implement preprocessing for this dataset, as they are not in the usual format
            # see here: https://huggingface.co/datasets/RobZamp/sick
            raise NotImplementedError()
        elif dataset_name in ['stanfordnlp/snli', 'shunk031/jsnli']:
            label_maping = {
                0: 'entailment',
                1: 'neutral',
                2: 'contradiction',
            }

    elif dataset_name in ['cais/mmlu',
                          'nlp-waseda/JMMLU',
                          'databricks/databricks-dolly-15k',
                          'llm-jp/databricks-dolly-15k-ja']:
        task_type = 'instrution'
        label_maping = None

    elif dataset_name.startswith('FR.'):
        task_type = 'factorized_reasoning'
        label_maping = None
    else:
        raise ValueError(dataset_name)
    logger.info('SFT task type is set to %s', task_type)

    if lang is None:
        if dataset_name in ['hpprc/janli', 'hpprc/jsick', 'llm-jp/databricks-dolly-15k-ja']:
            lang = 'jpn'
        else:
            lang = 'eng'
    logger.info('SFT language is set to %s', lang)

    if lang == 'eng':
        intro = 'Please answer the question based on the given context.'
        context_template = '### context'
        instruction_template = '### question'
        response_template = '### answer'

    elif lang == 'jpn':
        intro = '文脈に基づいて、質問に答えてください｡'
        context_template = '### 文脈'
        instruction_template = '### 質問'
        response_template = '### 回答'

    else:
        raise ValueError(lang)

    def make_formatted_texts(examples):
        formatted_texts = []

        def guess_field(candidate_fields: List[str], not_found='raise') -> str:
            field = None
            for candidate in candidate_fields:
                if candidate in examples:
                    field = candidate
                    break
            if field is None:
                msg = f'candidate fields {str(candidate_fields)} not found in the examples'
                if not_found == 'raise':
                    raise ValueError(msg)
                elif not_found == 'warning':
                    logger.warning(msg)
                else:
                    raise ValueError()
            return field

        for i in range(len(list(examples.values())[0])):

            def guess_value(candidate_fields: List[str], not_found='raise') -> Optional[str]:
                field = guess_field(candidate_fields, not_found=not_found)
                if field is None:
                    return None
                else:
                    return examples[field][i]

            if task_type in ['two_choice_nli', 'three_choice_nli']:
                premise = guess_value(['premise'])
                hypothesis = guess_value(['hypothesis'])
                label = guess_value(['label'])

                instruction = '\n'.join([
                    'Does the premise entails the hypothesis or not?',
                    'premise: ' + premise,
                    'hypothesis: ' + hypothesis,
                ])
                context = None

                if label_maping is not None:
                    if label in label_maping:
                        answer = label_maping[label]
                    else:
                        logger.warning(f'label {label} not found in the label mapping, will be skipped')
                        continue
                else:
                    answer = label

            elif task_type == 'instrution':
                instruction = guess_value(['instruction', 'question'])
                context = guess_value(['context'], not_found='warning')
                answer = guess_value(['response', 'answer'])
                if label_maping is not None:
                    raise Exception()

            elif task_type == 'factorized_reasoning':
                instruction = examples['prompt'][i]
                context = None
                answer = examples['answer'][i]

            else:
                raise ValueError(task_type)

            answer = str(answer)
            if context is not None:
                formatted_text = '\n\n'.join([intro, instruction_template, instruction, context_template, context, response_template, answer]) + tokenizer.eos_token
            else:
                formatted_text = '\n\n'.join([intro, instruction_template, instruction, response_template, answer]) + tokenizer.eos_token

            formatted_texts.append(formatted_text)

            if i == 0:
                logger.info('Example of the formatted prompt:')
                logger.info(formatted_texts[0])

        return formatted_texts

    trainer_cls = SFTTrainer
    trainer_kwargs = {
        'formatting_func': make_formatted_texts,
        'max_seq_length': block_size,
        'neftune_noise_alpha': sft_neftune_noise_alpha,
    }
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    return trainer_cls, trainer_kwargs, collator


def build_rec_adam_sft_trainer(dataset_name: str,
                               tokenizer,
                               block_size: int,
                               lang: Optional[str] = None,
                               rec_adam_target_task_weight=1.0,
                               rec_adam_fisher_coef=3000,
                               sft_neftune_noise_alpha=None):
    _, trainer_kwargs, collator = build_sft_trainer(dataset_name, tokenizer, block_size, lang, sft_neftune_noise_alpha)
    trainer_kwargs['rec_adam_target_task_weight'] = rec_adam_target_task_weight
    trainer_kwargs['rec_adam_fisher_coef'] = rec_adam_fisher_coef
    return RecAdamSFTTrainer, trainer_kwargs, collator
