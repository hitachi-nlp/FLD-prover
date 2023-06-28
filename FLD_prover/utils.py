import logging

logger = logging.getLogger()


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
            logger.debug('The input text has %d token ids, but they are up-converted into %d ids. This is no problem for learning but memory inefficient.',
                        len(_tokens_wo_truncation),
                        len(_tokens_with_truncation))
            # logger.debug('The input text is: "%s"', _text)
            # logger.debug('tokniezer() options are: %s', str(kwargs))
    return tokens_with_truncation

