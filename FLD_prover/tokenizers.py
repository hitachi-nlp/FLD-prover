from transformers import AutoTokenizer, LlamaTokenizer
import logging

logger = logging.getLogger(__name__)


def load(name: str,
         cache_dir=None,
         use_auth_token=False,
         use_fast_tokenizer=True,
         revision="main",
         trust_remote_code=True):
    if name.startswith('stabilityai'):
        tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1",
                                                   additional_special_tokens=['▁▁'],
                                                   use_auth_token=True if use_auth_token else None)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            cache_dir=cache_dir,
            use_fast=use_fast_tokenizer,
            revision=revision,
            use_auth_token=True if use_auth_token else None,
            trust_remote_code=trust_remote_code,
        )

    if tokenizer.eos_token == tokenizer.pad_token:
        # If the eos token is the same as the pad token,
        # the eos token in the labels will be replaced to ignore token (i.e., -100) as well as the pad tokens,
        # and the models will not learn to predict the eos token at the end of text.
        # see the followings:
        #     - https://github.com/huggingface/transformers/issues/22794#issuecomment-1573966012
        #     - https://github.com/huggingface/transformers/issues/22794#issuecomment-1598977285
        if name.find('ELYZA') >= 0:
            # elyza defaults: eos='</s>', pad='</s>'
            tokenizer.pad_token = '<PAD>'
        else:
            logger.critical('I have verified that the above hack works with ELYZA,'
                            'but not with other models. Please implement the hack for other models.')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if name == "line-corporation/japanese-large-lm-3.6b-instruction-sft":
        # For this model, we can not load the remote setting properly, possible due to the sentencepiece mapping.
        # We force reset the pad token.
        tokenizer.pad_token = '<pad>'

    return tokenizer
