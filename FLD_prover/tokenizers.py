from transformers import AutoTokenizer, LlamaTokenizer
import logging

logger = logging.getLogger(__name__)


def load(name: str,
         cache_dir=None,
         use_auth_token=False,
         use_fast_tokenizer=True,
         revision="main",
         trust_remote_code=True):
    if name == 'stabilityai/japanese-stablelm-base-alpha':
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

    if name.find('mistralai') >= 0:
        # See here: https://medium.com/@parikshitsaikia1619/mistral-mastery-fine-tuning-fast-inference-guide-62e163198b06#:~:text=Setting%20Up%20Tokenizer
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
    else:

        # PAD_TOKEN = '[PAD]'
        PAD_TOKEN = '<hono_pad>'
        if name == 'stabilityai/stablelm-2-1_6b':
            # this model allow only pre-registerd tokens
            PAD_TOKEN = '<|extra0|>'

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})

        elif tokenizer.pad_token == tokenizer.eos_token:
            # If the eos token is the same as the pad token,
            # the eos token in the labels will be replaced to ignore token (i.e., -100) as well as the pad tokens,
            # and the models will not learn to predict the eos token at the end of text.
            # see the followings:
            #     - https://github.com/huggingface/transformers/issues/22794#issuecomment-1573966012
            #     - https://github.com/huggingface/transformers/issues/22794#issuecomment-1598977285

            # XXX: this will not replace tokenizer.pad_token_id
            # tokenizer.pad_token = PAD_TOKEN

            tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})

    return tokenizer
