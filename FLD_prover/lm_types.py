from enum import Enum


class LMType(Enum):
    SEQ_2_SEQ = 'seq2seq'
    CAUSAL = 'causal'
