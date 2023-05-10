from typing import Dict, Any
from pprint import pprint

from FLD_prover.preprocess.serialize import serialize_example
from FLD_prover.schema import DeductionExample, SerializedDeductionExample


def test_serialize_example():

    def test_one_example(input_example: DeductionExample,
                         gold_example: SerializedDeductionExample):
        print('\n\n\n=================== _test_one_example() ==================')
        print('\n------ input_example ------')
        pprint(input_example.dict())
        serialized_example = serialize_example(input_example)
        print('\n------ serialized_example ------')
        pprint(serialized_example.dict())

        assert serialized_example.input_text == gold_example.input_text
        assert serialized_example.output_text == gold_example.output_text

    example = DeductionExample(**{
        'hypothesis': 'this is the hypothesis',
        'context': 'sent1: this is sentence1 sent2: this is sentence2 sent3: this is sentence3',
        'proofs': [
            'sent1 & sent2 -> int1: the conclusion of sentence1 and sentence2; sent3 & int1 -> int2: the conclusion of int1 and sent3;'
        ],
        'proof_stance': 'UNKNOWN',
        'answer': 'Unknown',
    })

    test_one_example(
        example,
        SerializedDeductionExample(**{
            'input_text': example.context,
            'output_text': example.hypothesis,
        })
    )


if __name__ == '__main__':
    test_serialize_example()
