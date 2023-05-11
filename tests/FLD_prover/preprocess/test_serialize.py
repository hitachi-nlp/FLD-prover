from typing import Dict, Any, List
from pprint import pprint

from FLD_prover.preprocess.serialize import serialize_example
from FLD_prover.schema import DeductionExample, SerializedDeductionStep


def test_serialize_example():

    def test_one_example(input_example: DeductionExample,
                         gold_examples: List[SerializedDeductionStep],
                         **kwargs):
        print('\n\n\n=================== _test_one_example() ==================')

        print('\n------------    input_example   ------------')
        pprint(input_example.dict())
        serialized_example = serialize_example(input_example, **kwargs)

        print('\n------------ serialized_example ------------')
        pprint(serialized_example.dict())

        for i_gold, gold_example in enumerate(gold_examples):
            print('')
            print(f'------------       gold-{i_gold}         ------------')
            pprint(gold_example.dict())
            if serialized_example.input == gold_example.input\
                    and serialized_example.next_step == gold_example.next_step:
                return

        assert False

    hypothesis = 'this is the hypothesis'
    context = 'sent1: this is sentence1 sent2: this is sentence2 sent3: this is sentence3'
    proof = 'sent1 & sent2 -> int1: the conclusion of sentence1 and sentence2; sent3 & int1 -> int2: the conclusion of int1 and sent3;'

    test_one_example(
        DeductionExample.parse_obj({
            'hypothesis': hypothesis,
            'context': context,
            'proofs': [proof],
            'proof_stance': 'PROOF',
            'answer': 'True',
        }),
        [
            SerializedDeductionStep.parse_obj({
                'input': f'$hypothesis$ = {hypothesis} ; $context$ = {context} ; $proof$ = ',
                'next_step': proof + ' __PROVED__',
            }),
        ],
        stepwise=False,
    )

    for _ in range(0, 10):
        test_one_example(
            DeductionExample.parse_obj({
                'hypothesis': hypothesis,
                'context': context,
                'proofs': [proof],
                'proof_stance': 'PROOF',
                'answer': 'True',
            }),
            [
                SerializedDeductionStep.parse_obj({
                    'input': f'$hypothesis$ = {hypothesis} ; $context$ = {context} ; $proof$ = ',
                    'next_step': 'sent1 & sent2 -> int1: the conclusion of sentence1 and sentence2;',
                }),
                SerializedDeductionStep.parse_obj({
                    'input': f'$hypothesis$ = {hypothesis} ; $context$ = {context} ; $proof$ = sent1 & sent2 -> int1: the conclusion of sentence1 and sentence2;',
                    'next_step': 'sent3 & int1 -> int2: the conclusion of int1 and sent3; __PROVED__',
                }),

            ],
            stepwise=True,
        )


if __name__ == '__main__':
    test_serialize_example()
