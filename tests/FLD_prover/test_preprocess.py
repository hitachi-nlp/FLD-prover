from typing import Dict, Any
from pprint import pprint

from FLD_prover.preprocess import _preprocess_example


def test_preprocess_example():

    def _test_one_example(input_example: Dict[str, Any],
                          gold_example: Dict[str, Any]):
        print('\n\n=================== _test_one_example() ==================')
        print('------ input_example ------')
        pprint(input_example)
        preprocessed_example = _preprocess_example(input_example)
        print('------ preprocessed_example ------')
        pprint(preprocessed_example)

        for gold_key, gold_val in gold_example.items():
            assert gold_key in preprocessed_example
            assert preprocessed_example[gold_key] == 'val'


if __name__ == '__main__':
    test_preprocess_example()
