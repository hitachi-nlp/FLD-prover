from typing import Union
from FLD_task.schema import DeductionExample


def load(dic: dict) -> DeductionExample:
    version = dic.get('__version__', '0.0')

    if version == '0.0':

        def convert_answer(ans: Union[bool, str]) -> str:
            if ans is True:
                return 'PROVED'
            elif ans is False:
                return 'DISPROVED'
            elif ans == 'Unknown':
                return 'UNKNOWN'
            else:
                raise ValueError(f'Unknown answer {ans}')

        dic['answer'] = convert_answer(dic['answer'])
        if dic.get('negative_answer', None) is not None:
            dic['negative_answer'] = convert_answer(dic['negative_answer'])

        def convert_stance(stance: str) -> str:
            if stance == 'PROOF':
                return 'PROVED'
            elif stance == 'DISPROOF':
                return 'DISPROVED'
            elif stance == 'UNKNOWN':
                return 'UNKNOWN'
            else:
                raise ValueError(f'Unknown stance {stance}')

        dic['proof_stance'] = convert_stance(dic['proof_stance'])
        if dic.get('negative_proof_stance', None) is not None:
            dic['negative_proof_stance'] = convert_stance(dic['negative_proof_stance'])

    elif version == '0.1':
        pass

    else:
        raise ValueError()

    return DeductionExample.parse_obj(dic)
