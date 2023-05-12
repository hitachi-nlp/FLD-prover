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
            else:
                return 'UNKNOWN'

        dic['answer'] = convert_answer(dic['answer'])
        if 'negative_answer' in dic:
            dic['negative_answer'] = convert_answer(dic['negative_answer'])

    elif version == '0.1':
        pass

    else:
        raise ValueError()

    return DeductionExample.parse_obj(dic)
