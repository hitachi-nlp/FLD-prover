from typing import Union, Optional
from FLD_task.schema import DeductionExample


def load(dic: dict, force_version: str = None) -> DeductionExample:
    if force_version is not None:
        version = force_version
    else:
        version = dic.get('version', None)
        version = version or dic.get('__version__', None)  # back compatibility
        version = version or force_version
        # if force_version is not None and force_version != version:
        #     raise ValueError(f'the forced version {force_version} does not match the found version {version}')
        version = version or '0.0'

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

    elif version == 'DeductionExampleInstance':
        pass

    else:
        raise ValueError()

    dic['version'] = 'DeductionExampleInstance'
    ex = DeductionExample.parse_obj(dic)
    return ex
