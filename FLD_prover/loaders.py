from FLD_prover.schema import DeductionExample


def load(dic: dict) -> DeductionExample:
    version = dic.get('__version__', '0.0')
    example = DeductionExample.parse_obj(dic)
    return example
