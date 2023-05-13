import math
import re

from FLD_task.evaluate.scoring import calc_score
from common import calc_F
from logger_setup import setup as setup_logger


def _calc_score(gold: str, pred: str, *args, **kwargs) -> float:
    return calc_score(
        re.sub('  *', ' ', gold),
        re.sub('  *', ' ', pred),
        *args,
        **kwargs,
    )


def test_calc_score_on_toy_examples():
    # prediction have irrelevant steps
    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',

            'sent4 & sent7 -> int5: this is wrong sentence A',
            'int5 & int8 -> int9: this is wrong sentence B',
        ]),

        zero_one=True,
    )
    assert math.isclose(score, 0.0)

    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',

            'sent4 & sent7 -> int5: this is wrong sentence A',
            'int5 & int8 -> int9: this is wrong sentence B',
        ]),

        zero_one=True,
        allowed_additional_proof_steps=1,
    )
    assert math.isclose(score, 0.0)

    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',

            'sent4 & sent7 -> int5: this is wrong sentence A',
            'int5 & int8 -> int9: this is wrong sentence B',
        ]),

        zero_one=True,
        allowed_additional_proof_steps=2,
    )
    assert math.isclose(score, 1.0)

    # prediction lacks steps
    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int2 & int4 -> hypothesis;',
        ]),

        zero_one=True,
    )
    assert (score == 0.0)

    # prediction have irrelevant steps + zero_one=False
    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',

            'sent4 & sent7 -> int5: this is wrong sentence A',
            'int5 & int8 -> int9: this is wrong sentence B',
        ]),

        zero_one=False,
    )
    assert (math.isclose(score, calc_F(5, 5, 2)[-1]))

    # prediction lacks steps + zero_one=False
    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',

            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
        ]),

        zero_one=False,
    )
    assert (math.isclose(score, calc_F(5, 3, 0)[-1]))

    # prediction is perfect
    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',
            'int2 & int4 -> hypothesis;',
        ]),

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    # The tree structure is corret. sntence content is wrong.
    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',

            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',

            'int1 -> int4: this is a sentence D',
            'int3 -> int2: this is a sentence C',

            'int2 & int4 -> hypothesis;',
        ]),

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    # context reordering of the previous example
    score = _calc_score(
        '; '.join([
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',

            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'int1 -> int4: this is a sentence D',
            'int3 -> int2: this is a sentence C',

            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',

            'int2 & int4 -> hypothesis;',
        ]),

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    # prediction is perfect
    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',

            'int2 & int4 -> hypothesis',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',

            'int2 & int4 -> hypothesis',

        ]),

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',
            '[assump1] & sent7 -> int5: hoge',

            'int2 & int4 -> hypothesis',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',
            '[assump1] & sent7 -> int5: hoge',

            'int2 & int4 -> hypothesis',

        ]),

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',
            '[assump1] & sent7 -> int5: hoge',

            'void -> assump2: fuga',
            '[assump2] & sent8 -> int6: fuga',

            'int2 & int4 -> hypothesis',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',
            '[assump1] & sent7 -> int5: hoge',

            'void -> assump2: fuga',
            '[assump2] & sent8 -> int5: fuga',

            'int2 & int4 -> hypothesis',

        ]),

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',
            '[assump1] & sent7 -> int5: hoge',

            'void -> assump2: fuga',
            '[assump2] & sent8 -> int6: fuga',

            'int2 & int4 -> hypothesis',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump2: hoge',
            '[assump2] & sent7 -> int5: hoge',

            'void -> assump1: fuga',
            '[assump1] & sent8 -> int6: fuga',

            'int2 & int4 -> hypothesis',

        ]),

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',
            '[assump1] & sent7 -> int5: hoge',

            'void -> assump2: fuga',
            '[assump2] & sent8 -> int6: fuga',

            'int2 & int4 -> hypothesis',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',
            '[assump1] & sent8 -> int6: fuga',

            'void -> assump2: fuga',
            '[assump2] & sent7 -> int5: hoge',

            'int2 & int4 -> hypothesis',

        ]),

        zero_one=True,
    )
    assert (math.isclose(score, 0.0))

    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',
            '[assump1] & sent7 -> int5: hoge',

            'void -> assump2: fuga',
            '[assump2] & sent8 -> int6: fuga',

            'int2 & int4 -> hypothesis',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A',
            'sent2 & sent6 -> int3: this is a sentence B',
            'int1 -> int2: this is a sentence C',
            'int3 -> int4: this is a sentence D',

            'void -> assump1: hoge',
            '[assump1] & sent8 -> int6: fuga',

            'void -> assump2: fuga',
            '[assump2] & sent7 -> int5: hoge',

            'int2 & int4 -> hypothesis',

        ]),

        zero_one=False,
    )
    assert (math.isclose(score, calc_F(9, 7, 2)[-1]))


def test_calc_score_on_real_examples():
    # the actual gold and precisions from our dataset
    score = _calc_score(
        'sent3 & sent5 -> int1: a tempter that is chauvinistic and does not single results in a stagflation that substantiates; sent6 & sent1 -> int2: a tempter does not single and is chauvinistic; int1 & int2 -> int3: a stagflation substantiates; int3 -> hypothesis;',

        'sent6 & sent1 -> int1: a tempter does not single and is chauvinistic; sent5 & sent3 -> int2: a tempter that is not single and is chauvinistic leads to a stagflation that substantiates; int2 & int1 -> int3: a stagflation substantiates; int3 -> hypothesis',

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        'sent3 & sent2 -> int1: that that a irruption occurs is not the fact leads to a abidance; sent5 -> int2: if a abidance occurs, then it is wrong that either a kickoff occurs or a ricochet occurs or both; int1 & int2 -> int3: if it is wrong that a irruption occurs, then it is not the fact that either a kickoff occurs or a ricochet occurs or both; sent6 & sent1 -> int4: if it is not true that a layout occurs, then either a kickoff occurs or a ricochet occurs or both; int4 -> int5: if it is wrong that either a kickoff occurs or a ricochet occurs or both, then a layout occurs; int3 & int5 -> hypothesis;',

        'sent2 & sent3 -> int1: that that a irruption occurs is not true results in a abidance; sent5 -> int2: if a abidance occurs, then it is wrong that either a kickoff occurs or a ricochet occurs or both; int1 & int2 -> int3: if it is wrong that a irruption occurs, then it is not true that either a kickoff occurs or a ricochet occurs or both; sent6 & sent1 -> int4: that that a layout occurs is not true results in that either a kickoff occurs or a ricochet occurs or both; int4 -> int5: if it is wrong that either a kickoff occurs or a ricochet occurs or both, then a layout occurs; int5 & int3 -> hypothesis;',

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        'sent6 & sent1 -> int1: if a telencephalon either finds or is serviceable or both, then it is wrong that a curbstone leaks; sent2 & sent3 -> int2: if it is wrong that a curbstone leaks, then it is wrong that a television smashes; int1 & int2 -> int3: if a telencephalon either finds or is serviceable or both, then it is not the fact that a television smashes; sent7 -> int4: a telencephalon finds; int4 -> int5: a telencephalon either finds or is serviceable or both; int3 & int5 -> hypothesis;',

        'sent7 -> int1: a telencephalon finds; int1 -> int2: a telencephalon either finds or is serviceable or both; sent1 & sent6 -> int3: if a telencephalon either finds or is serviceable or both, then it is not true that a curbstone leaks; sent2 & sent3 -> int4: if it is wrong that a curbstone leaks, then it is not the fact that a television smashes; int4 & int3 -> int5: a telencephalon that either finds or is serviceable or both leads to a television that does not smash; int5 & int2 -> hypothesis',

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        'sent8 & sent7 -> int1: if a conceptualization occurs, then it is not true that a flap and a return do not occur; sent3 & int1 -> int2: if a reduplication does not occur, then it is not true that a flap and a return do not occur; sent4 & sent5 -> int3: a confutation or a babysitting or both leads to that that a reduplication occurs is wrong; sent9 -> int4: either a confutation occurs or a babysitting occurs or both; int3 & int4 -> int5: it is not true that a reduplication occurs; int2 & int5 -> hypothesis;',

        'sent9 -> int1: either a confutation occurs or a babysitting occurs or both; sent8 & sent7 -> int2: if a conceptualization occurs, then it is not the fact that a flap and a return do not occur; sent4 & sent5 -> int3: a confutation or a babysitting or both causes that that a reduplication occurs is not the fact; sent3 & int1 -> int4: if either a confutation occurs or a babysitting occurs or both, then a conceptualization occurs; int3 & int4 -> int5: that either a confutation occurs or a babysitting occurs or both results in that a conceptualization occurs; int5 & int2 -> hypothesis',

        zero_one=True,
    )
    assert (math.isclose(score, 0.0))

    score = _calc_score(
        'sent9 -> int1: that that a overspill occurs is not true causes that that a Mills occurs is not true; sent3 & int1 -> int2: a psalmody and a coaching results in that that a Mills occurs is not the fact; sent1 & sent6 -> int3: a psalmody occurs; sent4 & sent2 -> int4: a coaching occurs; int3 & int4 -> int5: a psalmody and a coaching occurs; int2 & int5 -> hypothesis;',

        'sent6 & sent1 -> int1: a psalmody occurs; sent2 & sent4 -> int2: a coaching occurs; int2 & int1 -> int3: a psalmody and a coaching occurs; sent9 -> int4: if it is wrong that a overspill occurs, then it is not true that a Mills occurs; int4 & sent3 -> int5: that a psalmody and a coaching occurs causes that that a Mills occurs is not true; int3 & int5 -> hypothesis',

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        'sent7 & sent5 -> int1: a commemorative ketone leads to a semiterrestrial flail; int1 -> int2: a flail that is not semiterrestrial results in a ketone that is not commemorative; sent6 & sent3 -> int3: if a bacchante either behooves or blazes or both, then it is not true that a flail is semiterrestrial; sent1 -> int4: a bacchante either behooves or blazes or both; int3 & int4 -> int5: it is wrong that a flail is semiterrestrial; int2 & int5 -> hypothesis;',

        'sent1 -> int1: a bacchante either behooves or blazes or both; sent6 & sent3 -> int2: if a bacchante either behooves or blazes or both, then it is wrong that a flail is semiterrestrial; sent7 & sent5 -> int3: a commemorative ketone results in a flail that is semiterrestrial; int1 & int2 -> int4: it is not true that a flail is semiterrestrial; int3 -> int5: if it is wrong that a flail is semiterrestrial, then it is not true that a ketone is commemorative; int5 & int4 -> hypothesis',

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        'sent6 -> int1: that that a about-face occurs is wrong leads to that that a mercy occurs is wrong; sent4 & int1 -> int2: a plasmapheresis and a piss-up results in that that a mercy occurs is wrong; sent2 -> int3: a plasmapheresis occurs; sent5 -> int4: a piss-up occurs; int3 & int4 -> int5: a plasmapheresis and a piss-up occurs; int2 & int5 -> hypothesis;',

        'sent5 -> int1: a piss-up occurs; sent2 -> int2: a plasmapheresis occurs; int1 & int2 -> int3: a plasmapheresis and a piss-up occurs; sent6 -> int4: that that a about-face occurs is not true results in that that a mercy occurs is not true; int4 & sent4 -> int5: that a plasmapheresis and a piss-up occurs results in that that a mercy occurs is not true; int5 & int3 -> hypothesis',

        zero_one=True,
    )
    assert (math.isclose(score, 1.0))

    score = _calc_score(
        'sent8 & sent7 -> int1: if a conceptualization occurs, then it is not true that a flap and a return do not occur; sent3 & int1 -> int2: if a reduplication does not occur, then it is not true that a flap and a return do not occur; sent4 & sent5 -> int3: a confutation or a babysitting or both leads to that that a reduplication occurs is wrong; sent9 -> int4: either a confutation occurs or a babysitting occurs or both; int3 & int4 -> int5: it is not true that a reduplication occurs; int2 & int5 -> hypothesis;',

        'sent9 -> int1: either a confutation occurs or a babysitting occurs or both; sent8 & sent7 -> int2: if a conceptualization occurs, then it is not the fact that a flap and a return do not occur; sent4 & sent5 -> int3: a confutation or a babysitting or both causes that that a reduplication occurs is not the fact; sent3 & int1 -> int4: if either a confutation occurs or a babysitting occurs or both, then a conceptualization occurs; int3 & int4 -> int5: that either a confutation occurs or a babysitting occurs or both results in that a conceptualization occurs; int5 & int2 -> hypothesis',

        zero_one=True,
    )
    assert (math.isclose(score, 0.0))

    score = _calc_score(
        'sent7 &  sent8          ->           int1:       that either a headway occurs or a spoil occurs or both leads to that a repossession occurs;'
        'sent5 &  sent9          ->           int2:       that a repossession occurs leads to that that a saraband occurs is not true;'
        ' int1 &   int2          ->           int3:       that either a headway occurs or a spoil occurs or both causes that that a saraband occurs is not true;'
        'sent2                   ->           int4:       a headway occurs;'
        ' int4                   ->           int5:       either a headway occurs or a spoil occurs or both;'
        ' int3 &   int5          ->     hypothesis;',

        'sent2                   ->           int1:       a headway occurs;'
        ' int1                   ->           int2:       either a headway occurs or a spoil occurs or both;'
        'sent5 &  sent9          ->           int3:       that a repossession occurs results in that that a saraband occurs is not true;'
        'sent7 &  sent8          ->           int4:       that either a headway occurs or a spoil occurs or both causes that a repossession occurs;'
        ' int3 &   int4          ->           int5:       that either a headway occurs or a spoil occurs or both results in that that a saraband occurs is not the fact;'
        ' int2 &   int5          ->     hypothesis;',

        # zero_one=True,
        zero_one=False,
    )
    assert (math.isclose(score, 1.0))


def _check_limitation():
    # The tree structure is corret. sntence content is wrong.
    score = _calc_score(
        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A ;',
            'sent2 & sent6 -> int3: this is a sentence B ;',
            'int1 -> int2: this is a sentence C ;',
            'int3 -> int4: this is a sentence D ;',
            'int2 & int4 -> hypothesis;',
        ]),

        '; '.join([
            'sent1 & sent5 -> int1: this is a sentence A ;',
            'sent2 & sent6 -> int3: this is a sentence B ;',
            'int1 -> int4: this is a sentence D ;',
            'int3 -> int2: this is a sentence C ;',
            'int2 & int4 -> hypothesis;',
        ]),

        zero_one=True,
    )
    assert (score >= 0.99)


if __name__ == '__main__':
    setup_logger()

    test_calc_score_on_toy_examples()
    test_calc_score_on_real_examples()
    # _check_limitation()
