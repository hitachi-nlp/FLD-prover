from typing import Optional, Dict
from pprint import pprint

from proof_common import collapse_proof, add_final_reference, extract_ident_sents
from common import Answer


def test_collapse_proof():
    print('\n\n\n\n\n==============================    test_collapse_proof()     =================================')

    contexts = [
        ("sent1: hoge; sent2: fuga; sent3: piyo; sent4: hogo"),
        ("sent1: hoge; sent2: fuga; sent3: piyo; sent4: hogo"),
        ("sent1: hoge; sent2: fuga; sent3: piyo; sent4: hogo"),
    ]
    proofs = [
        "sent2 -> hypothesis;",
        "sent2 & sent3 -> hypothesis;",
        "sent1 & sent3 -> int1: The dog eats the dog; sent2 & int1 -> int2: The dog chases the dog; sent4 & int2 -> hypothesis;",
    ]

    for context, proof in zip(contexts, proofs):
        collapsed_context, collapsed_proof, dead_ids = collapse_proof(context, proof)
        print('\n\n')
        print('context            :', context)
        print('proof              :', proof)
        print('dead_ids           :', dead_ids)
        print('collapsed_context  :', collapsed_context)
        print('collapsed_proof    :', collapsed_proof)


def test_add_final_reference():
    print('\n\n\n\n\n==============================    test_add_final_reference()     =================================')

    def check(hypothesis: str,
              context: str,
              proof: str,
              answer: Answer,
              gold: str,
              dataset_depth: Optional[int] = None) -> None:
        proof_with_final_reference = add_final_reference(context, hypothesis, proof, answer, dataset_depth=dataset_depth)
        print('\n\n')
        print('hypothesis                      : ', hypothesis)
        print('context                         : ', context)
        print('proof                           : ', proof)
        print('gold                            : ', gold)
        print('proof_with_final_reference      : ', proof_with_final_reference)
        assert proof_with_final_reference == gold

    hypothesis = 'the man is happy'

    check(
        hypothesis,
        'sent1: hoge sent2: the man is happy',
        'sent2 -> hypothesis;',
        True,
        'sent2 -> hypothesis;',
        dataset_depth=0,
    )

    check(
        hypothesis,
        'sent1: hoge sent2: for everyone, he is happy',
        'sent2 -> hypothesis;',
        True,
        'sent2 -> int1: the man is happy; int1 -> hypothesis;',
        dataset_depth=1,
    )

    check(
        hypothesis,
        'sent1: the man eat much sent2: if someone eat much, he is full sent3: if someone is full, he is satisfied sent4: if someone is satisfied, he is happy',
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> hypothesis;',
        True,
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> int3: the man is happy; int3 -> hypothesis;',
    )

    check(
        hypothesis,
        'sent1: the man eat much sent2: if someone eat much, he is full sent3: if someone is full, he is satisfied sent4: if someone is satisfied, he is happy',
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> hypothesis;',
        True,
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> int3: the man is happy; int3 -> hypothesis;',
    )

    check(
        hypothesis,
        'sent1: the man eat much sent2: if someone eat much, he is full sent3: if someone is full, he is satisfied sent4: if someone is satisfied, he is happy',
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> int3: the man is happy; int3 -> hypothesis;',
        True,
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> int3: the man is happy; int3 -> hypothesis;',
    )

    check(
        'the man is very happy',
        'sent1: the man eat much sent2: if someone eat much, he is full sent3: if someone is full, he is satisfied sent4: if someone is satisfied, he is happy',
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> int3: the man is happy; int3 -> hypothesis;',
        True,
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> int3: the man is happy; int3 -> int4: the man is very happy; int4 -> hypothesis;',
    )

    check(
        'the man is not very happy',
        'sent1: the man eat much sent2: if someone eat much, he is full sent3: if someone is full, he is satisfied sent4: if someone is satisfied, he is happy',
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> int3: the man is happy; int3 -> hypothesis;',
        False,
        'sent1 & sent2 -> int1: the man is full; int1 & sent3 -> int2: the man is satisfied; int2 & sent4 -> int3: the man is happy; int3 -> int4: the following is not true: the man is not very happy; int4 -> hypothesis;',
    )

    # real examples
    check(
        'it is not true that, a brutalization happens',
        'sent1: pedaling substance occurs sent10: a TAT occurs sent11: a Mauritanian break-dance sent2: a brutalization dedicate sent3: vilifying peristome occurs sent4: a brutalization occurs sent5: a brutalization backfire sent6: flurrying mongoose occurs sent7: a brutalization nazify sent8: arthralgicness occurs sent9: a earache occurs',
        'sent4 -> hypothesis;',
        False,
        'sent4 -> hypothesis;',
        dataset_depth=0,
    )

    check(
        'the following is not true: it is false that, the Siouan will not underpay snailflower and it is wrong that it does cox',
        'hoge',
        'sent14 & sent4 -> int1: the following is not true: it is false that, the Siouan will not underpay snailflower and it is wrong that it does cox; int1 -> hypothesis',
        False,
        'sent14 & sent4 -> int1: the following is not true: it is false that, the Siouan will not underpay snailflower and it is wrong that it does cox; int1 -> hypothesis',
    )


def test_extract_ident_sent():

    def test(rep: str, gold: Dict[str, str]):
        extracted = extract_ident_sents(rep)

        print('\n\n ====================== test_extract_ident_sent ---------------')
        print('\n------------ extracted -------------')
        pprint(extracted)

        print('\n------------ gold -------------')
        pprint(gold)

        assert tuple(list(sorted(extracted.items()))) == tuple(list(sorted(gold.items())))


    test(
        'sent1: hoge sent2: he present hoge sent3: piyo',
        {
            'sent1': 'hoge',
            'sent2': 'he present hoge',
            'sent3': 'piyo',
        }
    )


    test(
        'sent1 -> int1: piyo; int1 & sent2 -> int2: fuga; int2 -> hypothesis;',
        {
            'int1': 'piyo',
            'int2': 'fuga',
        }
    )

    # test(
    #     'sent1: Eric is still young and round, but he is nice to everyone and kind to animals sent2: If a person is red and green, then they will present as big sent3: Someone who"s blue, nice and round will also be someone who"s green sent4: Alan vowed to always be rough, cold, blue, and big as possible sent5: An individual who is big and nice is also green sent6: A cold and blue appearing person will be young as well sent7: Someone can be big and rough, but cold even though they are kind sent8: A person who is cold, rough and young is also nice sent9: If someone has a red face and feels blue and has a round body then you"ll automatically think they are big sent10: Harry is very young to be a nurse but he loves helping others because he is so nice and kind',
    #     {
    #         'int1': 'piyo',
    #         'int2': 'fuga',
    #     }
    # )
   

if __name__ == '__main__':
    # test_collapse_proof()
    # test_add_final_reference()
    test_extract_ident_sent()
