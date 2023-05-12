from typing import List
from FLD_task.evaluate.scoring import (
    calc_levenstein_similarity_batch,
    calc_rouge_batch,
    calc_bleurt_similarity_batch,
    LEVENSTEIN_SIMILARITY_THRESHOLD,
    ROUGE_THRESHOLD,
    BLEURT_SIMILARITY_THRESHOLD,
)


similarities = {
    calc_levenstein_similarity_batch: LEVENSTEIN_SIMILARITY_THRESHOLD,
    calc_rouge_batch: ROUGE_THRESHOLD,
    # calc_bleurt_similarity_batch: BLEURT_SIMILARITY_THRESHOLD,
}


def test_similarity_thresholds():

    def check_threshold(gold: str, preds: List[str], preds_are_correct: bool):
        print('\n\n\n ========================================================')
        for pred in preds:
            print(f'gold: "{gold}"')
            print(f'pred: "{pred}"')
            print
            for calc_similarity_batch, threshold in similarities.items():
                sim = calc_similarity_batch([gold], [pred])[0]
                print(f'        {calc_similarity_batch.__name__:<35}    {sim:.4f}    {threshold:.4f}')
                if preds_are_correct:
                    assert sim >= threshold
                else:
                    assert sim < threshold
            print('\n')

    check_threshold(
        'hoge',
        ['hoge'],
        True,
    )

    check_threshold(
        'We ate many foods at that restaurant',
        [
            'We ate many foods at that restaurant',
        ],
        True,
    )

    check_threshold(
        'a tempter that is chauvinistic and does not single results in a stagflation that substantiates',
        [
            'a tempter that is chauvinistic and does not single results in a stagflation that substantiates',

            'if a tempter is chauvinistic and does not single then a stagflation substantiates',
            'when a tempter is chauvinistic and does not single then a stagflation substantiates',
            'a stagflation substantiates when a tempter is chauvinistic and does not single',

            'a tempter is chauvinistic and does not single thus a stagflation substantiates',

            'a tempter that is chauvinistic and does not single forces a stagflation to substantiate',
        ],
        True,
    )

    check_threshold(
        'that that a irruption occurs is not the fact leads to a abidance',
        [
            'that that a irruption occurs is not the fact leads to a abidance',
            'if a irruption occurs then a abidance occurs',
            'a abidance occurs when a irruption occurs ',
        ],
        True,
    )

    check_threshold(
        'if a abidance occurs, then it is wrong that either a kickoff occurs or a ricochet occurs or both',
        [
            'if a abidance occurs, then it is wrong that either a kickoff occurs or a ricochet occurs or both',

            'if a abidance occurs, then it is not the fact that either a kickoff occurs or a ricochet occurs or both',
            'it is not the fact that either a kickoff occurs or a ricochet occurs or both when a abidance occurs',
        ],
        True,
    )


if __name__ == '__main__':
    test_similarity_thresholds()
