import logging
from .ruletaker import evaluate_ruletaker

logger = logging.getLogger()

try:
    from .entailmentbank import evaluate_entailmentbank
except ImportError as e:
    logger.error('[TO BE FIXED] importing modules from ete3 failed. the original error message is the following:\n%s', str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["entailmentbank", "ruletaker", "FLNL"])
    parser.add_argument("--path", type=str)
    parser.add_argument("--path-val", type=str)
    parser.add_argument("--path-test", type=str)
    parser.add_argument("--skip-intermediates", action="store_true")
    parser.add_argument("--output-pdf", type=str, help="Path for outputing the PDF")
    args = parser.parse_args()
    print(args)

    if args.dataset == "entailmentbank":
        results = json.load(open(args.path))
        em, f1 = evaluate_entailmentbank(
            results, not args.skip_intermediates, args.output_pdf
        )
        print("Exact match: ", em)
        print("F1: ", f1)
    else:
        results_val = json.load(open(args.path_val))
        results_test = json.load(open(args.path_test))
        (
            answer_accuracies_val,
            proof_accuracies_val,
            proof_scores_val,
            answer_accuracies_test,
            proof_accuracies_test,
            proof_scores_test,
        ) = evaluate_ruletaker(results_val, results_test)
        print("Validation results:")
        print("Answer: ", answer_accuracies_val)
        print("Proof: ", proof_accuracies_val)
        print("Proof Scores: ", proof_scores_val)
        print("Testing results:")
        print("Answer: ", answer_accuracies_test)
        print("Proof: ", proof_accuracies_test)
        print("Proof Scores: ", proof_scores_test)
