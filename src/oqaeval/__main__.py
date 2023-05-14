"""
Basic usage:
python -m oqaeval /path/to/predict_file.jsonl
"""
import argparse
import logging
import numpy as np

from .eval import evaluate_file

logging.basicConfig(
    level=logging.INFO, format="%(levelname).1s %(asctime)s [ %(message)s ]", handlers=[logging.StreamHandler()]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predict_file",
        type=str,
        help="Path to predict file",
    )
    parser.add_argument("--dataset", type=str, default=None, help="dataset path")
    parser.add_argument(
        "--annotation", type=str, default=None, help="tsv file including additional answer annotations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=(
            "text-davinci-003",
            "gpt-3.5-turbo",
            "gpt-4",
        ),
        help="OpenAI GPT flavours",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (used in OpenAI API)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature used in generation (used in OpenAI API)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output file containing API outputs (default: will be saved in `--predict_file` directory in a tsv format)",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        default=False,
        help="Whether to overwrite cached evaluation results from a previous run (only when OpenAI API used)",
    )

    args = parser.parse_args()

    result = evaluate_file(
        args.predict_file,
        args.dataset,
        args.annotation,
        args.output,
        args.model,
        args.max_tokens,
        args.temperature,
        args.overwrite_cache,
        return_per_sample=True,
    )

    em_scores = result["EM"]
    f1_scores = result["F1"]

    print(f"EM: {100.0 * np.mean(em_scores):.2f} ({np.sum(em_scores)}/{len(em_scores)})")
    print(f"F1: {100.0 * np.mean(f1_scores):.2f}")

    if "AnnotatedEM" in result:
        annotated_em_scores = result["AnnotatedEM"]
        print(
            f"AnnotatedEM: {100.0 * np.mean(annotated_em_scores):.2f} ({np.sum(annotated_em_scores)}/{len(annotated_em_scores)})"
        )

    if args.model in result:
        gpt_scores = result[args.model]
        print(f"{args.model}: {100.0 * np.mean(gpt_scores):.2f} ({np.sum(gpt_scores)}/{len(gpt_scores)})")


if __name__ == "__main__":
    main()
