import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def prepare_dataset(split: str, output_csv: str, cache_dir: str | None) -> None:
    dataset = load_dataset("tau/commonsense_qa", split=split, cache_dir=cache_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "question",
                "question_concept",
                "choice_A",
                "choice_B",
                "choice_C",
                "choice_D",
                "choice_E",
                "answer_key",
            ]
        )

        for item in dataset:
            choice_mapping = dict(zip(item["choices"]["label"], item["choices"]["text"]))
            writer.writerow(
                [
                    item["id"],
                    item["question"],
                    item["question_concept"],
                    choice_mapping.get("A", ""),
                    choice_mapping.get("B", ""),
                    choice_mapping.get("C", ""),
                    choice_mapping.get("D", ""),
                    choice_mapping.get("E", ""),
                    item["answerKey"],
                ]
            )

    print(f"Saved {len(dataset)} examples to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CommonsenseQA CSVs.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default="data/commonsense_qa/commonsenseqa_validation.csv")
    parser.add_argument("--cache-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(args.split, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
