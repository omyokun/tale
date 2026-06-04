import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def prepare_boolq_dataset(split: str, output_csv: str, cache_dir: str | None) -> None:
    dataset = load_dataset("google/boolq", split=split, cache_dir=cache_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "passage", "answer", "answer_label"])

        for item in dataset:
            answer_label = "A" if item["answer"] else "B"
            answer_text = "True" if item["answer"] else "False"
            writer.writerow([item["question"], item["passage"], answer_text, answer_label])

    print(f"Saved {len(dataset)} examples to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare BoolQ CSVs.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default="data/boolq/boolq_validation.csv")
    parser.add_argument("--cache-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_boolq_dataset(args.split, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
