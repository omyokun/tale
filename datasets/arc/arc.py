import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def prepare_dataset(config: str, split: str, output_csv: str, cache_dir: str | None) -> None:
    dataset = load_dataset("allenai/ai2_arc", config, split=split, cache_dir=cache_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "choice_text", "choice_label", "answer_key"])

        for item in dataset:
            for label, choice in zip(item["choices"]["label"], item["choices"]["text"]):
                writer.writerow([item["question"], choice, label, item["answerKey"]])

    print(f"Saved {len(dataset)} examples to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ARC-Easy or ARC-Challenge CSVs.")
    parser.add_argument("--config", choices=["ARC-Easy", "ARC-Challenge"], default="ARC-Easy")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default="data/arc/arc_easy_validation.csv")
    parser.add_argument("--cache-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(args.config, args.split, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
