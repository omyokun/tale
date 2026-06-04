import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def prepare_dataset(subset: str, split: str, output_csv: str, cache_dir: str | None) -> None:
    dataset = load_dataset("allenai/winogrande", subset, split=split, cache_dir=cache_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence", "option1", "option2", "answer"])

        for item in dataset:
            writer.writerow([item["sentence"], item["option1"], item["option2"], item["answer"]])

    print(f"Saved {len(dataset)} examples to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare WinoGrande CSVs.")
    parser.add_argument("--subset", default="winogrande_debiased")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default="data/winogrande/winogrande_debiased_validation.csv")
    parser.add_argument("--cache-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(args.subset, args.split, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
