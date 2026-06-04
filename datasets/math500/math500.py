import argparse
import csv
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET = "HuggingFaceH4/MATH-500"
OUTPUT_FIELDS = ["problem", "answer", "solution", "subject", "level", "unique_id"]


def prepare_math500_dataset(
    dataset_name: str,
    split: str,
    output_csv: str,
    cache_dir: str | None,
) -> None:
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_FIELDS)

        for item in dataset:
            if "problem" not in item or "answer" not in item:
                raise KeyError("MATH500 examples must contain 'problem' and 'answer' fields.")
            writer.writerow([item.get(field, "") for field in OUTPUT_FIELDS])

    print(f"Saved {len(dataset)} examples to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MATH-500 CSVs.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="data/math500/MATH-500.csv")
    parser.add_argument("--cache-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_math500_dataset(args.dataset, args.split, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
