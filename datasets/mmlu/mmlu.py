import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def prepare_dataset(
    subset: str,
    split: str,
    output_csv: str,
    cache_dir: str | None,
    limit: int | None,
) -> None:
    dataset = load_dataset("cais/mmlu", subset, split=split, cache_dir=cache_dir)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question", "choice_A", "choice_B", "choice_C", "choice_D", "answer", "subject"])

        for item in dataset:
            answer_letter = chr(ord("A") + int(item["answer"]))
            writer.writerow(
                [
                    item["question"],
                    item["choices"][0],
                    item["choices"][1],
                    item["choices"][2],
                    item["choices"][3],
                    answer_letter,
                    item["subject"],
                ]
            )

    print(f"Saved {len(dataset)} examples to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MMLU CSVs.")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default="data/mmlu/mmlu_all_validation.csv")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(args.subset, args.split, args.output, args.cache_dir, args.limit)


if __name__ == "__main__":
    main()
