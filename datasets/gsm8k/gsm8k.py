import argparse
import csv
from pathlib import Path


def process_filtered_csv(input_csv: str, output_csv: str) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = row["answer"].split("####")[-1].strip()
            rows.append((row["question"], target))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input", "target"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} examples to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a filtered GSM8K-Hard CSV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/gsm8k/gsm8k_test_filtered.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_filtered_csv(args.input, args.output)


if __name__ == "__main__":
    main()
