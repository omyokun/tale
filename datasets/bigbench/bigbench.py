import argparse
import csv
from pathlib import Path

import requests


DEFAULT_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/boolean_expressions.json"


def download_and_prepare_dataset(url: str, output_csv: str) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input", "target"])

        for item in data["examples"]:
            writer.writerow([item["input"], item["target"]])

    print(f"Saved {len(data['examples'])} examples to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare BIG-Bench Hard Boolean Expressions CSVs.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="data/bigbench/bigbenchhard_boolean_expressions_train.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    download_and_prepare_dataset(args.url, args.output)


if __name__ == "__main__":
    main()
