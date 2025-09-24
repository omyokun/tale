import csv
import os
from datasets import Dataset

def process_filtered_csv(input_csv, output_csv):
    """Read filtered CSV and save in the same format as GSM8K preprocessing (input, target)."""

    print(f"Reading filtered dataset from {input_csv}...")

    filtered_questions = []
    filtered_answers = []

    # Read your filtered dataset
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question"]
            answer = row["answer"]

            # Extract final answer after "####"
            target = answer.split("####")[-1].strip()

            filtered_questions.append(question)
            filtered_answers.append(target)

    # Build HuggingFace dataset object (optional)
    new_dataset = Dataset.from_dict({
        "input": filtered_questions,
        "target": filtered_answers
    })

    print(new_dataset)

    # Save as CSV (like before)
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "target"])
        for q, a in zip(filtered_questions, filtered_answers):
            writer.writerow([q, a])

    print(f"Processed CSV saved as {output_csv}")
    print(f"Total filtered examples: {len(filtered_questions)}")


# Example usage
input_csv = "/tmpdir/m24047krsh/llama_project/llama_layer/data/gsm8k/gsm8k.csv"  # your filtered file
output_csv = "/tmpdir/m24047krsh/llama_project/llama_layer/data/gsm8k/gsm8k_test_filtered.csv"

process_filtered_csv(input_csv, output_csv)
