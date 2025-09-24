# boolq_dataset_prep.py
from datasets import load_dataset
import csv
import os

def prepare_boolq_dataset(split, output_csv):
    # Load the specified split of BoolQ from HF
    dataset = load_dataset("google/boolq", split=split, cache_dir="/tmpdir/m24047nmmr/pruning/datasets/cache")
    
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        # BoolQ has different structure - question, passage, and boolean answer
        writer.writerow(["question", "passage", "answer", "answer_label"])
        
        for item in dataset:
            question = item["question"]
            passage = item["passage"]
            answer = item["answer"]  # This is boolean (True/False)
            
            # Convert boolean to standard format
            answer_label = "A" if answer else "B"  # True = A, False = B
            answer_text = "True" if answer else "False"
            
            writer.writerow([question, passage, answer_text, answer_label])

    print(f"CSV for {split} saved as {output_csv}")
    print(f"Total samples in {split}: {len(dataset)}")

# Create output directory if it doesn't exist
os.makedirs("/tmpdir/m24047nmmr/pruning/datasets/boolq", exist_ok=True)

# Prepare only validation CSV as requested
# prepare_boolq_dataset("validation", "/tmpdir/m24047nmmr/pruning/datasets/boolq/boolq_validation.csv")
prepare_boolq_dataset("train", "/tmpdir/m24047nmmr/pruning/datasets/boolq/boolq_train.csv")


print("\nBoolQ dataset structure:")
print("- Question: A question about the passage")
print("- Passage: A paragraph of text containing the information")
print("- Answer: True or False")
print("- Answer Label: A (True) or B (False)")