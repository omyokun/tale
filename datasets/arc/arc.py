# dataset_prep.py
from datasets import load_dataset
import csv
import os

def prepare_dataset(split, output_csv):
    # Load the specified split of ARC-Easy from HF
    #dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, cache_dir="/tmpdir/m24047nmmr/pruning/datasets/cache")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir="/tmpdir/m24047nmmr/pruning/datasets/cache")
    
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "choice_text", "choice_label", "answer_key"])
        
        for item in dataset:
            question = item["question"]
            answer_key = item["answerKey"]
            for choice in item["choices"]["text"]:
                idx = item["choices"]["text"].index(choice)
                label = item["choices"]["label"][idx]
                writer.writerow([question, choice, label, answer_key])

    print(f"CSV for {split} saved as {output_csv}")

# Prepare train and validation CSVs
#prepare_dataset("train", "/tmpdir/m24047nmmr/pruning/datasets/arc/arc_easy_train.csv")
#prepare_dataset("validation", "/tmpdir/m24047nmmr/pruning/datasets/arc/arc_easy_validation.csv")
prepare_dataset("train", "/tmpdir/m24047nmmr/pruning/datasets/arc/arc_challenge_train.csv")
