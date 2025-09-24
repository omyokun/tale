# dataset_prep.py
from datasets import load_dataset
import csv
import os

def prepare_dataset(subset, split, output_csv):
    # Load the specified subset and split of Winogrande from HF
    dataset = load_dataset("allenai/winogrande", subset, split=split, cache_dir="/tmpdir/m24047nmmr/pruning/datasets/cache")
    
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "option1", "option2", "answer"])
        
        for item in dataset:
            sentence = item["sentence"]
            option1 = item["option1"]
            option2 = item["option2"]
            answer = item["answer"]  # "1" or "2" indicating which option is correct
            
            writer.writerow([
                sentence,
                option1,
                option2,
                answer
            ])
    
    print(f"CSV for subset '{subset}' split '{split}' saved as {output_csv}")
    print(f"Total examples: {len(dataset)}")

# Prepare validation CSV for Winogrande debiased subset
#prepare_dataset("winogrande_debiased", "validation", "/tmpdir/m24047nmmr/pruning/datasets/winogrande/winogrande_debiased_validation.csv")
prepare_dataset("winogrande_debiased", "train", "/tmpdir/m24047nmmr/pruning/datasets/winogrande/winogrande_debiased_train.csv")
