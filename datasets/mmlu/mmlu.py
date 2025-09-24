# dataset_prep.py
from datasets import load_dataset
import csv
import os

def prepare_dataset(subset, split, output_csv):
    # Load the specified subset and split of MMLU from HF
    dataset = load_dataset("cais/mmlu", subset, split=split, cache_dir="/tmpdir/m24047nmmr/pruning/datasets/cache")
    dataset = dataset.select(range(10000))

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "choice_A", "choice_B", "choice_C", "choice_D", "answer", "subject"])
        
        for item in dataset:
            question = item["question"]
            choices = item["choices"]  # This is a list of 4 choices
            answer = item["answer"]    # This is the index (0, 1, 2, 3) corresponding to A, B, C, D
            subject = item["subject"]  # Subject name
            
            # Convert answer index to letter (0->A, 1->B, 2->C, 3->D)
            answer_letter = chr(ord('A') + answer)
            
            # Write row with all choices as separate columns
            writer.writerow([
                question, 
                choices[0],  # Choice A
                choices[1],  # Choice B  
                choices[2],  # Choice C
                choices[3],  # Choice D
                answer_letter,
                subject
            ])
    
    print(f"CSV for subset '{subset}' split '{split}' saved as {output_csv}")
    print(f"Total examples: {len(dataset)}")

# Prepare validation CSV for "all" subset
#prepare_dataset("all", "validation", "/tmpdir/m24047nmmr/pruning/datasets/mmlu/mmlu_all_validation.csv")
prepare_dataset("all", "auxiliary_train", "/tmpdir/m24047nmmr/pruning/datasets/mmlu/mmlu_all_train.csv")