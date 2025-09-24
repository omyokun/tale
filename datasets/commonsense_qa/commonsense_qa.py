# dataset_prep.py
from datasets import load_dataset
import csv
import os

def prepare_dataset(split, output_csv):
    # Load the specified split of CommonsenseQA from HF
    dataset = load_dataset("tau/commonsense_qa", split=split, cache_dir="/tmpdir/m24047nmmr/pruning/datasets/cache")
    
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "question_concept", "choice_A", "choice_B", "choice_C", "choice_D", "choice_E", "answer_key"])
        
        for item in dataset:
            id_val = item["id"]
            question = item["question"]
            question_concept = item["question_concept"]
            choices = item["choices"]
            answer_key = item["answerKey"]
            
            # Extract choices - CommonsenseQA has 5 choices (A, B, C, D, E)
            choice_texts = choices["text"]  # List of choice texts
            choice_labels = choices["label"]  # List of labels ['A', 'B', 'C', 'D', 'E']
            
            # Create a mapping from labels to texts
            choice_mapping = dict(zip(choice_labels, choice_texts))
            
            # Write row with all choices as separate columns
            writer.writerow([
                id_val,
                question, 
                question_concept,
                choice_mapping.get('A', ''),  # Choice A
                choice_mapping.get('B', ''),  # Choice B  
                choice_mapping.get('C', ''),  # Choice C
                choice_mapping.get('D', ''),  # Choice D
                choice_mapping.get('E', ''),  # Choice E
                answer_key
            ])
    
    print(f"CSV for split '{split}' saved as {output_csv}")
    print(f"Total examples: {len(dataset)}")

# Prepare validation CSV for CommonsenseQA
#prepare_dataset("validation", "/tmpdir/m24047nmmr/pruning/datasets/commonsense_qa/commonsenseqa_validation.csv")
prepare_dataset("train", "/tmpdir/m24047nmmr/pruning/datasets/commonsense_qa/commonsenseqa_train.csv")