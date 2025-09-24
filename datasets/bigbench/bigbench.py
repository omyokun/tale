# dataset_prep.py
import requests
import json
import csv
import os

def download_and_prepare_dataset(output_csv):
    """Download BigBench Hard Boolean Expressions data directly from GitHub"""
    
    # URL for the boolean expressions JSON file
    url = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/boolean_expressions.json"
    
    print(f"Downloading BigBench Hard Boolean Expressions from GitHub...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        print(f"Downloaded {len(data['examples'])} examples")
        
        with open(output_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "target"])
            
            for item in data['examples']:
                input_text = item["input"]
                target = item["target"]
                
                writer.writerow([
                    input_text,
                    target
                ])
        
        print(f"CSV saved as {output_csv}")
        print(f"Total examples: {len(data['examples'])}")
        
    except requests.RequestException as e:
        print(f"Error downloading data: {e}")
        print("Trying alternative approach with local data creation...")
        
        # Fallback: create some sample boolean expressions
        sample_data = [
            ("not ( True ) and ( True )", "False"),
            ("True and not not ( not False )", "False"),
            ("not True or False or ( False )", "False"),
            ("False or not ( True ) and False", "False"),
            ("True or not False and True and False", "True"),
            ("False or not not not False and True", "True"),
            ("not True and ( False or True )", "False"),
            ("True and not False or ( True )", "True"),
            ("not True or ( False and True )", "False"),
            ("not True or ( True or False )", "True"),
        ]
        
        with open(output_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "target"])
            
            for input_text, target in sample_data:
                writer.writerow([input_text, target])
        
        print(f"Created sample CSV with {len(sample_data)} examples at {output_csv}")

# Create the output directory if it doesn't exist
output_dir = "/tmpdir/m24047nmmr/pruning/datasets/bigbench/"
os.makedirs(output_dir, exist_ok=True)

# Download and prepare the dataset
download_and_prepare_dataset(f"{output_dir}/bigbenchhard_boolean_expressions_train.csv")