# Data

This directory is the default location for prepared CSV files. Dataset files are ignored by git.

Expected filenames:

```text
data/arc/arc_easy_validation.csv
data/arc/arc_challenge_validation.csv
data/boolq/boolq_validation.csv
data/commonsense_qa/commonsenseqa_validation.csv
data/mmlu/mmlu_all_validation.csv
data/winogrande/winogrande_debiased_validation.csv
data/bigbench/bigbenchhard_boolean_expressions_train.csv
data/gsm8k/gsm8k_test_filtered.csv
data/math500/MATH-500.csv
```

Use the scripts under `datasets/` to prepare the CSVs where possible.
