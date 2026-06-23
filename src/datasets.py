"""
Dataset loaders for greedy layer pruning.

Each loader returns a dict with keys:
  items          – list of sample dicts
  format_chat    – fn(item) → list[dict]   chat turns for apply_chat_template
  get_answer     – fn(item) → str          gold answer string
  extract_pred   – fn(generated_text) → str  clean prediction from model output
  max_new_tokens – int                     tokens to generate per sample

Supported datasets
------------------
arc_challenge, arc_easy, boolq, bigbench, common_qa, gsm8k, mmlu, winogrande, math500

Expected CSV column names
-------------------------
arc_challenge / arc_easy  : question, choice_label, choice_text, answer_key
boolq                     : question, passage, answer_label
bigbench                  : input, target
common_qa                 : question, choice_A … choice_E, answer_key
gsm8k                     : input, target
mmlu                      : question, choice_A … choice_D, answer, subject
winogrande                : sentence, option1, option2, answer
math500                   : problem (or input), solution (or target)
"""
import re
import pandas as pd


# ─── ARC Challenge / ARC Easy ────────────────────────────────────────────────

def load_arc(csv_path):
    df = pd.read_csv(csv_path)
    items = []
    for q, group in df.groupby("question"):
        choices = [(row["choice_label"], row["choice_text"]) for _, row in group.iterrows()]
        items.append({"question": q, "choices": choices,
                      "answer": group["answer_key"].iloc[0]})

    def format_chat(item):
        body = item["question"] + "\n" + "\n".join(
            f"({l}) {t}" for l, t in item["choices"]
        )
        return [
            {"role": "system", "content": (
                "You are a science expert. Read the question and choose the best answer "
                "(A, B, C, or D). Respond with only the answer letter."
            )},
            {"role": "user", "content": body},
        ]

    def get_answer(item):
        return item["answer"]

    def extract_pred(text):
        t = text.strip().upper()
        return t[0] if t and t[0] in "ABCD" else "INVALID"

    return dict(items=items, format_chat=format_chat, get_answer=get_answer,
                extract_pred=extract_pred, max_new_tokens=1)


# ─── BoolQ ───────────────────────────────────────────────────────────────────

def load_boolq(csv_path):
    df = pd.read_csv(csv_path)
    items = [
        {"question": r["question"], "passage": r["passage"], "answer": r["answer_label"]}
        for _, r in df.iterrows()
    ]

    def format_chat(item):
        body = (
            f"Passage: {item['passage']}\n\n"
            f"Question: {item['question']}\n"
            "(A) True\n(B) False"
        )
        return [
            {"role": "system", "content": (
                "Answer true/false questions based on the passage. "
                "Reply with only 'A' for True or 'B' for False."
            )},
            {"role": "user", "content": body},
        ]

    def get_answer(item):
        return item["answer"]

    def extract_pred(text):
        t = text.strip().upper()
        return t[0] if t and t[0] in "AB" else "INVALID"

    return dict(items=items, format_chat=format_chat, get_answer=get_answer,
                extract_pred=extract_pred, max_new_tokens=1)


# ─── BigBench Hard (Boolean Expressions) ─────────────────────────────────────

def load_bigbench(csv_path):
    df = pd.read_csv(csv_path)
    items = [
        {"input": r["input"], "answer": str(r["target"]).strip()}
        for _, r in df.iterrows()
    ]

    def format_chat(item):
        return [
            {"role": "system", "content": (
                "Evaluate boolean expressions. "
                "Respond with exactly one word: 'True' or 'False'."
            )},
            {"role": "user", "content": f"{item['input']}\n\nAnswer (True or False):"},
        ]

    def get_answer(item):
        return item["answer"]

    def extract_pred(text):
        t = text.strip().lower()
        if "true" in t:
            return "True"
        if "false" in t:
            return "False"
        return text.strip()

    return dict(items=items, format_chat=format_chat, get_answer=get_answer,
                extract_pred=extract_pred, max_new_tokens=5)


# ─── CommonsenseQA ───────────────────────────────────────────────────────────

def load_common_qa(csv_path):
    df = pd.read_csv(csv_path)
    items = []
    for _, r in df.iterrows():
        choices = [
            (l, r[f"choice_{l}"])
            for l in "ABCDE"
            if pd.notna(r.get(f"choice_{l}")) and str(r[f"choice_{l}"]).strip()
        ]
        items.append({"question": r["question"], "choices": choices,
                      "answer": r["answer_key"]})

    def format_chat(item):
        body = item["question"] + "\n" + "\n".join(
            f"({l}) {t}" for l, t in item["choices"]
        )
        return [
            {"role": "system", "content": (
                "Answer commonsense multiple-choice questions. "
                "Choose A, B, C, D, or E. Respond with only the answer letter."
            )},
            {"role": "user", "content": body},
        ]

    def get_answer(item):
        return item["answer"]

    def extract_pred(text):
        t = text.strip().upper()
        return t[0] if t and t[0] in "ABCDE" else "INVALID"

    return dict(items=items, format_chat=format_chat, get_answer=get_answer,
                extract_pred=extract_pred, max_new_tokens=1)


# ─── GSM8K ───────────────────────────────────────────────────────────────────

def _extract_number(text):
    """Pull the final numeric answer from generated text."""
    if "####" in text:
        after = text.split("####")[-1].strip()
        nums = re.findall(r"-?\d+\.?\d*", after)
        if nums:
            try:
                return str(int(float(nums[0])))
            except (ValueError, OverflowError):
                return nums[0]
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if nums:
        try:
            return str(int(float(nums[-1])))
        except (ValueError, OverflowError):
            return nums[-1]
    return ""


def load_gsm8k(csv_path):
    df = pd.read_csv(csv_path)
    items = [
        {"question": r["input"], "answer": str(r["target"])}
        for _, r in df.iterrows()
    ]

    def format_chat(item):
        return [
            {"role": "system", "content": (
                "Solve math problems step by step. "
                "At the end write your answer as: #### [number]"
            )},
            {"role": "user", "content": item["question"]},
        ]

    def get_answer(item):
        return _extract_number(item["answer"]) or item["answer"].strip()

    def extract_pred(text):
        return _extract_number(text)

    return dict(items=items, format_chat=format_chat, get_answer=get_answer,
                extract_pred=extract_pred, max_new_tokens=256)


# ─── MMLU ────────────────────────────────────────────────────────────────────

def load_mmlu(csv_path):
    df = pd.read_csv(csv_path)
    items = []
    for _, r in df.iterrows():
        choices = [(l, r[f"choice_{l}"]) for l in "ABCD"]
        items.append({
            "question": r["question"],
            "choices": choices,
            "answer": r["answer"],
            "subject": r.get("subject", ""),
        })

    def format_chat(item):
        body = item["question"] + "\n" + "\n".join(
            f"({l}) {t}" for l, t in item["choices"]
        )
        return [
            {"role": "system", "content": (
                "Answer academic multiple-choice questions across all subjects. "
                "Choose A, B, C, or D. Respond with only the answer letter."
            )},
            {"role": "user", "content": body},
        ]

    def get_answer(item):
        return item["answer"]

    def extract_pred(text):
        t = text.strip().upper()
        return t[0] if t and t[0] in "ABCD" else "INVALID"

    return dict(items=items, format_chat=format_chat, get_answer=get_answer,
                extract_pred=extract_pred, max_new_tokens=1)


# ─── Winogrande ──────────────────────────────────────────────────────────────

def load_winogrande(csv_path):
    df = pd.read_csv(csv_path)
    items = [
        {
            "sentence": r["sentence"],
            "option1": r["option1"],
            "option2": r["option2"],
            "answer": str(r["answer"]),
        }
        for _, r in df.iterrows()
    ]

    def format_chat(item):
        body = (
            f"Sentence: {item['sentence']}\n"
            f"Option 1: {item['option1']}\n"
            f"Option 2: {item['option2']}\n"
            "Which option fills the blank (_)?"
        )
        return [
            {"role": "system", "content": (
                "Complete sentences using commonsense reasoning. "
                "Respond with only '1' or '2'."
            )},
            {"role": "user", "content": body},
        ]

    def get_answer(item):
        return item["answer"]

    def extract_pred(text):
        t = text.strip()
        if "1" in t:
            return "1"
        if "2" in t:
            return "2"
        return "INVALID"

    return dict(items=items, format_chat=format_chat, get_answer=get_answer,
                extract_pred=extract_pred, max_new_tokens=1)


# ─── MATH-500 ────────────────────────────────────────────────────────────────

def load_math500(csv_path):
    df = pd.read_csv(csv_path)
    q_col = "problem" if "problem" in df.columns else "input"
    a_col = "solution" if "solution" in df.columns else "target"
    items = [
        {"question": r[q_col], "answer": str(r[a_col])}
        for _, r in df.iterrows()
    ]

    def format_chat(item):
        return [
            {"role": "system", "content": (
                "Solve math problems step by step. "
                "End your answer with '#### [final answer]'."
            )},
            {"role": "user", "content": item["question"]},
        ]

    def get_answer(item):
        return _extract_number(item["answer"]) or item["answer"].strip()

    def extract_pred(text):
        return _extract_number(text)

    return dict(items=items, format_chat=format_chat, get_answer=get_answer,
                extract_pred=extract_pred, max_new_tokens=512)


# ─── Registry ─────────────────────────────────────────────────────────────────

DATASET_LOADERS = {
    "arc_challenge": load_arc,
    "arc_easy":      load_arc,
    "boolq":         load_boolq,
    "bigbench":      load_bigbench,
    "common_qa":     load_common_qa,
    "gsm8k":         load_gsm8k,
    "mmlu":          load_mmlu,
    "winogrande":    load_winogrande,
    "math500":       load_math500,
}

# Mapping from dataset name → lm-eval task identifier
LMEVAL_TASK_MAP = {
    "arc_challenge": "arc_challenge",
    "arc_easy":      "arc_easy",
    "boolq":         "boolq",
    "bigbench":      "bbh_zeroshot_boolean_expressions",
    "common_qa":     "commonsense_qa",
    "gsm8k":         "gsm8k",
    "mmlu":          "mmlu",
    "winogrande":    "winogrande",
}


def load_dataset(name, csv_path):
    if name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {list(DATASET_LOADERS)}"
        )
    return DATASET_LOADERS[name](csv_path)
