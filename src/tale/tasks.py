from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch


TASK_NAMES = (
    "arc_easy",
    "arc_challenge",
    "boolq",
    "mmlu",
    "commonqa",
    "winogrande",
    "bigbench",
    "gsm8k_hard",
    "math500",
)


def format_question_with_choices(
    question: str, choices: Sequence[Tuple[str, str]]
) -> str:
    lines = [str(question).strip()]
    for label, text in choices:
        lines.append(f"({label}) {str(text).strip()}")
    return "\n".join(lines)


def format_boolq_question(question: str, passage: str) -> str:
    return (
        f"Passage: {passage}\n\n"
        f"Question: {question}\n"
        "(A) True\n"
        "(B) False"
    )


def format_bigbench_question(input_expr: str) -> str:
    return f"Evaluate this boolean expression: {input_expr}"


def format_winogrande_question(sentence: str, option1: str, option2: str) -> str:
    return (
        f"Sentence: {sentence}\n"
        f"Option 1: {option1}\n"
        f"Option 2: {option2}\n"
        "Which option should fill in the blank (_)?"
    )


def arc_role(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a Science expert assistant. "
                "Your task is to answer multiple-choice science questions at grade-school level. "
                "Each question has answer choices labeled A, B, C, and D. "
                "Respond only with the single correct answer letter."
            ),
        },
        {"role": "user", "content": question},
    ]


def boolq_role(question: str, passage: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You answer True/False questions from passages. "
                "Respond only with 'A' for True or 'B' for False."
            ),
        },
        {"role": "user", "content": format_boolq_question(question, passage)},
    ]


def mmlu_role(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You answer multiple-choice questions across academic subjects. "
                "Respond only with the single best answer letter: A, B, C, or D."
            ),
        },
        {"role": "user", "content": question},
    ]


def commonqa_role(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You answer commonsense multiple-choice questions. "
                "Respond only with the single best answer letter: A, B, C, D, or E."
            ),
        },
        {"role": "user", "content": question},
    ]


def winogrande_role(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You solve pronoun resolution tasks. "
                "Choose the better option and respond only with '1' or '2'."
            ),
        },
        {"role": "user", "content": question},
    ]


def bigbench_role(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a boolean expression evaluator. "
                "Respond with exactly one word: 'True' or 'False'."
            ),
        },
        {"role": "user", "content": f"{question}\n\nAnswer (True or False):"},
    ]


def math_role(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a careful math problem solver. "
                "Show full step-by-step reasoning. "
                "On the final line, write the answer in the form '#### <integer>'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem: {question}\n\n"
                "Solve step by step and finish with the final answer on a new line as: #### <integer>."
            ),
        },
    ]


def load_arc_dataset(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    grouped = df.groupby("question", sort=False)
    records: List[Dict[str, Any]] = []

    for question, group in grouped:
        choices = [
            (str(row["choice_label"]).strip(), str(row["choice_text"]).strip())
            for _, row in group.iterrows()
        ]
        records.append(
            {
                "question": str(question).strip(),
                "choices": choices,
                "target": str(group["answer_key"].iloc[0]).strip(),
            }
        )

    return records


def load_boolq(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    return [
        {
            "question": str(row["question"]).strip(),
            "passage": str(row["passage"]).strip(),
            "target": str(row["answer_label"]).strip(),
        }
        for _, row in df.iterrows()
    ]


def load_mmlu(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        choices = [
            ("A", str(row["choice_A"]).strip()),
            ("B", str(row["choice_B"]).strip()),
            ("C", str(row["choice_C"]).strip()),
            ("D", str(row["choice_D"]).strip()),
        ]
        records.append(
            {
                "question": str(row["question"]).strip(),
                "choices": choices,
                "target": str(row["answer"]).strip(),
            }
        )

    return records


def load_commonqa(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        choices: List[Tuple[str, str]] = []
        for label in ["A", "B", "C", "D", "E"]:
            value = row.get(f"choice_{label}")
            if pd.notna(value) and str(value).strip():
                choices.append((label, str(value).strip()))
        records.append(
            {
                "question": str(row["question"]).strip(),
                "choices": choices,
                "target": str(row["answer_key"]).strip(),
            }
        )

    return records


def load_winogrande(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        answer = str(row["answer"]).strip()
        if answer.endswith(".0"):
            answer = answer[:-2]
        records.append(
            {
                "sentence": str(row["sentence"]).strip(),
                "option1": str(row["option1"]).strip(),
                "option2": str(row["option2"]).strip(),
                "target": answer,
            }
        )

    return records


def load_bigbench(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    return [
        {"input": str(row["input"]).strip(), "target": str(row["target"]).strip()}
        for _, row in df.iterrows()
    ]


def load_gsm8k_hard(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    return [
        {"input": str(row["input"]).strip(), "target": str(row["target"]).strip()}
        for _, row in df.iterrows()
    ]


def load_math500(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    return [
        {"problem": str(row["problem"]).strip(), "target": str(row["answer"]).strip()}
        for _, row in df.iterrows()
    ]


def load_task_dataset(task: str, path: str) -> List[Dict[str, Any]]:
    if task in {"arc_easy", "arc_challenge"}:
        return load_arc_dataset(path)
    if task == "boolq":
        return load_boolq(path)
    if task == "mmlu":
        return load_mmlu(path)
    if task == "commonqa":
        return load_commonqa(path)
    if task == "winogrande":
        return load_winogrande(path)
    if task == "bigbench":
        return load_bigbench(path)
    if task == "gsm8k_hard":
        return load_gsm8k_hard(path)
    if task == "math500":
        return load_math500(path)
    raise ValueError(f"Unsupported task: {task}")


def build_chat_prompt(task: str, record: Dict[str, Any]) -> List[Dict[str, str]]:
    if task in {"arc_easy", "arc_challenge"}:
        question = format_question_with_choices(record["question"], record["choices"])
        return arc_role(question)
    if task == "boolq":
        return boolq_role(record["question"], record["passage"])
    if task == "mmlu":
        question = format_question_with_choices(record["question"], record["choices"])
        return mmlu_role(question)
    if task == "commonqa":
        question = format_question_with_choices(record["question"], record["choices"])
        return commonqa_role(question)
    if task == "winogrande":
        question = format_winogrande_question(
            record["sentence"], record["option1"], record["option2"]
        )
        return winogrande_role(question)
    if task == "bigbench":
        return bigbench_role(format_bigbench_question(record["input"]))
    if task == "gsm8k_hard":
        return math_role(record["input"])
    if task == "math500":
        return math_role(record["problem"])
    raise ValueError(f"Unsupported task: {task}")


def tokenize_chat(
    tokenizer, chat: List[Dict[str, str]], device: torch.device
) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = "\n\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in chat
        )
    inputs = tokenizer(input_text, return_tensors="pt")
    return inputs.input_ids.to(device)

