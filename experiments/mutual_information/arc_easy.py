import torch.nn as nn
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM
)
import pandas as pd
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import time
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv(os.environ.get("TALE_DATA_PATH", "data/arc/arc_easy_validation.csv"))
grouped = df.groupby('question')
question_list = []
choices_by_question = []
answer_keys = []
print(f"dataset length :{len(df)}")

tokenizer = AutoTokenizer.from_pretrained(os.environ.get("TALE_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"))
model_base = LlamaForCausalLM.from_pretrained(os.environ.get("TALE_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"))

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_base = model_base.to(device)

class ModifiedLlamaModel(nn.Module):
    def __init__(self, original_model, delete_indices=set()):
        super().__init__()
        self.config = original_model.config
        self.embed_tokens = original_model.model.embed_tokens
        self.layers = nn.ModuleList([
            layer for i, layer in enumerate(original_model.model.layers)
            if i not in delete_indices
        ])
        self.norm = original_model.model.norm
        self.lm_head = original_model.lm_head
        self.vocab_size = original_model.config.vocab_size
        self.to(device)

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)

        past_key_values_length = 0
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_len),
            hidden_states,
            past_key_values_length
        )

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=causal_attention_mask,
                position_ids=position_ids
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return {'logits': logits}

def role(question):    # System prompt and four in-context examples for ARC-Easy with llama-3.1 8b
    chat = [
        {
            "role": "system",
            "content": (
                "You are a Science expert assistant. "
                "Your task is to answer multiple-choice science questions at grade-school level. "
                "Each question has four answer choices, labeled A, B, C, and D. "
                "For each question:\n"
                "- Carefully read the question and all answer choices.\n"
                "- Select the single best answer from the options (A, B, C, or D).\n"
                "- Respond only with the letter of the correct answer, and nothing else—no explanation or extra words.\n"
                "Be precise and consistent: Only the answer letter."
            ),
        },
        {
            "role": "user",
            "content": question
        }
    ]
    return chat

question_list = []
choices_by_question = []
answer_keys = []

for q, group in grouped:
    choices = [(row['choice_label'], row['choice_text']) for _, row in group.iterrows()]
    question_list.append(q)
    choices_by_question.append(choices)
    answer_keys.append(group['answer_key'].iloc[0])

NUM_QUESTIONS = len(question_list)


def format_question_with_choices(question, choices):
    """Format question with multiple choice options"""
    formatted_question = question + "\n"
    for label, text in choices:
        formatted_question += f"({label}) {text}\n"
    return formatted_question.strip()

def empirical_entropy_bits(labels):
    """H(Y) in bits."""
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * (np.log2(p + 1e-12))))




def conditional_entropy_bits_from_probe(X, y, random_state=42):
    """
    Train a calibrated logistic probe on (X, y), return H(Y|X) ≈ E[-log2 p(y|x)].
    Handles common LLM output variations like mixing letters and numbers.
    
    Args:
        expected_classes: Optional list of expected class labels (e.g., ['A', 'B', 'C', 'D'])
                         If provided, will attempt to map unexpected outputs
    """
    y_array = np.asarray(y).astype(str)  # Ensure string type for consistency
    
    expected_classes = ['A' , 'B' , 'C' , 'D']
    if expected_classes is not None:
        y_cleaned = clean_llm_outputs(y_array, expected_classes)
    else:
        y_cleaned = y_array
    
    unique_labels = np.unique(y_cleaned)
    
    if len(unique_labels) == 1:
        return 0.0
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_cleaned)
    
    if len(unique_labels) == 2:
        base = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        base = LogisticRegression(max_iter=1000, random_state=random_state, multi_class='ovr')
    
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=min(3, len(y_encoded)))
    pipe = make_pipeline(StandardScaler(with_mean=False), clf)
    
    try:
        pipe.fit(X, y_encoded)
        proba = pipe.predict_proba(X)  # (N, n_classes)
        eps = 1e-12
        
        pyx = proba[np.arange(len(y_encoded)), y_encoded]
        
        Hcond = float(np.mean(-np.log2(pyx + eps)))
        return Hcond
        
    except Exception as e:
        print(f"Warning: Probe training failed: {e}")
        return empirical_entropy_bits(y_cleaned)


def clean_llm_outputs(y_array, expected_classes):
    """
    Clean and normalize LLM outputs to expected class labels.
    Maps common variations like numbers to letters.
    """
    letter_to_number = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}
    number_to_letter = {v: k for k, v in letter_to_number.items()}
    
    cleaned = []
    for label in y_array:
        label_clean = str(label).strip().upper()
        
        if label_clean in expected_classes:
            cleaned.append(label_clean)
        elif label_clean in number_to_letter and number_to_letter[label_clean] in expected_classes:
            cleaned.append(number_to_letter[label_clean])
            print(f"Mapped '{label}' -> '{number_to_letter[label_clean]}'")
        elif label_clean in letter_to_number and letter_to_number[label_clean] in expected_classes:
            cleaned.append(letter_to_number[label_clean])
            print(f"Mapped '{label}' -> '{letter_to_number[label_clean]}'")
        else:
            cleaned.append(label_clean)
            print(f"Warning: Unexpected class '{label}', keeping as-is")
    
    return np.array(cleaned)

    
        
        
    
        
        

    
def get_layer_reps_simple(model, tokenizer, prompts, passages ,  targets, device, take='last', max_samples=None):
    """
    Extract representations from each layer using a simpler approach.
    """
    model.eval()
    
    if max_samples:
        prompts = prompts[:max_samples]
        targets = targets[:max_samples]
        passages = passages[:max_samples]
    
    all_layer_reps = [[] for _ in range(len(model.layers))]
    
    print(f"Extracting representations from {len(prompts)} samples...")
    
    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts, desc="Extracting representations")):
            try:
                formatted_question = format_question_with_choices(prompt , passages[i])
                chat = role(formatted_question)
                input_text = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                batch_size, seq_len = input_ids.shape
                hidden_states = model.embed_tokens(input_ids)
                
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                past_key_values_length = 0
                causal_attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_len),
                    hidden_states,
                    past_key_values_length
                )
                
                current_hidden_states = hidden_states
                
                for layer_idx, layer in enumerate(model.layers):
                    layer_outputs = layer(
                        hidden_states=current_hidden_states,
                        attention_mask=causal_attention_mask,
                        position_ids=position_ids
                    )
                    current_hidden_states = layer_outputs[0]
                    
                    if take == 'last':
                        last_token_idx_tensor = attention_mask.sum(dim=1) - 1  # shape (batch_size,)
                        last_token_idx = int(last_token_idx_tensor[0].item())  # scalar index for batch element 0
                        rep_tensor = current_hidden_states[0, last_token_idx, :]  # shape (hidden_dim,)
                        rep = rep_tensor.detach().cpu().numpy().reshape(-1)  # 1-D vector
                    elif take == 'mean':
                        mask_expanded = attention_mask.unsqueeze(-1).expand_as(current_hidden_states)
                        masked_hidden = current_hidden_states * mask_expanded
                        denom = attention_mask.sum(dim=1, keepdim=True)  # shape (batch_size, 1)
                        pooled = masked_hidden.sum(dim=1) / denom  # shape (batch_size, hidden_dim)
                        rep = pooled[0].detach().cpu().numpy().reshape(-1)  # 1-D vector
                    else:
                        raise ValueError("take must be 'last' or 'mean'")
                    
                    all_layer_reps[layer_idx].append(rep)
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    layer_reps = []
    for layer_reps_list in all_layer_reps:
        if len(layer_reps_list) > 0:
            layer_reps.append(np.array(layer_reps_list))
        else:
            print(f"Warning: No representations collected for layer {len(layer_reps)}")
            break
    
    return layer_reps


def compute_layer_MI(model, tokenizer, prompts, passages, labels, device, max_samples=50):
    """
    Returns dict: layer_index -> dict(HY, HYX, I)
    """
    print("Computing mutual information for each layer...")
    
    if len(prompts) > max_samples:
        print(f"Limiting to {max_samples} samples for MI computation")
    
    reps = get_layer_reps_simple(model, tokenizer, prompts, passages , labels, device, take='last', max_samples=max_samples)
    actual_samples = len(reps[0]) if len(reps) > 0 else 0
    HY = empirical_entropy_bits(labels[:actual_samples])
    
    print(f"H(Y) = {HY:.4f} bits")
    print(f"Successfully extracted representations from {len(reps)} layers")
    
    out = {}
    for L, X in enumerate(reps, start=1):
        print(f"Processing layer {L}/{len(reps)}...")
        HYX = conditional_entropy_bits_from_probe(X, labels[:len(X)])
        mutual_info = max(0.0, HY - HYX)
        out[L] = {
            "H(Y)": HY, 
            "H(Y|X^l)": HYX, 
            "I(X^l;Y)": mutual_info
        }
        print(f"Layer {L}: H(Y|X) = {HYX:.4f}, I(X;Y) = {mutual_info:.4f}")
    
    return out


def plot_information_flow(mi_results, save_path=None):
    """Plot the information flow across layers"""
    if not mi_results:
        print("No MI results to plot")
        return None

    layers = list(mi_results.keys())
    mutual_info = [mi_results[l]["I(X^l;Y)"] for l in layers]
    conditional_entropy = [mi_results[l]["H(Y|X^l)"] for l in layers]

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    print("mutual_info:", mutual_info)
    ax1.plot(layers, mutual_info, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('I(X^l; Y) (bits)')
    ax1.set_title('ARC-EASY - FULL Model')
    ax1.set_ylim(0.0 , 2)
    ax1.grid(True, alpha=0.3)



    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    if mutual_info:
        max_mi_layer = layers[np.argmax(mutual_info)]
        min_ce_layer = layers[np.argmin(conditional_entropy)]

        print(f"\nInformation Flow Summary:")
        print(f"Maximum MI at layer {max_mi_layer}: {max(mutual_info):.4f} bits")
        print(f"Minimum conditional entropy at layer {min_ce_layer}: {min(conditional_entropy):.4f} bits")

    return fig


def evaluate_model(model_instance, model_name="Model"):
    """Evaluate model performance on ARC dataset"""
    correct = 0
    total = 0
    predictions = []

    print(f"\nEvaluating {model_name}...")

    for i in tqdm(range(100), desc=f"Processing {model_name}"):
        question = question_list[i]
        choices = choices_by_question[i]
        answer_key = answer_keys[i]

        formatted_question = format_question_with_choices(question, choices)

        chat = role(formatted_question)

        input_text = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)

        with torch.no_grad():

            def generate(input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
                with torch.no_grad():
                    current_ids = input_ids
                    generated_tokens = []

                    for _ in range(max_new_tokens):
                        outputs = model_instance(current_ids, attention_mask=attention_mask)
                        logits = outputs['logits']

                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        generated_tokens.append(next_token)

                        current_ids = torch.cat([current_ids, next_token], dim=1)

                        if attention_mask is not None:
                            attention_mask = torch.cat([
                                attention_mask,
                                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                            ], dim=1)

                    gen_tokens = current_ids[:, input_ids.shape[1]:]

                    if gen_tokens.numel() > 0:
                        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                        gen_text = gen_text.strip()  # Remove any leading/trailing whitespace
                    else:
                        gen_text = ""

                    return gen_text

            predicted_text = generate(input_ids, max_new_tokens=1)

            if len(predicted_text) > 0:
                predicted_answer = predicted_text.upper()
            else:
                predicted_answer = "INVALID"

            predictions.append(predicted_answer)

            if predicted_answer == answer_key:
                correct += 1

            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"{model_name} Accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy, predictions








def analyze_layer_importance_with_MI():
    """Complete analysis including mutual information computation"""
    print("\n" + "="*80)
    print("LAYER IMPORTANCE ANALYSIS WITH MUTUAL INFORMATION")
    print("="*80)
    
    delete_indices = set([19])
    full_model = ModifiedLlamaModel(model_base, delete_indices=delete_indices)
    full_model.eval()
    
    max_samples_mi = min(100, len(question_list)) 
    
    try:
        mi_results = compute_layer_MI(
            full_model, 
            tokenizer, 
            question_list, 
            choices_by_question,
            answer_keys, 
            device, 
            max_samples=max_samples_mi
        )
        
        plot_information_flow(mi_results, save_path=os.environ.get("TALE_PLOT_PATH", "experiments/mutual_information/plots/arc_easy_mi.png"))
        
        if mi_results:
            mi_values = [(layer, results["I(X^l;Y)"]) for layer, results in mi_results.items()]
            mi_values.sort(key=lambda x: x[1], reverse=True)
            
            print("Layers ranked by mutual information:")
            for layer, mi_val in mi_values[:10]:
                print(f"Layer {layer}: {mi_val:.4f} bits")
        
    except Exception as e:
        print(f"Error in MI computation: {e}")
        import traceback
        traceback.print_exc()
        mi_results = {}
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT LAYER COUNTS")
    print("="*60)
    
    
    del full_model
    torch.cuda.empty_cache()
    
    return mi_results

if __name__ == "__main__":
    print(f"Loaded {NUM_QUESTIONS} questions from Boolean Expressions dataset")
    print(f"Model has {len(model_base.model.layers)} layers")

    try:
        mi_results = analyze_layer_importance_with_MI()

        print(f"\n{'='*80}")
        print("COMPLETE ANALYSIS FINISHED")
        print(f"{'='*80}")

        if mi_results:
            print("\nMutual Information Summary (First 5 layers):")
            for layer, results in list(mi_results.items())[:5]:
                print(f"Layer {layer}: I(X;Y) = {results['I(X^l;Y)']:.4f} bits")


    except Exception as e:
        print(f"Error in complete analysis: {e}")
        import traceback
        traceback.print_exc()

        print("Running basic layer analysis only...")


