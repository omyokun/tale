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
#import seaborn as sns
from collections import defaultdict

# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -- Load Data --
# -- Load BoolQ Data --
df = pd.read_csv("/tmpdir/m24047nmmr/pruning/datasets/boolq/boolq_validation.csv")
print(f"BoolQ dataset length: {len(df)}")
# 
tokenizer = AutoTokenizer.from_pretrained("/work/m24047/m24047nmmr/llama-3.1-8b-instruct")
model_base = LlamaForCausalLM.from_pretrained("/work/m24047/m24047nmmr/llama-3.1-8b-instruct")

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU
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

def create_boolq_role(question, passage):
    chat = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers True/False questions based on given passages. "
                "Read the passage carefully and determine if the question can be answered as True or False based on the information in the passage. "
                "Respond with only 'A' for True or 'B' for False."
            ),
        },
        {
            "role": "user",
            "content": format_boolq_question(question, passage)
        }
    ]
    return chat

def format_boolq_question(question, passage):
    formatted_question = f"Passage: {passage}\n\nQuestion: {question}\n"
    formatted_question += "(A) True\n(B) False"
    return formatted_question

# Prepare BoolQ data in the same format as your ARC data
question_list = []
passage_list = []
answer_keys = []

for _, row in df.iterrows():
    question_list.append(row['question'])
    passage_list.append(row['passage'])
    answer_keys.append(row['answer_label'])  # A or B

NUM_QUESTIONS = len(question_list)

def empirical_entropy_bits(labels):
    """H(Y) in bits."""
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * (np.log2(p + 1e-12))))

# def conditional_entropy_bits_from_probe(X, y):
#     """
#     Train a calibrated logistic probe on (X, y), return H(Y|X) ≈ E[-log2 p(y|x)].
#     """
#     y_bin = (np.asarray(y).astype(str) == "True").astype(int)

#     # Handle edge case where all labels are the same
#     if len(np.unique(y_bin)) == 1:
#         return 0.0

#     # Standardize then logistic; calibrate with sigmoid for probabilities
#     base = LogisticRegression(max_iter=1000, random_state=42)
#     clf = CalibratedClassifierCV(base, method="sigmoid", cv=min(3, len(y_bin)))
#     pipe = make_pipeline(StandardScaler(with_mean=False), clf)

#     try:
#         pipe.fit(X, y_bin)
#         proba = pipe.predict_proba(X)  # (N, 2)
#         eps = 1e-12
#         # pick p(y_i|x_i) for the true class
#         pyx = proba[np.arange(len(y_bin)), y_bin]
#         # bits
#         Hcond = float(np.mean(-np.log2(pyx + eps)))
#         return Hcond
#     except Exception as e:
#         print(f"Warning: Probe training failed: {e}")
#         return empirical_entropy_bits(y)  # Fallback to marginal entropy

def conditional_entropy_bits_from_probe(X, y, random_state=42):
    """
    Train a calibrated logistic probe on (X, y), return H(Y|X) ≈ E[-log2 p(y|x)].
    Works with any discrete label set (binary or multiclass).
    """
    y_array = np.asarray(y)
    unique_labels = np.unique(y_array)
    n_classes = len(unique_labels)
    print(n_classes)
    
    # Handle edge case where all labels are the same
    if n_classes == 1:
        return 0.0
    
    # For binary classification, use the existing approach
    if n_classes == 2:
        # Convert to 0/1, doesn't matter which label gets which
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_encoded = np.array([label_to_int[label] for label in y_array])
        
        base = LogisticRegression(max_iter=1000, random_state=random_state)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=min(3, len(y_encoded)))
        pipe = make_pipeline(StandardScaler(with_mean=False), clf)
        
        try:
            pipe.fit(X, y_encoded)
            proba = pipe.predict_proba(X)  # (N, 2)
            eps = 1e-12
            pyx = proba[np.arange(len(y_encoded)), y_encoded]
            Hcond = float(np.mean(-np.log2(pyx + eps)))
            return Hcond
        except Exception as e:
            print(f"Warning: Binary probe training failed: {e}")
            return empirical_entropy_bits(y)
    
    # For multiclass, use one-vs-rest approach
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_array)
        
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
            print(f"Warning: Multiclass probe training failed: {e}")
            return empirical_entropy_bits

    
def get_layer_reps_simple(model, tokenizer, prompts, passages ,  targets, device, take='last', max_samples=None):
    """
    Extract representations from each layer using a simpler approach.
    """
    model.eval()
    
    if max_samples:
        prompts = prompts[:max_samples]
        targets = targets[:max_samples]
        passages = passages[:max_samples]
    
    # Store representations from each layer
    all_layer_reps = [[] for _ in range(len(model.layers))]
    
    print(f"Extracting representations from {len(prompts)} samples...")
    
    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts, desc="Extracting representations")):
            try:
                # Format the question
                formatted_question = format_boolq_question(prompt , passages[i])
                chat = create_boolq_role(prompt , passages[i])
                input_text = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
                
                # Tokenize
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                # Forward pass through embedding
                batch_size, seq_len = input_ids.shape
                hidden_states = model.embed_tokens(input_ids)
                
                # Prepare inputs for layers
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                past_key_values_length = 0
                causal_attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_len),
                    hidden_states,
                    past_key_values_length
                )
                
                # Forward through each layer and collect representations
                current_hidden_states = hidden_states
                
                for layer_idx, layer in enumerate(model.layers):
                    layer_outputs = layer(
                        hidden_states=current_hidden_states,
                        attention_mask=causal_attention_mask,
                        position_ids=position_ids
                    )
                    current_hidden_states = layer_outputs[0]
                    
                    # Extract representation (last token or mean)
                    if take == 'last':
                        # Take the last non-padding token
                        # ensure last_token_idx is an integer (scalar)
                        last_token_idx_tensor = attention_mask.sum(dim=1) - 1  # shape (batch_size,)
                        last_token_idx = int(last_token_idx_tensor[0].item())  # scalar index for batch element 0
                        rep_tensor = current_hidden_states[0, last_token_idx, :]  # shape (hidden_dim,)
                        rep = rep_tensor.detach().cpu().numpy().reshape(-1)  # 1-D vector
                    elif take == 'mean':
                        # Mean pooling over non-padding tokens
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
    
    # Convert to numpy arrays
    layer_reps = []
    for layer_reps_list in all_layer_reps:
        if len(layer_reps_list) > 0:
            layer_reps.append(np.array(layer_reps_list))
        else:
            # If no representations collected for this layer, skip it
            print(f"Warning: No representations collected for layer {len(layer_reps)}")
            break
    
    return layer_reps


def compute_layer_MI(model, tokenizer, prompts, passages, labels, device, max_samples=50):
    """
    Returns dict: layer_index -> dict(HY, HYX, I)
    """
    print("Computing mutual information for each layer...")
    
    # Limit samples for efficiency
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

    # Plot mutual information
    ax1.plot(layers, mutual_info, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('I(X^l; Y) (bits)')
    ax1.set_title('BOOLQ - Drop Model')
    ax1.set_ylim(0.0 , 0.58)
    ax1.grid(True, alpha=0.3)


    # # Plot conditional entropy
    # ax2.plot(layers, conditional_entropy, 'r-s', linewidth=2, markersize=6)
    # ax2.set_xlabel('Layer')
    # ax2.set_ylabel('H(Y | X^l) (bits)')
    # ax2.set_title('Conditional Entropy Across Layers')
    # ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    if mutual_info:
        max_mi_layer = layers[np.argmax(mutual_info)]
        min_ce_layer = layers[np.argmin(conditional_entropy)]

        print(f"\nInformation Flow Summary:")
        print(f"Maximum MI at layer {max_mi_layer}: {max(mutual_info):.4f} bits")
        print(f"Minimum conditional entropy at layer {min_ce_layer}: {min(conditional_entropy):.4f} bits")

    return fig

def evaluate_model(model_instance, model_name="Original"):
    correct = 0
    total = 0
    predictions = []
    print(f"\nEvaluating {model_name} model on BoolQ...")

    num_to_process = min(1000, NUM_QUESTIONS)
    for i in tqdm(range(num_to_process), desc=f"Processing {model_name}"):
        question = question_list[i]
        passage = passage_list[i]
        answer_key = answer_keys[i]
        chat = create_boolq_role(question, passage)
        input_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
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
                        gen_text = gen_text.strip()
                    else:
                        gen_text = ""
                    return gen_text
            predicted_text = generate(input_ids, max_new_tokens=1)
            if len(predicted_text) > 0:
                predicted_answer = predicted_text.upper()
            else:
                predicted_answer = "INVALID"
            if predicted_answer not in ['A', 'B']:
                predicted_answer = "INVALID"
            predictions.append(predicted_answer)
            if predicted_answer == answer_key:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"{model_name} BoolQ Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy, predictions



# def test_different_layer_counts():
#     """Test model performance with different numbers of layers"""
#     original_layer_count = len(model_base.model.layers)
#     print(f"Total layer count of llama: {original_layer_count}")

#     # Test a few different layer counts
#     layer_counts_to_test = [original_layer_count, original_layer_count - 5, original_layer_count // 2]
#     layer_counts_to_test = [l for l in layer_counts_to_test if l > 0 and l <= original_layer_count]

#     results = {}

#     for num_layers in layer_counts_to_test:
#         print(f"\nCreating modified model with {num_layers} layers...")
#         modified_model = ModifiedLlamaModel(model_base, num_layers=num_layers)
#         modified_model.eval()
#         accuracy, _ = evaluate_model(
#             modified_model,
#             num_layers=num_layers,
#             model_name=f"Modified-{num_layers}L"
#         )
#         results[f"{num_layers} Layers"] = accuracy
#         print(f"{num_layers} layers accuracy: {accuracy}")

#         # Clean up
#         del modified_model
#         torch.cuda.empty_cache()

#     return results

def analyze_layer_importance_with_MI():
    """Complete analysis including mutual information computation"""
    print("\n" + "="*80)
    print("LAYER IMPORTANCE ANALYSIS WITH MUTUAL INFORMATION")
    print("="*80)
    
    # Create a model with all layers for MI analysis
    delete_indices = set([])
    full_model = ModifiedLlamaModel(model_base, delete_indices=delete_indices)
    full_model.eval()
    
    # Use a subset of data for MI computation (computational efficiency)
    max_samples_mi = min(100, len(question_list)) 
    
    try:
        # Compute mutual information
        mi_results = compute_layer_MI(
            full_model, 
            tokenizer, 
            question_list, 
            passage_list,
            answer_keys, 
            device, 
            max_samples=max_samples_mi
        )
        
        # Plot results
        plot_information_flow(mi_results, save_path='/tmpdir/m24047nmmr/pruning/MI/plots/boolq-mi-plot.png')
        if mi_results:
    # Extract MI values in layer order
            mi_list = []
            for layer in sorted(mi_results.keys()):
                mi_value = mi_results[layer]["I(X^l;Y)"]
                mi_list.append(round(mi_value, 4))  # Round to 4 decimal places
    
            print(f"\nMutual Information List (all layers):")
            print(mi_list)        
        # Get layers with highest MI for analysis
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
    
    # Test performance with different layer counts
    print("\n" + "="*60)
    print("TESTING DIFFERENT LAYER COUNTS")
    print("="*60)
    
    # test_results = test_different_layer_counts()
    
    # Clean up
    del full_model
    torch.cuda.empty_cache()
    
    return mi_results

# Main execution
if __name__ == "__main__":
    print(f"Loaded {NUM_QUESTIONS} questions from Boolean Expressions dataset")
    print(f"Model has {len(model_base.model.layers)} layers")

    # Run complete analysis
    try:
        mi_results = analyze_layer_importance_with_MI()

        print(f"\n{'='*80}")
        print("COMPLETE ANALYSIS FINISHED")
        print(f"{'='*80}")

        # Print final summary
        if mi_results:
            print("\nMutual Information Summary (First 5 layers):")
            for layer, results in list(mi_results.items())[:5]:
                print(f"Layer {layer}: I(X;Y) = {results['I(X^l;Y)']:.4f} bits")

        # print("\nAccuracy Results:")
        # for config, acc in test_results.items():
        #     print(f"{config}: {acc:.4f}")

    except Exception as e:
        print(f"Error in complete analysis: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to basic analysis
        print("Running basic layer analysis only...")
        # test_results = test_different_layer_counts()
        # print("\nBasic Results:")
        # for config, acc in test_results.items():


