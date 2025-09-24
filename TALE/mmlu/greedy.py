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

import time   # <-- ADDED: For timing
import random
def set_seed(seed=42):
    """Set seed for reproducibility across all random number generators"""
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ Seed set to {seed} for reproducible results")

# Call this at the very beginning of your script
set_seed(42)  # You and your friend should use the same seed number

# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


df = pd.read_csv("/tmpdir/m24047nmmr/pruning/datasets/mmlu/mmlu_all_validation.csv")
print(f"mmlu dataset length: {len(df)}")


tokenizer = AutoTokenizer.from_pretrained("/work/m24047/m24047nmmr/llama-3.1-8b-instruct")
model_base = LlamaForCausalLM.from_pretrained("/work/m24047/m24047nmmr/llama-3.1-8b-instruct/Llama-8B-mmlu64")
#model_base = LlamaForCausalLM.from_pretrained("/work/m24047/m24047nmmr/llama-3.1-8b-instruct")

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

    def expand_attention_mask(self, attention_mask, dtype, tgt_len=None):
        batch_size, src_len = attention_mask.shape
        if tgt_len is None:
            tgt_len = src_len
        expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len)
        return expanded_mask.to(dtype)

    def forward(self, input_ids, attention_mask=None, position_ids=None, position_embeddings=None, **kwargs):
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_len), device=device)

        past_key_values_length = 0  # No caching in your setup
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,  # [bs, seq_len] padding mask
            (batch_size, seq_len),  # input_shape
            hidden_states,  # For dtype
            past_key_values_length
        )

        for i, layer in enumerate(self.layers):
            layer_kwargs = {
                'hidden_states': hidden_states,
                'attention_mask': causal_attention_mask,
            }
            if position_embeddings is not None:
                layer_kwargs['position_embeddings'] = position_embeddings
            else:
                layer_kwargs['position_ids'] = position_ids
            layer_outputs = layer(**layer_kwargs)
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return {'logits': logits}



def role(question):
    """0-shot system prompt for MMLU - academic and professional knowledge"""
    chat = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers multiple-choice questions across various academic subjects including humanities, social sciences, STEM, and professional fields. "
                "Read each question carefully and select the best answer from the given options. "
                "Respond with only the letter of your chosen answer (A, B, C, or D)."
            ),
        },
        # ---- Actual test question ----
        {
            "role": "user",
            "content": question
        }
    ]
    return chat


# Process MMLU data
question_list = []
choices_by_question = []
answer_keys = []
subjects = []

for _, row in df.iterrows():
    question = row['question']
    choices = [
        ('A', row['choice_A']),
        ('B', row['choice_B']),
        ('C', row['choice_C']),
        ('D', row['choice_D'])
    ]
    answer_key = row['answer']  # Already in A, B, C, D format
    subject = row['subject']
    
    question_list.append(question)
    choices_by_question.append(choices)
    answer_keys.append(answer_key)
    subjects.append(subject)

NUM_QUESTIONS = len(question_list)

def format_question_with_choices(question, choices):
    """Format question with multiple choice options"""
    formatted_question = question + "\n"
    for label, text in choices:
        formatted_question += f"({label}) {text}\n"
    return formatted_question.strip()

def evaluate_model(model_instance, num_layers=None, model_name="Original"):
    """Evaluate model performance on MMLU dataset"""
    correct = 0
    total = 0
    predictions = []
    subject_results = {}
    
    print(f"\nEvaluating {model_name} model...")
    if num_layers:
        print(f"Using {num_layers} layers out of {len(model_base.model.layers)} total layers")
    
    # Determine number of questions to process
    num_to_process = NUM_QUESTIONS
    
    for i in tqdm(range(num_to_process), desc=f"Processing {model_name}"):
        question = question_list[i]
        choices = choices_by_question[i] 
        answer_key = answer_keys[i]
        subject = subjects[i]
        
        # Track subject-wise performance
        if subject not in subject_results:
            subject_results[subject] = {'correct': 0, 'total': 0}
        
        # Format the question with choices
        formatted_question = format_question_with_choices(question, choices)
        
        # Create chat template
        chat = role(formatted_question)
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input and move to GPU
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
    
        with torch.no_grad():
            def generate(input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
                with torch.no_grad():
                    current_ids = input_ids
                    generated_tokens = []

                    for _ in range(max_new_tokens):
                        # Forward pass
                        outputs = model_instance(current_ids, attention_mask=attention_mask)
                        logits = outputs['logits']

                        # Get next token (greedy decoding)
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        generated_tokens.append(next_token)

                        # Update current_ids for next iteration
                        current_ids = torch.cat([current_ids, next_token], dim=1)

                        # Update attention mask if provided
                        if attention_mask is not None:
                            attention_mask = torch.cat([
                                attention_mask, 
                                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                            ], dim=1)
                    
                    # Extract only the newly generated tokens
                    gen_tokens = current_ids[:, input_ids.shape[1]:]
                    
                    # Decode the generated tokens to text
                    if gen_tokens.numel() > 0:
                        # Take the first batch item and decode
                        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                        gen_text = gen_text.strip()  # Remove any leading/trailing whitespace
                    else:
                        gen_text = ""

                    return gen_text

            predicted_text = generate(input_ids, max_new_tokens=1)
            
            # Process the predicted text
            if len(predicted_text) > 0:
                predicted_answer = predicted_text.upper()
            else:
                predicted_answer = "INVALID"
            
            predictions.append(predicted_answer)
            
            # Check if prediction is correct
            if predicted_answer == answer_key:
                correct += 1
                subject_results[subject]['correct'] += 1
            
            total += 1
            subject_results[subject]['total'] += 1
            
    accuracy = correct / total if total > 0 else 0
    print(f"{model_name} Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Print subject-wise results (top 10 subjects by count)
    print(f"\nTop 10 Subject-wise results for {model_name}:")
    sorted_subjects = sorted(subject_results.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
    for subject, results in sorted_subjects:
        subj_acc = results['correct'] / results['total'] if results['total'] > 0 else 0
        print(f"  {subject:25}: {subj_acc:.3f} ({results['correct']}/{results['total']})")
    
    return accuracy, predictions



def get_baseline_accuracy(return_model=False):
    print("\n" + "="*60)
    print("Getting baseline accuracy (full model, no layers dropped)...")
    baseline_model = ModifiedLlamaModel(model_base, delete_indices=set())
    baseline_model.eval()
    baseline_accuracy, _ = evaluate_model(baseline_model, "Baseline-Full-Model")
    if return_model:
        return baseline_accuracy, baseline_model
    # Clean up memory
    del baseline_model
    torch.cuda.empty_cache()
    return baseline_accuracy


def greedy_layer_dropping():
    """
    Greedy algorithm to find the best layers to drop while maintaining performance above threshold.
    """
    original_layer_count = len(model_base.model.layers)
    # Get baseline accuracy (full model, no layers dropped)
    # ----- TIMING CHANGE: baseline ------------
    baseline_start = time.time()
    baseline_accuracy, baseline_model = get_baseline_accuracy(return_model=True)
    baseline_end = time.time()
    baseline_duration = baseline_end - baseline_start
    print(f"Baseline evaluation time: {baseline_duration:.2f} seconds")
    # ------------------------------------------
    threshold_accuracy = baseline_accuracy - 0.08
    print(f"\nOriginal model has {original_layer_count} layers")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Threshold accuracy: {threshold_accuracy:.4f} (baseline - 8%)")

    permanently_dropped_layers = set()
    available_layers = set(range(original_layer_count))
    iteration = 1
    all_results = {}

    print("\n" + "="*80)
    print("STARTING GREEDY LAYER DROPPING ALGORITHM")
    print("="*80)

    while True:
        print(f"\n{'='*20} ITERATION {iteration} {'='*20}")
        print(f"Currently dropped layers: {sorted(permanently_dropped_layers) if permanently_dropped_layers else 'None'}")
        print(f"Available layers to test: {sorted(available_layers - permanently_dropped_layers)}")
        testable_layers = available_layers - permanently_dropped_layers
        if not testable_layers:
            print("No more layers available to test. Stopping.")
            break
        iteration_results = {}
        for layer_to_test in testable_layers:
            print(f"\n{'-'*40}")
            print(f"Testing drop of layer {layer_to_test}")
            current_drop_set = permanently_dropped_layers.union({layer_to_test})
            test_model = ModifiedLlamaModel(model_base, delete_indices=current_drop_set)
            test_model.eval()
            model_name = f"Iter{iteration}-Drop-{layer_to_test}"
            time_begin = time.time()
            accuracy, _ = evaluate_model(test_model, model_name)
            time_end = time.time()
            time_duration = time_end - time_begin
            iteration_results[layer_to_test] = accuracy
            print(f"time taken for {layer_to_test} layer config : {time_duration}")
            print(f"Accuracy with layer {layer_to_test} dropped: {accuracy:.4f}")
            print(f"Current drop set: {sorted(current_drop_set)}")
            print(f"Layers kept: {len(model_base.model.layers) - len(current_drop_set)}")
            del test_model
            torch.cuda.empty_cache()
        best_layer = max(iteration_results.items(), key=lambda x: x[1])
        best_layer_idx, best_accuracy = best_layer
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration} RESULTS:")
        print(f"{'='*50}")
        sorted_results = sorted(iteration_results.items(), key=lambda x: x[1], reverse=True)
        for layer_idx, acc in sorted_results:
            print(f"Drop layer {layer_idx:2d}: {acc:.4f}")
        print(f"\nBest layer to drop: {best_layer_idx} (accuracy: {best_accuracy:.4f})")
        print(f"Threshold: {threshold_accuracy:.4f}")
        if best_accuracy >= threshold_accuracy:
            print(f"✅ Accuracy {best_accuracy:.4f} is above threshold {threshold_accuracy:.4f}")
            print(f"Adding layer {best_layer_idx} to permanent drop list")
            permanently_dropped_layers.add(best_layer_idx)
            all_results[f"Iteration_{iteration}"] = {
                'best_layer': best_layer_idx,
                'best_accuracy': best_accuracy,
                'all_results': iteration_results,
                'permanently_dropped': permanently_dropped_layers.copy()
            }
            iteration += 1
        else:
            print(f"❌ Accuracy {best_accuracy:.4f} is below threshold {threshold_accuracy:.4f}")
            print("Stopping the algorithm - threshold reached")
            break

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Baseline accuracy (full model): {baseline_accuracy:.4f}")
    print(f"Threshold accuracy: {threshold_accuracy:.4f}")
    print(f"Final dropped layers: {sorted(permanently_dropped_layers) if permanently_dropped_layers else 'None'}")
    print(f"Total layers dropped: {len(permanently_dropped_layers)}")
    print(f"Layers remaining: {original_layer_count - len(permanently_dropped_layers)}")
    
    # -------- Final timing and speedup ----------
    if permanently_dropped_layers:
        print(f"\n{'-'*40}")
        print("Testing final optimized model...")
        final_model = ModifiedLlamaModel(model_base, delete_indices=permanently_dropped_layers)
        final_model.eval()
        # TIME the evaluation here
        final_start = time.time()
        final_accuracy, _ = evaluate_model(final_model, "Final-Optimized-Model")
        final_end = time.time()
        final_duration = final_end - final_start
        print(f"Final optimized model evaluation time: {final_duration:.2f} seconds")
        if final_duration > 0:
            speedup = baseline_duration / final_duration
        else:
            speedup = float('inf')
        print(f"Speed increase compared to baseline: {speedup:.2f}x")
        print(f"\nFinal optimized model accuracy: {final_accuracy:.4f}")
        print(f"Accuracy drop from baseline: {baseline_accuracy - final_accuracy:.4f} ({((baseline_accuracy - final_accuracy) / baseline_accuracy * 100):.1f}%)")
        print(f"Model compression: {len(permanently_dropped_layers)}/{original_layer_count} layers removed ({(len(permanently_dropped_layers)/original_layer_count*100):.1f}%)")
        del final_model
        torch.cuda.empty_cache()
    else:
        print("No layers could be dropped while maintaining the threshold accuracy.")

    if all_results:
        print(f"\n{'='*60}")
        print("ITERATION BY ITERATION SUMMARY")
        print(f"{'='*60}")
        for iter_name, iter_data in all_results.items():
            print(f"{iter_name}: Best layer {iter_data['best_layer']} (acc: {iter_data['best_accuracy']:.4f})")
            print(f" Dropped so far: {sorted(iter_data['permanently_dropped'])}")

    # Clean up baseline_model resident memory
    del baseline_model
    torch.cuda.empty_cache()

    return permanently_dropped_layers, all_results, baseline_accuracy

if __name__ == "__main__":
    print(f"Loaded {NUM_QUESTIONS} questions from ARC-Easy validation set")
    print(f"Model has {len(model_base.model.layers)} layers")
    final_dropped_layers, iteration_results, baseline_acc = greedy_layer_dropping()
    #get_baseline_accuracy()
    print(f"\n{'='*80}")
    print("ALGORITHM COMPLETED Dlt")
    print(f"{'='*80}")
    print(f"Layers to drop for optimal performance: {sorted(final_dropped_layers) if final_dropped_layers else 'None'}")
