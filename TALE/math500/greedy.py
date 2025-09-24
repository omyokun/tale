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
import re
import time

# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -- Load GSM8K Data --
df = pd.read_csv("/tmpdir/m24047nmmr/pruning/datasets/math500/MATH-500.csv")
print(f"Dataset length: {len(df)}")

# tokenizer = AutoTokenizer.from_pretrained("/work/m24047/m24047nmmr/llama-3.1-8b-instruct")
# model_base = LlamaForCausalLM.from_pretrained("/work/m24047/m24047nmmr/llama-3.1-8b-instruct")
tokenizer = AutoTokenizer.from_pretrained("/work/m24047/m24047nmmr/mistral-7b-instruct-v03")
model_base = AutoModelForCausalLM.from_pretrained("/work/m24047/m24047nmmr/mistral-7b-instruct-v03")

# Move model to GPU
model_base = model_base.to(device)

# Use the same ModifiedLlamaModel class from your original code
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
        
        past_key_values_length = 0
        
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_len),
            hidden_states,
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

# def role(question):
#     """System prompt for math problem solving with full reasoning (strict) and in-context examples"""
#     system_prompt = (
#         "You are a careful math problem solver. Show complete step-by-step reasoning "
#         "and all calculations needed to arrive at the answer. Use clear, numbered or "
#         "labeled steps so the reasoning is easy to follow.\n\n"
#         "IMPORTANT (formatting): After the full reasoning, on a NEW LINE BY ITSELF write the final answer "
#         "in exactly this format:\n\n"
#         "#### <integer>\n\n"
#         "- <integer> must be digits only, optionally with a leading '-' for negative numbers (e.g. -7).\n"
#         "- Do NOT add words, punctuation, units, or commentary on the same line as the '####' line.\n"
#         "- The '####' line MUST be the final line of the output (nothing may follow it).\n\n"
#         "Assume the dataset's problems expect integer answers; ensure the final '####' line contains a single integer."
#     )

#     examples = [
#         {"role": "user", "content": "Problem: Solve for x: 7x + 5 = 3x + 29\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."},
#         {"role": "assistant", "content": "Concise solution: Subtract 3x from both sides to get 4x + 5 = 29. Subtract 5 from both sides to get 4x = 24. Divide by 4 to get x = 6.\n\n#### 6"},

#         {"role": "user", "content": "Problem: What is the greatest common divisor of 84 and 210?\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."},
#         {"role": "assistant", "content": "Concise solution: Prime factors: 84 = 2^2 * 3 * 7, 210 = 2 * 3 * 5 * 7. The common factors are 2 * 3 * 7 = 42.\n\n#### 42"},

#         {"role": "user", "content": "Problem: Find the area of a right triangle with legs of lengths 6 and 8.\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."},
#         {"role": "assistant", "content": "Concise solution: Area = (1/2) * leg1 * leg2 = 0.5 * 6 * 8 = 24.\n\n#### 24"},

#         {"role": "user", "content": "Problem: What is the sum of the first 20 positive integers?\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."},
#         {"role": "assistant", "content": "Concise solution: Sum = n(n+1)/2 with n = 20, so sum = 20*21/2 = 210.\n\n#### 210"},

#         {"role": "user", "content": "Problem: For the sequence a_n = 3n - 1, what is a_50?\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."},
#         {"role": "assistant", "content": "Concise solution: Substitute n = 50: a_50 = 3*50 - 1 = 150 - 1 = 149.\n\n#### 149"},

#         {"role": "user", "content": "Problem: Find the smallest positive integer n such that 2^n > 1000.\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."},
#         {"role": "assistant", "content": "Concise solution: 2^9 = 512, 2^10 = 1024 which is the first power exceeding 1000, so n = 10.\n\n#### 10"},

#         {"role": "user", "content": "Problem: How many integers between 1 and 100 inclusive are divisible by 3 or 5?\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."},
#         {"role": "assistant", "content": "Concise solution: Count multiples: floor(100/3)=33, floor(100/5)=20, subtract overlap floor(100/15)=6. Total = 33 + 20 - 6 = 47.\n\n#### 47"},

#         {"role": "user", "content": "Problem: If 5 fair coins are flipped, how many outcomes have exactly 2 heads?\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."},
#         {"role": "assistant", "content": "Concise solution: Number of ways = C(5,2) = 5!/(2!3!) = 10.\n\n#### 10"}
#     ]

#     chat = [
#         {"role": "system", "content": system_prompt},
#         *examples,
#         {"role": "user", "content": f"Problem: {question}\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."}
#     ]
#     return chat

def role(question):
    """System prompt for math problem solving with full reasoning (strict) - no examples"""
    system_prompt = (
        "You are a careful math problem solver. Show complete step-by-step reasoning "
        "and all calculations needed to arrive at the answer. Use clear, numbered or "
        "labeled steps so the reasoning is easy to follow.\n\n"
        "IMPORTANT (formatting): After the full reasoning, on a NEW LINE BY ITSELF write the final answer "
        "in exactly this format:\n\n"
        "#### <integer>\n\n"
        "- <integer> must be digits only, optionally with a leading '-' for negative numbers (e.g. -7).\n"
        "- Do NOT add words, punctuation, units, or commentary on the same line as the '####' line.\n"
        "- The '####' line MUST be the final line of the output (nothing may follow it).\n\n"
        "Assume the dataset's problems expect integer answers; ensure the final '####' line contains a single integer."
    )

    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Problem: {question}\n\nPlease solve step-by-step and finish with the final answer on a new line as: #### <integer>."}
    ]
    return chat


# Process GSM8K data
input_questions = []
targets = []

for _, row in df.iterrows():
    question = row['problem']
    target = row['answer']
    input_questions.append(question)
    targets.append(target)

NUM_QUESTIONS = len(input_questions)

def extract_final_answer(text):
    """Extract the final numerical answer after #### from generated text"""
    # Look for #### pattern
    if "####" in text:
        # Get everything after the last ####
        answer_part = text.split("####")[-1].strip()
        # Extract the first number found after ####
        numbers = re.findall(r'-?\d+\.?\d*', answer_part)
        if numbers:
            try:
                # Return the first number as string (convert to int then back to string to normalize)
                return str(int(float(numbers[0])))
            except:
                return numbers
    
    # Fallback: try to find the last number in the entire text
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    if numbers:
        try:
            return str(int(float(numbers[-1])))
        except:
            return numbers[-1]
    
    return ""

def evaluate_model(model_instance, num_layers=None, model_name="Original"):
    """Evaluate model performance on GSM8K dataset"""
    correct = 0
    total = 0
    predictions = []
    
    print(f"\nEvaluating {model_name} model...")
    if num_layers:
        print(f"Using {num_layers} layers out of {len(model_base.model.layers)} total layers")
    
    # Determine number of questions to process
    num_to_process = min(NUM_QUESTIONS, 100)
    
    for i in tqdm(range(num_to_process), desc=f"Processing {model_name}"):
        question = input_questions[i]
        target = targets[i]
        
        # Create chat template
        chat = role(question)
        
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
            def generate(input_ids, attention_mask=None, max_new_tokens=200, **kwargs):
                """Generate response with more tokens to allow full reasoning"""
                with torch.no_grad():
                    current_ids = input_ids
                    
                    for step in range(max_new_tokens):
                        # Forward pass
                        outputs = model_instance(current_ids, attention_mask=attention_mask)
                        logits = outputs['logits']
                        
                        # Get next token (greedy decoding)
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        
                        # Update current_ids for next iteration
                        current_ids = torch.cat([current_ids, next_token], dim=1)
                        
                        # Update attention mask if provided
                        if attention_mask is not None:
                            attention_mask = torch.cat([
                                attention_mask,
                                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                            ], dim=1)
                        
                        # Check if we should stop (optional: stop on EOS token)
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                    
                    # Extract only the newly generated tokens
                    gen_tokens = current_ids[:, input_ids.shape[1]:]
                    
                    # Decode the generated tokens to text
                    if gen_tokens.numel() > 0:
                        gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                        gen_text = gen_text.strip()
                    else:
                        gen_text = ""
                    
                    return gen_text
            
            predicted_text = generate(input_ids, max_new_tokens=200)
            
            # Extract final numerical answer after ####
            predicted_answer = extract_final_answer(predicted_text)
            predictions.append(predicted_answer)
            
            # Check if prediction is correct
            try:
                if str(predicted_answer).strip() == str(target).strip():
                    correct += 1
            except:
                pass
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"{model_name} Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Show some examples of full reasoning
    print(f"\nFirst 5 examples with full reasoning:")
    for i in range(min(1, len(predictions))):
        question = input_questions[i]
        target = targets[i]
        pred = predictions[i]
        status = "✓" if str(pred).strip() == str(target).strip() else "✗"
        
        # Generate one example to show reasoning
        chat = role(question)
        input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        
        with torch.no_grad():
            def generate_example(input_ids, max_new_tokens=200):
                current_ids = input_ids
                for _ in range(max_new_tokens):
                    outputs = model_instance(current_ids)
                    logits = outputs['logits']
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                gen_tokens = current_ids[:, input_ids.shape[1]:]
                if gen_tokens.numel() > 0:
                    return tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
                return ""
            
            full_reasoning = generate_example(input_ids , max_new_tokens=200)
        
        print(f"\n--- Example {i+1} {status} ---")
        print(f"Question: {question}")
        print(f"Model's reasoning: {full_reasoning[:200]}{'...' if len(full_reasoning) > 500 else ''}")
        print(f"Extracted answer: {pred}")
        print(f"Correct answer: {target}")
        print()
    
    return accuracy, predictions

def get_baseline_accuracy(return_model=False):
    print("\n" + "="*60)
    print("Getting baseline accuracy (full model, no layers dropped)...")
    delete_indices = set([])
    baseline_model = ModifiedLlamaModel(model_base, delete_indices=delete_indices)
    baseline_model.eval()
    baseline_accuracy, _ = evaluate_model(baseline_model, "Baseline-Full-Model")
    
    if return_model:
        return baseline_accuracy, baseline_model
    
    # Clean up memory
    del baseline_model
    torch.cuda.empty_cache()
    return baseline_accuracy

def greedy_layer_dropping():
    """Greedy algorithm to find the best layers to drop while maintaining performance above threshold."""
    
    original_layer_count = len(model_base.model.layers)
    
    # Get baseline accuracy
    baseline_start = time.time()
    baseline_accuracy, baseline_model = get_baseline_accuracy(return_model=True)
    baseline_end = time.time()
    baseline_duration = baseline_end - baseline_start
    
    print(f"Baseline evaluation time: {baseline_duration:.2f} seconds")
    
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
            print(f"Time taken for layer {layer_to_test} config: {time_duration:.2f}s")
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
    
    # Final timing and speedup
    if permanently_dropped_layers:
        print(f"\n{'-'*40}")
        print("Testing final optimized model...")
        final_model = ModifiedLlamaModel(model_base, delete_indices=permanently_dropped_layers)
        final_model.eval()
        
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
        print(f"Final optimized model accuracy: {final_accuracy:.4f}")
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
            print(f"  Dropped so far: {sorted(iter_data['permanently_dropped'])}")
    
    # Clean up baseline_model
    del baseline_model
    torch.cuda.empty_cache()
    
    return permanently_dropped_layers, all_results, baseline_accuracy

if __name__ == "__main__":
    print(f"Loaded {NUM_QUESTIONS} questions from GSM8K test set")
    print(f"Model has {len(model_base.model.layers)} layers")
    
    final_dropped_layers, iteration_results, baseline_acc = greedy_layer_dropping()

    print(f"\n{'='*80}")
    print("ALGORITHM COMPLETED")
    print(f"{'='*80}")
    print(f"Layers to drop for optimal performance: {sorted(final_dropped_layers) if final_dropped_layers else 'None'}")

