import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import re
import os
import json
from tqdm import tqdm
import argparse
from main import (
    stochastic_transformer_sample,
    evaluate_log_psi_selected_tokens,
    binary_cross_entropy,
    get_l_bce,
    train
)
import time

# Constants
MODEL_NAME = "google/gemma-2b-it"
MAX_LENGTH = 1024
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
NUM_TWIST_SAMPLES = 8
OUTPUT_LENGTH = 256

def load_gsm8k_dataset(split='train', num_examples=None):
    """Load GSM8K dataset from Hugging Face."""
    # dataset = datasets.load_dataset("openai/gsm8k", "main")
    dataset = datasets.load_dataset("openai/gsm8k", "main")[split]
    if num_examples is not None:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    return dataset

def create_math_prompt(question):
    """Create a prompt for math problem solving."""
    return f"""Solve the following math problem step by step. End your solution with the final answer in a \\boxed.

Problem: {question}

Solution: Let's solve this step by step:

"""

def extract_answer_from_response(response):
    """Extract the boxed answer from the model's response."""
    # Look for \boxed{...} pattern
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # If no boxed answer, try to find the last number in the response
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', response)
    if numbers:
        return numbers[-1]
    
    return None

def is_correct_answer(predicted, actual):
    """Check if the predicted answer is correct."""
    try:
        # Convert both to float for comparison
        pred_float = float(predicted)
        actual_float = float(actual)
        
        # Check if they're close enough
        return abs(pred_float - actual_float) < 1e-6
    except:
        # If conversion fails, do string comparison
        return predicted.strip() == actual.strip()

def calculate_reward(prompt, answer, tokenizer, model=None):
    """Calculate reward for a given prompt and answer."""
    # print(f"Calculating reward for prompt: {prompt}")
    # print(f"Expected answer: {answer}")
    
    # Initialize reward components
    reward = 0.0
    reward_components = {
        'exact_match': 0.0,
        'partial_match': 0.0,
        'numerical_match': 0.0,
        'length_penalty': 0.0
    }
    
    try:
        # Extract the answer from the response using the extract_answer_from_response function
        extracted_answer = extract_answer_from_response(prompt)
        print(f"Extracted answer: {extracted_answer}")
        
        # Calculate exact match reward using is_correct_answer function
        if extracted_answer is not None:
            if is_correct_answer(extracted_answer, answer):
                reward_components['exact_match'] = 1.0
                print("Exact match found!")
            else:
                print(f"No exact match. Extracted: '{extracted_answer}', Expected: '{answer}'")
        
        # Tokenize prompt and answer for other reward components
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").numpy()[0]
        answer_tokens = tokenizer.encode(answer, return_tensors="pt").numpy()[0]
        
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Answer tokens: {answer_tokens}")
        
        # Calculate partial match reward
        if len(prompt_tokens) > 0 and len(answer_tokens) > 0:
            common_tokens = set(prompt_tokens) & set(answer_tokens)
            if len(common_tokens) > 0:
                reward_components['partial_match'] = len(common_tokens) / max(len(prompt_tokens), len(answer_tokens))
                print(f"Partial match: {reward_components['partial_match']}")
        
        # Calculate numerical match reward
        prompt_numbers = [float(token) for token in prompt.split() if token.replace('.', '').isdigit()]
        answer_numbers = [float(token) for token in answer.split() if token.replace('.', '').isdigit()]
        
        if prompt_numbers and answer_numbers:
            prompt_sum = sum(prompt_numbers)
            answer_sum = sum(answer_numbers)
            if abs(prompt_sum - answer_sum) < 1e-6:
                reward_components['numerical_match'] = 1.0
                print("Numerical match found!")
            else:
                reward_components['numerical_match'] = 1.0 - min(1.0, abs(prompt_sum - answer_sum) / max(abs(prompt_sum), abs(answer_sum)))
                print(f"Numerical match: {reward_components['numerical_match']}")
        
        # Calculate length penalty
        if len(prompt_tokens) > 0 and len(answer_tokens) > 0:
            length_ratio = min(len(prompt_tokens), len(answer_tokens)) / max(len(prompt_tokens), len(answer_tokens))
            reward_components['length_penalty'] = length_ratio
            print(f"Length penalty: {length_ratio}")
        
        # Calculate total reward
        reward = sum(reward_components.values()) / len(reward_components)
        print(f"Total reward: {reward}")
        print(f"Reward components: {reward_components}")
        
    except Exception as e:
        print(f"ERROR in calculate_reward: {e}")
        reward = 0.0
    
    return reward

def log_true_final_twist(samples, prompt, correct_answer, tokenizer):
    """Calculate log true final twist for given samples."""
    print(f"log_true_final_twist called with {len(samples)} samples")
    # print(f"Prompt: {prompt}")
    # print(f"Correct answer: {correct_answer}")
    
    try:
        # Calculate rewards for each sample
        rewards = jnp.zeros(len(samples))
        for i in range(len(samples)):
            # Decode the sample
            sample_text = tokenizer.decode(samples[i], skip_special_tokens=True)
            print(f"Sample {i} text: {sample_text}")
            
            # Calculate reward for this sample
            # Note: We're passing the sample_text as the prompt and correct_answer as the answer
            # This is because calculate_reward expects the model's response as the prompt
            # and the ground truth as the answer
            reward = calculate_reward(sample_text, correct_answer, tokenizer)
            print(f"Reward for sample {i}: {reward}")
            
            # Set the reward
            rewards = rewards.at[i].set(reward)
        
        print(f"Final rewards: {rewards}")
        return rewards
        
    except Exception as e:
        print(f"ERROR in log_true_final_twist: {e}")
        return jnp.zeros(len(samples))

def create_huggingface_model(model_name):
    """Create a HuggingFace model wrapper for JAX."""
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create a wrapper for the model that handles both actual inference and JAX tracing
    def p_model(input_ids, params):
        # For actual inference (not during tracing)
        if not isinstance(input_ids, jax.core.Tracer):
            # Ensure input_ids is a 2D array
            if len(input_ids.shape) == 1:
                input_ids = input_ids[None, :]
            
            # Convert to PyTorch tensor for model input
            input_ids_tensor = torch.tensor(np.array(input_ids), device=device)
            
            # Get logits from the model
            with torch.no_grad():
                outputs = model(input_ids=input_ids_tensor, return_dict=True)
                logits = outputs.logits
            
            # Convert to JAX array
            logits_jax = jnp.array(logits.cpu().numpy())
            return logits_jax
        
        # During tracing, return a placeholder with the correct shape
        batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
        seq_len = input_ids.shape[-1]
        return jnp.zeros((batch_size, seq_len, tokenizer.vocab_size))
    
    def twist_model(input_ids, params):
        # Ensure input_ids is a 2D array
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        
        # Check if params has the expected structure
        if 'transformer' in params:
            # Get the transformer parameters
            w = params['transformer']['w']
            b = params['transformer']['b']
        else:
            # If params doesn't have the expected structure, use default values
            # This is a fallback for when the function is called during JAX transformation
            w = jnp.zeros((tokenizer.vocab_size, 1))
            b = jnp.zeros(1)
        
        # Create one-hot encodings for the input tokens
        batch_size, seq_len = input_ids.shape
        one_hot = jnp.zeros((batch_size, seq_len, tokenizer.vocab_size))
        
        # Set the appropriate indices to 1
        for i in range(batch_size):
            for j in range(seq_len):
                one_hot = one_hot.at[i, j, input_ids[i, j]].set(1.0)
        
        # Apply linear transformation to each token
        twist_values = jnp.zeros((batch_size, seq_len, 1))
        for i in range(batch_size):
            for j in range(seq_len):
                # Get the one-hot vector for this token
                token_one_hot = one_hot[i, j]
                # Apply the linear transformation
                twist_values = twist_values.at[i, j, 0].set(jnp.dot(token_one_hot, w) + b)
        
        return twist_values
    
    return {
        'p': p_model,
        'twist': twist_model,
        'tokenizer': tokenizer
    }

def train_twist_for_math_problems(args):
    """Train twist function for math problem solving."""
    # Initialize random key
    rng_key = jax.random.PRNGKey(args.seed)
    
    # Load dataset
    dataset = load_gsm8k_dataset(split=args.split, num_examples=args.num_examples)
    
    # Create HuggingFace model with token if provided
    huggingface_model = create_huggingface_model(args.model_name)
    tokenizer = huggingface_model['tokenizer']
    
    # Initialize model parameters
    params_p = {}  # Base model parameters (not used in this implementation)
    
    # Initialize twist parameters with the correct structure
    rng_key, subkey = jax.random.split(rng_key)
    params_twist = {
        'transformer': {
            'w': jax.random.normal(subkey, (tokenizer.vocab_size, 1)),
            'b': jnp.zeros(1)
        }
    }
    
    # Initialize optimizer
    optimizer_twist = optax.adam(learning_rate=args.learning_rate)
    optim_twist_state = optimizer_twist.init(params_twist)
    
    # Reduce batch size and output length for memory efficiency
    effective_batch_size = min(args.batch_size, 2)  # Limit batch size to 2
    effective_output_length = min(args.output_length, 64)  # Limit output length to 64
    effective_num_twist_samples = min(NUM_TWIST_SAMPLES, 4)  # Limit number of twist samples to 4
    
    print(f"Using effective batch size: {effective_batch_size}")
    print(f"Using effective output length: {effective_output_length}")
    print(f"Using effective number of twist samples: {effective_num_twist_samples}")
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Shuffle dataset
        indices = np.random.permutation(len(dataset))
        
        for i in tqdm(range(0, len(dataset), effective_batch_size)):
            batch_indices = indices[i:i+effective_batch_size]
            batch = dataset.select(batch_indices)
            
            for j in range(len(batch)):
                # Get question and answer
                question = batch[j]['question']
                answer = batch[j]['answer'].split("####")[-1].strip()
                
                print(f"Processing question: {question}")
                print(f"Correct answer: {answer}")
                
                # Create prompt
                prompt_text = create_math_prompt(question)
                prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").numpy()[0]
                prompt = jnp.array(prompt_ids)
                
                # Define log_true_final_twist function for this example
                def log_true_final_twist_fn(samples, condition_twist_on_tokens=None):
                    print(f"Wrapper function called with {len(samples)} samples")
                    print(f"condition_twist_on_tokens: {condition_twist_on_tokens}")
                    # Make sure we're passing all required parameters
                    result = log_true_final_twist(samples, prompt_text, answer, tokenizer)
                    print(f"log_true_final_twist result: {result}")
                    return result
                
                # Train on this example
                for _ in range(args.twist_updates_per_example):
                    rng_key, subkey = jax.random.split(rng_key)
                    
                    # Set a timeout for the loss computation
                    start_time = time.time()
                    timeout = 60  # 1 minute timeout
                    
                    # Compute loss and gradients
                    def loss_fn(params_twist):
                        print("Processing loss_fn")
                        try:
                            return train(subkey, prompt, params_twist, params_p,
                                      log_true_final_twist=log_true_final_twist_fn,
                                      output_len=effective_output_length,
                                      n_twist=effective_num_twist_samples,
                                      condition_twist_on_tokens=None,
                                      smc_procedure_type="smc",
                                      huggingface_model=huggingface_model)
                        except Exception as e:
                            print(f"ERROR in loss_fn: {e}")
                            return jnp.array(0.0)  # Return a dummy value
                    
                    # Compute gradients with respect to params_twist only
                    try:
                        grads = jax.grad(loss_fn)(params_twist)
                        print("Computed gradients")
                    except Exception as e:
                        print(f"ERROR in gradient computation: {e}")
                        grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params_twist)
                    
                    # Check timeout
                    if time.time() - start_time > timeout:
                        print("Timeout reached during loss and gradient computation")
                        grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params_twist)
                    
                    # Update twist parameters
                    updates, optim_twist_state = optimizer_twist.update(grads, optim_twist_state)
                    params_twist = optax.apply_updates(params_twist, updates)
                    
                    # Print loss
                    try:
                        loss = loss_fn(params_twist)
                        print(f"Example {i+j}, Update {_}, Loss: {loss:.4f}")
                    except Exception as e:
                        print(f"ERROR in loss computation: {e}")
                        loss = jnp.array(0.0)
            
            # Save checkpoint
            if (i + effective_batch_size) % args.save_every == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}_batch{i+effective_batch_size}.json")
                os.makedirs(args.output_dir, exist_ok=True)
                
                # Convert params_twist to a serializable format
                params_twist_serializable = {
                    'transformer': {
                        'w': params_twist['transformer']['w'].tolist(),
                        'b': params_twist['transformer']['b'].tolist()
                    }
                }
                
                with open(checkpoint_path, 'w') as f:
                    json.dump(params_twist_serializable, f)
                
                print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.json")
    params_twist_serializable = {
        'transformer': {
            'w': params_twist['transformer']['w'].tolist(),
            'b': params_twist['transformer']['b'].tolist()
        }
    }
    
    with open(final_path, 'w') as f:
        json.dump(params_twist_serializable, f)
    
    print(f"Saved final model to {final_path}")
    
    return params_twist

def main():
    parser = argparse.ArgumentParser(description='Train twist function for math problems')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Name of the model to use')
    parser.add_argument('--num_examples', type=int, default=100, help='Number of examples to use')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--output_length', type=int, default=OUTPUT_LENGTH, help='Output sequence length')
    parser.add_argument('--twist_updates_per_example', type=int, default=1, help='Number of twist updates per example')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N batches')
    parser.add_argument('--disable_jit', action='store_true', help='Disable JIT compilation for debugging')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Disable JIT if requested
    if args.disable_jit:
        print("Disabling JIT compilation for debugging")
        jax.config.update('jax_disable_jit', True)
    
    # Train twist function
    params_twist = train_twist_for_math_problems(args)
    
    # Save twist parameters
    with open(os.path.join(args.output_dir, 'params_twist.json'), 'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: x.tolist(), params_twist), f)

if __name__ == "__main__":
    # python train_gemma2.py --model_name google/gemma-2-2b-it --num_examples 200 --num_epochs 1 --output_dir math_twist_model --disable_jit
    main() 