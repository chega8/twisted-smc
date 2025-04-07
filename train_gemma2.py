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
    return f"""Solve the following math problem step by step. End your solution with the final answer in a box.

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

def calculate_reward(samples, prompt, correct_answer, tokenizer):
    """
    Calculate rewards for generated sequences.
    
    Args:
        samples: Generated sequences of shape (batch_size, seq_len)
        prompt: Input prompt
        correct_answer: The correct answer to the math problem
        tokenizer: Tokenizer for decoding samples
        
    Returns:
        rewards: Array of shape (batch_size,) containing rewards for each sequence
    """
    batch_size = samples.shape[0]
    rewards = jnp.zeros(batch_size)
    
    for i in range(batch_size):
        # Decode the sample
        sample_text = tokenizer.decode(samples[i], skip_special_tokens=True)
        
        # Extract the answer from the response
        predicted_answer = extract_answer_from_response(sample_text)
        
        # Calculate reward components
        format_reward = 0.0
        correctness_reward = 0.0
        step_by_step_reward = 0.0
        
        # Format reward: check if the response contains \boxed{...}
        if "\\boxed{" in sample_text:
            format_reward = 0.3
        
        # Correctness reward: check if the answer is correct
        if predicted_answer is not None and is_correct_answer(predicted_answer, correct_answer):
            correctness_reward = 0.7
        
        # Step-by-step reward: check if the response contains multiple steps
        if "step" in sample_text.lower() and sample_text.count("\n") > 3:
            step_by_step_reward = 0.2
        
        # Combine rewards
        total_reward = format_reward + correctness_reward + step_by_step_reward
        
        # Set the reward
        rewards = rewards.at[i].set(total_reward)
    
    return rewards

def log_true_final_twist(samples, prompt, correct_answer, tokenizer):
    """
    Calculate log probabilities for the true final twist based on rewards.
    
    Args:
        samples: Generated sequences
        prompt: Input prompt
        correct_answer: The correct answer to the math problem
        tokenizer: Tokenizer for decoding samples
        
    Returns:
        log_probs: Log probabilities for each sequence
    """
    # Calculate rewards
    rewards = calculate_reward(samples, prompt, correct_answer, tokenizer)
    
    # Convert rewards to log probabilities using softmax
    log_probs = jax.nn.log_softmax(rewards)
    
    return log_probs

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
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Shuffle dataset
        indices = np.random.permutation(len(dataset))
        
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            batch_indices = indices[i:i+args.batch_size]
            batch = dataset.select(batch_indices)
            
            for j in range(len(batch)):
                # Get question and answer
                question = batch[j]['question']
                answer = batch[j]['answer'].split("####")[-1].strip()
                
                # Create prompt
                prompt_text = create_math_prompt(question)
                prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").numpy()[0]
                prompt = jnp.array(prompt_ids)
                
                # Define log_true_final_twist function for this example
                def log_true_final_twist_fn(samples, condition_twist_on_tokens=None):
                    return log_true_final_twist(samples, prompt_text, answer, tokenizer)
                
                # Train on this example
                for _ in range(args.twist_updates_per_example):
                    rng_key, subkey = jax.random.split(rng_key)
                    
                    # Compute loss and gradients
                    def loss_fn(params_twist):
                        return train(subkey, prompt, params_twist, params_p,
                                  log_true_final_twist=log_true_final_twist_fn,
                                  output_len=args.output_length,
                                  n_twist=NUM_TWIST_SAMPLES,
                                  condition_twist_on_tokens=None,
                                  smc_procedure_type="smc",
                                  huggingface_model=huggingface_model)
                    
                    # Compute gradients with respect to params_twist only
                    grads = jax.grad(loss_fn)(params_twist)
                    
                    # Update twist parameters
                    updates, optim_twist_state = optimizer_twist.update(grads, optim_twist_state)
                    params_twist = optax.apply_updates(params_twist, updates)
                    
                    # Print loss
                    if _ % 10 == 0:
                        loss = loss_fn(params_twist)
                        print(f"Example {i+j}, Update {_}, Loss: {loss:.4f}")
            
            # Save checkpoint
            if (i + args.batch_size) % args.save_every == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}_batch{i+args.batch_size}.json")
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
    parser.add_argument('--save_every', type=int, default=100, help='Save checkpoint every N batches')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train twist function
    params_twist = train_twist_for_math_problems(args)
    
    # Save twist parameters
    with open(os.path.join(args.output_dir, 'params_twist.json'), 'w') as f:
        json.dump(jax.tree_util.tree_map(lambda x: x.tolist(), params_twist), f)

if __name__ == "__main__":
    # python train_gemma2.py --model_name google/gemma-2-2b-it --num_examples 200 --num_epochs 5 --output_dir math_twist_model
    main() 