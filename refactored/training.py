import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import time
import logging
import os

# Change relative imports to absolute imports
from model import ModelWrapper
from rewards import calculate_reward
from smc_sampling import smc_proposal_sampling

def create_math_prompt(question, logger=None):
    """
    Create a math problem prompt.
    
    Args:
        question: The math question
        logger: Optional logger instance
        
    Returns:
        prompt: The formatted prompt
    """
    prompt = f"Solve this math problem: {question}\nLet's think step by step:\n"
    if logger:
        logger.debug(f"Created prompt: {prompt}")
    return prompt

def load_gsm8k_dataset(num_examples, split='train', logger=None):
    """
    Load the GSM8K dataset.
    
    Args:
        num_examples: Number of examples to load
        split: Dataset split to use
        logger: Optional logger instance
        
    Returns:
        dataset: The loaded dataset
    """
    if logger:
        logger.info(f"Loading GSM8K {split} split with {num_examples} examples")
    
    dataset = load_dataset("openai/gsm8k", "main")[split]
    dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    if logger:
        logger.info(f"Loaded {len(dataset)} examples")
    
    return dataset

def compute_binary_cross_entropy_loss(logits, targets, logger=None):
    """
    Compute binary cross entropy loss.
    
    Args:
        logits: Model logits
        targets: Target values
        logger: Optional logger instance
        
    Returns:
        loss: The computed loss
    """
    if logger:
        logger.debug(f"Computing BCE loss for logits shape {logits.shape}, targets shape {targets.shape}")
    
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    
    if logger:
        logger.debug(f"Computed loss: {loss.item():.4f}")
    
    return loss

def calculate_CTL_loss(rewards, particles, log_psi_record, device):
    """Build CTL loss"""
    baseline = rewards.mean() 
    loss = torch.tensor(0.0, device=device)
    for k in range(len(particles)):
        # sum_{t} log psi_t for particle k   (already a list of scalars)
        logsum = torch.stack(log_psi_record[k]).sum()
        loss += -(rewards[k] - baseline) * logsum
    loss = loss / len(particles)
    return loss

def train_twist_for_math_problems(
    model_name: str,
    num_examples: int = 10,
    num_epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    output_dir: str = "output",
    seed: int = 42,
    split: str = "train",
    output_length: int = 200,
    num_twist_samples: int = 4,
    twist_updates_per_example: int = 1,
    save_every: int = 5,
    logger = None
):
    """
    Train twist function for math problems.
    
    Args:
        model_name: Name of the base model
        num_examples: Number of examples to use
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        output_dir: Directory to save outputs
        seed: Random seed
        split: Dataset split to use
        output_length: Length of output sequences
        num_twist_samples: Number of twist samples
        twist_updates_per_example: Number of twist updates per example
        save_every: Save checkpoint every N examples
        logger: Optional logger instance
    """
    if logger:
        logger.info("Starting training")
        start_time = time.time()
    
    # Set random seed
    torch.manual_seed(seed)
    if logger:
        logger.info(f"Set random seed to {seed}")
    
    # Load dataset
    dataset = load_gsm8k_dataset(num_examples, split, logger)
    
    # Create model wrapper
    model = ModelWrapper(model_name, logger=logger)
    device = model.device
    
    # Create optimizer for twist model
    optimizer = torch.optim.Adam(model.twist_model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        if logger:
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        for i, example in enumerate(dataset):
            if i >= num_examples:
                break
                
            if logger:
                logger.info(f"Processing example {i + 1}/{num_examples}")
            
            # Create prompt
            prompt = create_math_prompt(example["question"])
            prompt_tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            if logger:
                logger.debug(f"Prompt: {prompt}")
                logger.debug(f"Prompt tokens shape: {prompt_tokens.shape}")
            
            # Generate samples using SMC
            particles, weights, log_psi_record = smc_proposal_sampling(
                prompt_tokens[0].tolist(),
                model.get_base_model_logits_for_sequence,
                model.get_twist_values_for_particles,
                num_twist_samples,
                output_length,
                device,
                logger,
                record_log_psi=True 
            )
            
            if logger:
                logger.debug(f"Generated {len(particles)} particles")
            
            # Calculate rewards for each particle
            rewards = []
            for particle in particles:
                response = model.tokenizer.decode(particle[len(prompt_tokens[0]):], skip_special_tokens=True)
                reward = calculate_reward(
                    response,
                    example['answer'],
                    model.tokenizer,
                    logger
                )
                rewards.append(reward)
            
            rewards = torch.tensor(rewards, device=device)
            
            if logger:
                logger.debug(f"Rewards: {rewards.tolist()}")
            
            # Compute loss
            # loss = compute_binary_cross_entropy_loss(weights, rewards, logger)

            loss = calculate_CTL_loss(rewards, particles, log_psi_record, device)
            
            if logger:
                # logger.info(f"Loss: {loss.item():.4f}")
                logger.info(f"CTL loss: {loss.item():.4f}  |  rewards: {rewards.tolist()}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Save checkpoint
            if (i + 1) % save_every == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{i + 1}.pt")
                model.save_state(checkpoint_path)
                if logger:
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    model.save_state(final_path)
    if logger:
        logger.info(f"Saved final model to {final_path}")
    
    if logger:
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds") 