import time
import logging
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from jaxtyping import Float, Int
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


def calculate_CTL_loss(
    rewards: Float[torch.Tensor, "num_particles"],
    weights: Float[torch.Tensor, "num_particles"],
    particles: Int[torch.Tensor, "num_particles seq_len"],
    log_psi_record: Float[torch.Tensor, "new_tokens_count num_particles"],
    device: torch.device,
):
    """Build CTL loss"""
    baseline = rewards.mean()
    loss = torch.tensor(0.0, device=device)
    weights = torch.exp(weights)
    norm_w = weights / weights.sum()
    for k in range(len(particles)):
        # sum_{t} log psi_t for particle k   (already a list of scalars)
        logsum_psi = log_psi_record[:, k].sum()
        loss += -norm_w[k] * (rewards[k] - baseline) * logsum_psi
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
    num_particles: int = 4,
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
        num_particles: Number of particles in SMC (samples in a batch over which we jointly optimize)
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
            
            for j in range(twist_updates_per_example):
                if logger:
                    logger.info(f"Processing example {i + 1}/{num_examples}")
            
                # Create prompt
                prompt = create_math_prompt(example["question"])
                
                # always using 1 prompts - hence can squeeze batch dimension
                prompt_tokens: Float[torch.Tensor, "seq_len"] = model.tokenizer.encode(prompt, return_tensors="pt").to(device).squeeze(0)
                
                if logger:
                    logger.debug(f"Prompt: {prompt}")
                    logger.debug(f"Prompt tokens shape: {prompt_tokens.shape}")
                
                # Generate samples using the aligned SMC
                final_particles_sequences, final_log_weights, log_psi_incremental_record, log_z_hat = smc_proposal_sampling(
                    text_prompt_tokens=prompt_tokens,
                    model_forward_fn=model.get_base_model_logits_for_sequence,  # function for p(s_t|s_{1:t-1}) that return log_p
                    twist_forward_fn=model.get_twist_values_for_particles,      # function for psi_t(s_t|s_{1:t-1}) that returns psi (not log_psi!)
                    num_particles=num_particles,
                    new_tokens_count=output_length,
                    device=device,
                    logger_instance=logger,
                    record_log_psi_incrementals=True  # Essential for CTL loss
                )

                if logger:
                    logger.debug(f"  SMC generated {len(final_particles_sequences)} particles.")

                rewards = []
                for particle_seq_tokens in final_particles_sequences:
                    # Decode only the generated part (after prompt)
                    response_tokens = particle_seq_tokens[len(prompt_tokens):]
                    response_text = model.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    reward_val = calculate_reward(response_text, example['answer'], model.tokenizer, logger)
                    rewards.append(reward_val)
                rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float)

                if logger:
                    logger.debug(f"  Rewards calculated (first 5): {rewards_tensor[:5].tolist()}")

                loss = calculate_CTL_loss(rewards=rewards_tensor,
                                          weights=final_log_weights,
                                          particles=final_particles_sequences,
                                          log_psi_record=log_psi_incremental_record,
                                          device=device)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if logger:
                    logger.info(f"  Example {i+1} - Loss: {loss.item():.4f} | Est. Log Z: {log_z_hat.item():.4f}")
            
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


if __name__ == '__main__':
    # Configure logger for detailed output when running standalone
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(ch)
    logger.propagate = False

    train_twist_for_math_problems(
        model_name="gpt2",
        num_examples=2,       # Number of math problems to train on
        num_epochs=1,         # Number of passes through the dataset
        output_length=11,     # Number of tokens to generate per problem
        num_particles=8,  # Number of particles in SMC
        learning_rate=5e-5,
        logger=logger
    )
    print("Test run finished. Check logs for output.")
