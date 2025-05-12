import time
import logging
import os
import torch
import torch.nn as nn
from datasets import load_dataset
# from jaxtyping import Float, Int
from model import ModelWrapper
from rewards import calculate_reward
from smc_sampling import smc_proposal_sampling
import pandas as pd
from utils import TrainLogger


def create_math_prompt(question, logger=None):
    """
    Create a math problem prompt.
    
    Args:
        question: The math question
        logger: Optional logger instance
        
    Returns:
        prompt: The formatted prompt
    """
    boxed_str = '\\boxed{...}'
    prompt = f"Solve this math problem: {question}\nBe brief, use only 100 tokens for the answer.\nPut the final answer in {boxed_str}.\n"
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

def load_qa_dataset(num_examples, logger=None):
    dataset = pd.read_csv('/home/jovyan/fida/code/smc/twisted-smc/refactored/output/data.csv').sample(num_examples).reset_index(drop=True)
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
    rewards: torch.Tensor,
    weights: torch.Tensor,
    particles: torch.Tensor,
    log_psi_record: torch.Tensor,
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

def calculate_CTL_loss(
    rewards: torch.Tensor,          # (K,)
    log_weights: torch.Tensor,      # (K,)  -- **LOG-space** weights w_T
    log_psi_record: torch.Tensor,   # (T, K) -- each entry is log ψ_t(s1:t)
    *,                              # force keyword args below
    entropy_coef: float = 1e-3      # small ψ-entropy regulariser (paper App B)
) -> torch.Tensor:
    # 1)  Convert log-weights → normalised probabilities  (softmax is stable)
    weight_probs = torch.softmax(log_weights, dim=0)       # Σ_k w̄_k = 1

    # 2)  Weighted baseline  (unbiased, lower variance) and detach() it
    baseline = (weight_probs * rewards).sum().detach()

    # 3)  Advantage-weighted policy-gradient term
    #     log_psi_record shaped [T, K].  Sum over T first, then dot with weights.
    logsum_psi_per_particle = log_psi_record.sum(dim=0)    # (K,)
    pg_loss = - (weight_probs * (rewards - baseline) * logsum_psi_per_particle).sum()

    # 4)  Small entropy bonus on ψ  (prevents ψ → +∞ or 0)
    #     H(ψ) = E_w[ ψ*logψ ];  we approximate with the same particles.
    #     log_psi_record is already log ψ;  exp(log ψ) = ψ
    entropy_term = (weight_probs.unsqueeze(0) *
                    torch.exp(log_psi_record) * log_psi_record).mean()

    loss = pg_loss / rewards.numel() + entropy_coef * entropy_term + 1e-10
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
        
    logger_png = TrainLogger(output_dir + "/plots")      # e.g. ./output/plots
    
    # Load dataset
    # dataset = load_gsm8k_dataset(num_examples, split, logger)
    dataset = load_qa_dataset(num_examples, logger)
    
    # Create model wrapper
    model = ModelWrapper(model_name, logger=logger)
    device = model.device
    
    # Create optimizer for twist model
    optimizer = torch.optim.Adam(model.twist_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.99, total_iters=500)   # first 500 steps

    
    # Training loop
    for epoch in range(num_epochs):
        if logger:
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # for i, example in enumerate(dataset):
        for i, example in dataset.iterrows():
            if i >= num_examples:
                break
            
            for j in range(twist_updates_per_example):
                if logger:
                    logger.info(f"Processing example {i + 1}/{num_examples}")
            
                # Create prompt
                # prompt = create_math_prompt(example["question"])
                prompt = example["question"]
                
                # always using 1 prompts - hence can squeeze batch dimension
                # prompt_tokens: Float[torch.Tensor, "seq_len"] = model.tokenizer.encode(prompt, return_tensors="pt").to(device).squeeze(0)
                prompt_tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(device).squeeze(0)
                
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
                    # reward_val = calculate_reward(response_text, None, model.tokenizer, logger)
                    rewards.append(reward_val)
                rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float)

                if logger:
                    logger.debug(f"  Rewards calculated (first 5): {rewards_tensor[:5].tolist()}")

                # loss = calculate_CTL_loss(rewards=rewards_tensor,
                #                           log_weights=final_log_weights,
                #                           particles=final_particles_sequences,
                #                           log_psi_record=log_psi_incremental_record,
                #                           device=device)
                
                loss = calculate_CTL_loss(
                    rewards=rewards_tensor,            # shape (K,)
                    log_weights=final_log_weights,     # still log-space, shape (K,)
                    log_psi_record=log_psi_incremental_record,   # shape (T, K)
                    entropy_coef=1e-3)                 # optional, change if you like


                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.twist_model.parameters(), 1.0)
                scheduler.step()
                optimizer.step()

                for n, p in model.twist_model.named_parameters():
                    print(n, p.grad is None,
                        0.0 if p.grad is None else p.grad.abs().mean().item())

                if logger:
                    logger.info(f"\n\n  Example {i+1} - Loss: {loss.item():.4f} | Est. Log Z: {log_z_hat.item():.4f}\n\n")
                    
                global_step = epoch * num_examples + i + j
                logger_png.log(
                        step=global_step,
                        loss=loss.item(),
                        logZ=log_z_hat.item(),
                        reward_mean=sum(rewards) / len(rewards)
                )
            
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

    logger_png.plot_all()


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
