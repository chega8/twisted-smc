import torch
import logging
import time

def initialize_particles(text_prompt, num_particles, device, logger=None):
    """
    Initialize particles from a given text prompt.
    
    Each particle is a copy of the provided prompt.
    
    Args:
        text_prompt (list[int]): Initial sequence of token indices (e.g., from a text prompt).
        num_particles (int): Number of particles to initialize.
        device (torch.device): Device on which tensors will be allocated.
        logger: Optional logger instance
        
    Returns:
        particles (list[list[int]]): List containing num_particles copies of the text_prompt.
        weights (torch.Tensor): Tensor of initial weights (all ones), shape (num_particles,).
    """
    if logger:
        logger.debug(f"Initializing {num_particles} particles from prompt of length {len(text_prompt)}")
    
    particles = [list(text_prompt) for _ in range(num_particles)]
    weights = torch.ones(num_particles, device=device)
    
    if logger:
        logger.debug(f"Initialized particles and weights of shape {weights.shape}")

    return particles, weights

def sample_next_token(particles, model, twist, device, logger=None):
    """
    Sample the next token for each particle using the base model and twist function.
    
    For each particle:
      - Compute base logits from the model given its current sequence.
      - Convert logits to probabilities.
      - Compute twist scores and form a twisted proposal distribution.
      - Sample the next token from the proposal distribution.
      - Return the normalization constant for this step.
    
    Args:
        particles (list[list[int]]): Current sequences (each particle is a list of token indices).
        model (callable): Function that given a sequence returns logits for the next token.
                           Expected output: torch.Tensor of shape (vocab_size,).
        twist (callable): Function that given a list of particles returns twist scores.
                          Expected output: torch.Tensor of shape (num_particles, vocab_size).
        device (torch.device): Device for tensor computations.
        logger: Optional logger instance
        
    Returns:
        new_particles (list[list[int]]): Updated particles with the newly sampled token appended.
        norm_constants (torch.Tensor): Normalization constants Z_t for each particle (shape: (num_particles,)).
    """
    if logger:
        logger.debug(f"Sampling next token for {len(particles)} particles")
    
    num_particles = len(particles)
    # Compute base logits for each particle and stack them to form a batch tensor.
    batch_logits = []
    for seq in particles:
        logits = model(seq)  # Expected shape: (vocab_size,)
        batch_logits.append(logits)
    batch_logits = torch.stack(batch_logits, dim=0).to(device)
    
    if logger:
        logger.debug(f"Computed base logits of shape {batch_logits.shape}")
    
    # Compute base probabilities.
    base_prob = torch.softmax(batch_logits, dim=-1)  # Shape: (num_particles, vocab_size)
    
    # Get twist scores for the current particles.
    psi = twist(particles).to(device)  # Expected shape: (num_particles, vocab_size)
    
    if logger:
        logger.debug(f"Computed twist scores of shape {psi.shape}")
    
    # Form the twisted (unnormalized) proposal distribution.
    twisted = base_prob * psi  # Elementwise multiplication.
    
    # Compute the normalization constant Z_t for each particle.
    norm_constants = twisted.sum(dim=-1) + 1e-10  # Shape: (num_particles,)
    
    # Normalize to obtain the proposal distribution.
    proposal = twisted / norm_constants.unsqueeze(-1)  # Shape: (num_particles, vocab_size)
    
    # Sample one token per particle from the categorical proposal.
    new_particles = []
    for i in range(num_particles):
        token = torch.multinomial(proposal[i], 1)  # Sample one token.
        new_seq = particles[i] + [token.item()]
        new_particles.append(new_seq)
    
    if logger:
        logger.debug(f"Sampled next tokens, new particles length: {len(new_particles[0])}")
    
    return new_particles, norm_constants

def compute_effective_sample_size(weights, logger=None):
    """
    Compute the Effective Sample Size (ESS) to assess particle degeneracy.
    
    ESS = (sum(weights)^2) / sum(weights^2)
    
    Args:
        weights (torch.Tensor): Tensor of particle weights.
        logger: Optional logger instance
        
    Returns:
        ess (float): The effective sample size.
    """
    ess = (weights.sum() ** 2) / (weights ** 2).sum()
    
    if logger:
        logger.debug(f"Computed ESS: {ess.item():.2f}")
    
    return ess

def resample_particles(particles, weights, device, logger=None):
    """
    Resample particles based on their weights to prevent weight degeneracy.
    
    Particles are resampled in proportion to their weights, and the weights are reset to one.
    
    Args:
        particles (list[list[int]]): Current particles.
        weights (torch.Tensor): Tensor of current weights (shape: (num_particles,)).
        device (torch.device): Device for tensor computations.
        logger: Optional logger instance
        
    Returns:
        resampled_particles (list[list[int]]): Particles after resampling.
        new_weights (torch.Tensor): Reset weights (all ones), shape: (num_particles,).
    """
    if logger:
        logger.debug("Performing resampling")
    
    num_particles = len(particles)
    indices = torch.multinomial(weights, num_particles, replacement=True)
    # Explicitly convert tensor indices to integers.
    resampled_particles = [particles[i.item()] for i in indices]
    new_weights = torch.ones(num_particles, device=device)
    
    if logger:
        logger.debug("Resampling complete")
    
    return resampled_particles, new_weights

def smc_proposal_sampling(text_prompt, model, twist, num_particles, new_tokens_count, device, logger=None):
    """
    Perform SMC-PROPOSAL sampling starting from a given text prompt.
    
    The algorithm generates new tokens sequentially, updating particle weights using
    normalization constants from the twisted proposal distribution. If the effective
    sample size falls below a threshold, particles are resampled.
    
    Args:
        text_prompt (list[int]): Initial sequence of token indices (text prompt).
        model (callable): Function that given a sequence returns logits for the next token.
                          Expected output for each call: torch.Tensor of shape (vocab_size,).
        twist (callable): Function that given a list of particles returns twist scores.
                          Expected output: torch.Tensor of shape (num_particles, vocab_size).
        num_particles (int): Number of particles to maintain.
        new_tokens_count (int): Number of new tokens to generate (i.e., generation steps beyond the prompt).
        device (torch.device): Device for tensor computations.
        logger: Optional logger instance
        
    Returns:
        particles (list[list[int]]): Final sequences (each a list of token indices).
        weights (torch.Tensor): Final importance weights for each particle.
    """
    if logger:
        logger.info(f"Starting SMC sampling with {num_particles} particles")
        start_time = time.time()
    
    # Initialize particles with the text prompt.
    particles, weights = initialize_particles(text_prompt, num_particles, device, logger)
    
    # Generate new tokens sequentially.
    for t in range(new_tokens_count):
        if logger:
            logger.debug(f"Generating token {t+1}/{new_tokens_count}")
        
        # Sample the next token for each particle.
        particles, norm_constants = sample_next_token(particles, model, twist, device, logger)
        
        # Update the weights by multiplying with the normalization constants.
        weights = weights * norm_constants
        
        # Optionally perform resampling if the effective sample size (ESS) is too low.
        ess = compute_effective_sample_size(weights, logger)
        if ess < num_particles / 2:
            if logger:
                logger.debug(f"ESS {ess.item():.2f} below threshold, resampling")
            particles, weights = resample_particles(particles, weights, device, logger)
    
    if logger:
        logger.info(f"SMC sampling completed in {time.time() - start_time:.2f} seconds")
    
    return particles, weights 