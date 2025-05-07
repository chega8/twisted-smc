import torch
import logging
import time


def incremental_weight_update(previous_weights, current_weights):
    # assuming weights are in log space
    # incremental weights should estimate Z_t/Z_{t-1}
    incremental_weights = current_weights - previous_weights
    return incremental_weights


def initialize_particles_and_state(text_prompt_tokens, num_particles, device):

    if isinstance(text_prompt_tokens, torch.Tensor):
        prompt_tensor = text_prompt_tokens.to(dtype=torch.long, device=device)
        if prompt_tensor.ndim == 2 and prompt_tensor.shape[0] == 1:
            prompt_tensor = prompt_tensor.squeeze(0)
        elif prompt_tensor.ndim == 0:
            prompt_tensor = prompt_tensor.unsqueeze(0)
    else:
        prompt_tensor = torch.tensor(text_prompt_tokens, dtype=torch.long, device=device)
        if prompt_tensor.ndim == 0:
            prompt_tensor = prompt_tensor.unsqueeze(0)
        elif prompt_tensor.ndim == 2 and prompt_tensor.shape[0] == 1: # handle [1,N]
            prompt_tensor = prompt_tensor.squeeze(0)

    assert prompt_tensor.ndim == 1, f"Prompt tensor must be 1D after processing, got shape {prompt_tensor.shape}"

    logger.debug(f"Initializing {num_particles} particles from prompt tensor of shape {prompt_tensor.shape}")
    particle_sequences = prompt_tensor.unsqueeze(0).expand(num_particles, -1)
    log_weights = torch.zeros(num_particles, device=device)
    log_p_theta_cumulative = torch.zeros(num_particles, device=device)
    log_psi_cumulative = torch.zeros(num_particles, device=device)
    log_z_hat = torch.tensor(0.0, device=device)
    logger.debug("Initialized particles and SMC states (log_weights, log_p_theta_cumulative, log_psi_cumulative, log_z_hat).")
    return (
        particle_sequences,
        log_weights,
        log_p_theta_cumulative,
        log_psi_cumulative,
        log_z_hat,
    )


def resample_particles_and_state(
    particle_sequences,
    log_weights,
    log_p_theta_cumulative,
    log_psi_cumulative,
    device,
    log_psi_record_for_resampling=None,
    logger_instance=None,
):
    global logger
    if logger_instance: logger = logger_instance

    num_particles = particle_sequences.shape[0]
    if num_particles == 0:
        return (
            particle_sequences,
            log_weights,
            log_p_theta_cumulative,
            log_psi_cumulative,
            log_psi_record_for_resampling,
            torch.empty(0, dtype=torch.long, device=device)
        )

    logger.debug(f"Performing resampling for {num_particles} particles.")
    probabilities = torch.softmax(log_weights, dim=0)
    indices = torch.multinomial(probabilities, num_particles, replacement=True)

    resampled_particle_sequences = particle_sequences[indices]
    resampled_log_p_theta_cumulative = log_p_theta_cumulative[indices]
    resampled_log_psi_cumulative = log_psi_cumulative[indices]

    resampled_log_psi_record = None
    if log_psi_record_for_resampling is not None:
        if isinstance(log_psi_record_for_resampling, list) and all(isinstance(el, list) for el in log_psi_record_for_resampling):
            resampled_log_psi_record = [log_psi_record_for_resampling[i.item()] for i in indices]
        # Add handling if it's a tensor
        elif isinstance(log_psi_record_for_resampling, torch.Tensor) and log_psi_record_for_resampling.ndim >= 1:
            resampled_log_psi_record = log_psi_record_for_resampling[indices]

    new_log_weights = torch.zeros(num_particles, device=device)
    logger.debug("Resampling complete. Log_weights reset.")
    return (
        resampled_particle_sequences,
        new_log_weights,
        resampled_log_p_theta_cumulative,
        resampled_log_psi_cumulative,
        resampled_log_psi_record,
        indices
    )


def smc_proposal_sampling(
    text_prompt_tokens,
    model_forward_fn,
    twist_forward_fn,
    num_particles,
    new_tokens_count,
    device,
    logger_instance=None,
    record_log_psi_incrementals=False,
):
    global logger
    if logger_instance: logger = logger_instance

    logger.info(f"Starting SMC sampling with {num_particles} particles for {new_tokens_count} new tokens.")
    start_time = time.time()

    (
        current_particle_sequences,
        current_log_weights,
        current_log_p_theta_cumulative,
        current_log_psi_cumulative,
        current_log_z_hat,
    ) = initialize_particles_and_state(text_prompt_tokens, num_particles, device)

    log_psi_incremental_selected_record = [[] for _ in range(num_particles)] if record_log_psi_incrementals else None
    assert num_particles > 0, "SMC called with zero particles."
    
    for t in range(new_tokens_count):
        logger.debug(f"SMC Step {t+1}/{new_tokens_count}. Particle sequences shape: {current_particle_sequences.shape}")
        
        previous_log_weights = current_log_weights.clone()

        log_p_incremental_all_vocab = model_forward_fn(current_particle_sequences)
        log_psi_incremental_all_vocab = twist_forward_fn(current_particle_sequences)
        
        # Ensure shapes are (num_particles, vocab_size)
        assert log_p_incremental_all_vocab.ndim == 2 and log_p_incremental_all_vocab.shape[0] == num_particles, f"log_p shape error: {log_p_incremental_all_vocab.shape}"
        assert log_psi_incremental_all_vocab.ndim == 2 and log_psi_incremental_all_vocab.shape[0] == num_particles, f"log_psi shape error: {log_psi_incremental_all_vocab.shape}"

        log_q_unnormalized_all_vocab = log_p_incremental_all_vocab + log_psi_incremental_all_vocab
        log_proposal_norm_const_at_t = torch.logsumexp(log_q_unnormalized_all_vocab, dim=-1, keepdim=True)
        log_q_normalized_all_vocab = log_q_unnormalized_all_vocab - log_proposal_norm_const_at_t

        proposal_probs = torch.exp(log_q_normalized_all_vocab)
        proposal_probs = torch.clamp(proposal_probs, min=1e-20, max=1.0)  # for numerical stability

        logger.debug(f"Proposal probs shape: {proposal_probs.shape}")
        logger.debug(f"Proposal probs: {proposal_probs}")
        try:
            sampled_token_indices = torch.multinomial(proposal_probs, 1)
        except Exception as e:
            logger.error(f"Error sampling token indices: {e}")
            logger.error(f"Proposal probs: {proposal_probs}")
            raise e
        current_particle_sequences = torch.cat((current_particle_sequences, sampled_token_indices), dim=1)

        log_p_incremental_selected = log_p_incremental_all_vocab.gather(-1, sampled_token_indices).squeeze(-1)
        log_psi_incremental_selected = log_psi_incremental_all_vocab.gather(-1, sampled_token_indices).squeeze(-1)
        log_q_normalized_selected = log_q_normalized_all_vocab.gather(-1, sampled_token_indices).squeeze(-1)

        current_log_p_theta_cumulative = current_log_p_theta_cumulative + log_p_incremental_selected
        current_log_psi_cumulative = current_log_psi_cumulative + log_psi_incremental_selected
        
        log_alpha_t = log_p_incremental_selected + log_psi_incremental_selected - log_q_normalized_selected
        current_log_weights = previous_log_weights + log_alpha_t

        log_sum_w_t = torch.logsumexp(current_log_weights, dim=0)
        log_sum_w_t_minus_1 = torch.logsumexp(previous_log_weights, dim=0)
        current_log_z_hat = current_log_z_hat + (log_sum_w_t - log_sum_w_t_minus_1)

        logger.debug(f"  Step {t+1} Log Weights (first 5): {current_log_weights[:min(5, num_particles)].tolist()}")
        logger.debug(f"  Step {t+1} Log Alpha_t (first 5): {log_alpha_t[:min(5, num_particles)].tolist()}")
        logger.debug(f"  Step {t+1} Log Z_hat: {current_log_z_hat.item():.4f}")

        if record_log_psi_incrementals:
            for i in range(num_particles):
                log_psi_incremental_selected_record[i].append(log_psi_incremental_selected[i].detach()) # Detach for CTL

        ess = compute_effective_sample_size(current_log_weights, logger)
        if ess < num_particles / 2 and (t < new_tokens_count - 1):
            logger.info(f"  Step {t+1}: ESS {ess.item():.2f} < {num_particles / 2}. Resampling.")
            (
                current_particle_sequences,
                current_log_weights,
                current_log_p_theta_cumulative,
                current_log_psi_cumulative,
                log_psi_incremental_selected_record,
                _
            ) = resample_particles_and_state(
                current_particle_sequences,
                current_log_weights,
                current_log_p_theta_cumulative,
                current_log_psi_cumulative,
                device,
                log_psi_incremental_selected_record,
                logger,
            )
        elif logger:
             logger.debug(f"  Step {t+1}: ESS {ess.item():.2f} >= {num_particles / 2}. No resampling.")

    logger.info(f"SMC sampling completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Final Log Z_hat: {current_log_z_hat.item():.4f}")
    if num_particles > 0:
        logger.info(f"Final Log Weights (mean): {current_log_weights.mean().item():.4f}")

    return (
        current_particle_sequences, # Now a tensor
        current_log_weights,
        log_psi_incremental_selected_record,
        current_log_z_hat
    )


def compute_effective_sample_size(weights, logger=None):
    """
    Compute the Effective Sample Size (ESS) to assess particle degeneracy.
    
    ESS = (sum(weights)^2) / sum(weights^2)
    
    Args:
        weights (torch.Tensor): Tensor of particle weights. Can be log-weights or normal weights.
                                If log-weights, they are converted to normal weights first.
        logger: Optional logger instance
        
    Returns:
        ess (float): The effective sample size.
    """
    # If weights are log-weights, convert to normal scale for ESS calculation.
    # A common heuristic is to check if most weights are <= 0.
    if torch.all(weights <= 0):  # Heuristic for log-weights
        max_log_weight = weights.max()
        weights_normalized = torch.exp(weights - max_log_weight)  # Subtract max for numerical stability
    else:
        weights_normalized = weights

    # Ensure weights are not all zero to avoid division by zero.
    # If sum of squares is zero, it means all weights are zero.
    sum_sq_weights = (weights_normalized ** 2).sum()
    if sum_sq_weights == 0:
        if logger:
            logger.warning("ESS calculation: All weights are zero. Returning ESS of 0.")
        return torch.tensor(0.0, device=weights.device)
        
    ess = (weights_normalized.sum() ** 2) / sum_sq_weights
    
    if logger:
        logger.debug(f"Computed ESS: {ess.item():.2f}")
    
    return ess


def resample_particles(particles, weights, device, logger=None):
    """
    Resample particles based on their weights to prevent weight degeneracy.
    
    Particles are resampled in proportion to their weights, and the weights are reset to one.
    This function assumes `particles` is a list of lists. 
    If `particles` becomes a tensor, this function will need an update.
    
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
    
    return resampled_particles, new_weights, indices