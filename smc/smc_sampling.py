import torch
import logging
import time
from typing import List, Tuple, Union, Optional, Callable

# from jaxtyping import Float, Int

# import torch
# torch.autograd.set_detect_anomaly(True)       # put once at program start


def log_stats(t: torch.Tensor, name: str, logger):
    if not logger.isEnabledFor(logging.DEBUG):    # skip if not in debug mode
        return
    logger.debug(f"{name:<12}  shape {tuple(t.shape)}  "
                 f"min {t.min().item():9.3e}  max {t.max().item():9.3e}  "
                 f"nan {torch.isnan(t).any().item()}  inf {torch.isinf(t).any().item()}")



def incremental_weight_update(previous_log_weights, current_log_weights):
    # assuming weights are in log space
    # incremental weights should estimate Z_t/Z_{t-1}
    incremental_log_weights = current_log_weights - previous_log_weights
    return incremental_log_weights


def initialize_particles_and_state(text_prompt_tokens: list[int], num_particles: int, device: torch.device):

    if isinstance(text_prompt_tokens, torch.Tensor):
        prompt_tensor = text_prompt_tokens.to(dtype=torch.long, device=device)
        if prompt_tensor.ndim == 2 and prompt_tensor.shape[0] == 1:
            # just for extra safety / convenience
            prompt_tensor = prompt_tensor.squeeze(0)
        elif prompt_tensor.ndim == 0:
            prompt_tensor = prompt_tensor.unsqueeze(0)
    else:
        # just for extra safety / convenience
        prompt_tensor = torch.tensor(text_prompt_tokens, dtype=torch.long, device=device)
        if prompt_tensor.ndim == 0:
            prompt_tensor = prompt_tensor.unsqueeze(0)
        elif prompt_tensor.ndim == 2 and prompt_tensor.shape[0] == 1:
            prompt_tensor = prompt_tensor.squeeze(0)

    assert prompt_tensor.ndim == 1, f"Prompt tensor must be 1D after processing, got shape {prompt_tensor.shape}"

    logger.debug(f"Initializing {num_particles} particles from prompt tensor of shape {prompt_tensor.shape}")
    particle_sequences = prompt_tensor.unsqueeze(0).expand(num_particles, -1)
    log_weights = torch.zeros(num_particles, device=device)
    log_p_cumulative = torch.zeros(num_particles, device=device)
    log_psi_cumulative = torch.zeros(num_particles, device=device)
    log_z_hat = torch.tensor(0.0, device=device)
    logger.debug("Initialized particles and SMC states (log_weights, log_p_cumulative, log_psi_cumulative, log_z_hat).")
    return (
        particle_sequences,
        log_weights,
        log_p_cumulative,
        log_psi_cumulative,
        log_z_hat,
    )


def resample_particles_and_state(
    particle_sequences,
    log_weights,
    log_p_cumulative,
    log_psi_cumulative,
):
    num_particles = particle_sequences.shape[0]
    assert num_particles > 0, "SMC called with zero particles."

    probabilities = torch.softmax(log_weights, dim=0)
    indices = torch.multinomial(probabilities, num_particles, replacement=True)

    rs_particle_sequences = particle_sequences[indices]
    rs_log_p_cumulative = log_p_cumulative[indices]
    rs_log_psi_cumulative = log_psi_cumulative[indices]

    return indices, (rs_particle_sequences, rs_log_p_cumulative, rs_log_psi_cumulative)


def smc_proposal_sampling(
    text_prompt_tokens,  # here assuming batch_size dim is squeezed
    model_forward_fn: Callable,
    twist_forward_fn: Callable,
    num_particles: int,
    new_tokens_count: int,
    device: torch.device,
    logger_instance: Optional[logging.Logger] = None,
    record_log_psi_incrementals: bool = False,
):
    global logger
    if logger_instance: logger = logger_instance

    logger.info(f"Starting SMC sampling with {num_particles} particles for {new_tokens_count} new tokens.")
    start_time = time.time()

    (
        curr_particle_sequences,
        curr_log_weights,
        curr_log_p_cumulative,
        curr_log_psi_cumulative,
        curr_log_z_hat,
    ) = initialize_particles_and_state(text_prompt_tokens, num_particles, device)

    if record_log_psi_incrementals:
        # log_psi_incremental_selected_record = torch.zeros(new_tokens_count, num_particles, device=device)
        log_psi_incremental_selected_record = []
    else:
        log_psi_incremental_selected_record = None
    assert num_particles > 0, "SMC called with zero particles."
    
    
    MAX_LOG_PSI = 6.0                    # e^6 ≈ 403
    MIN_LOG_PSI = -20.0                  # e^-20 ≈ 2e-9
    for t in range(new_tokens_count):
        logger.debug(f"SMC Step {t+1}/{new_tokens_count}. Particle sequences shape: {curr_particle_sequences.shape}")
        
        # the whole idea is to update these weights during the sequence generation
        # and to use them for resampling of running sequences in the batch (particles)
        previous_log_weights = curr_log_weights.clone()

        # only the last position
        log_p_incremental_all_vocab = model_forward_fn(curr_particle_sequences)
        psi_incremental_all_vocab = twist_forward_fn(curr_particle_sequences)
        log_psi_incremental_all_vocab = torch.log(psi_incremental_all_vocab)
        
        log_stats(psi_incremental_all_vocab,     "psi",      logger)
        log_stats(log_psi_incremental_all_vocab, "log_psi", logger)
        
        # Ensure shapes are (num_particles, vocab_size)
        assert log_p_incremental_all_vocab.ndim == 2 and log_p_incremental_all_vocab.shape[0] == num_particles, f"log_p shape error: {log_p_incremental_all_vocab.shape}"
        assert log_psi_incremental_all_vocab.ndim == 2 and log_psi_incremental_all_vocab.shape[0] == num_particles, f"log_psi shape error: {log_psi_incremental_all_vocab.shape}"

        log_q_unnormalized_all_vocab = log_p_incremental_all_vocab + log_psi_incremental_all_vocab
        log_q_Z_t = torch.logsumexp(log_q_unnormalized_all_vocab, dim=-1, keepdim=True)
        log_q_normalized_all_vocab = log_q_unnormalized_all_vocab - log_q_Z_t

        proposal_probs = torch.exp(log_q_normalized_all_vocab)
        proposal_probs = torch.clamp(proposal_probs, min=1e-20, max=1.0)  # for numerical stability

        try:
            sampled_token_indices = torch.multinomial(proposal_probs, 1)
        except Exception as e:
            logger.error(f"Error sampling token indices: {e}")
            logger.error(f"Proposal probs: {proposal_probs}")
            raise e
        # just attach last token index to a sequence
        curr_particle_sequences = torch.cat((curr_particle_sequences, sampled_token_indices), dim=1)

        # all below are also Float[torch.Tensor, "num_particles"]
        log_p_incremental_selected = log_p_incremental_all_vocab.gather(-1, sampled_token_indices).squeeze(-1)
        log_psi_incremental_selected = log_psi_incremental_all_vocab.gather(-1, sampled_token_indices).squeeze(-1)
        log_q_normalized_selected = log_q_normalized_all_vocab.gather(-1, sampled_token_indices).squeeze(-1)

        log_stats(proposal_probs,                "proposal", logger)
        
        curr_log_p_cumulative = curr_log_p_cumulative + log_p_incremental_selected
        curr_log_psi_cumulative = curr_log_psi_cumulative + log_psi_incremental_selected
        
        log_alpha_t = log_p_incremental_selected + log_psi_incremental_selected - log_q_normalized_selected
        
        # NOTE: Below is *the key* assumption for the Proposal \prod_t{p_{1:t}*psi_{t}}
        # Note that psi_t here must be conjugated with p_t *at every step*
        # and  that p_{1:t} is a *cumulative* prob of a sequence
        # Under such Proposal, the *Average* Incremental Weight [\hat{w}_t / \hat{w}_{t-1}]
        # is approximately equal to the update in normalizing constants Z_t/Z_{t-1}!
        curr_log_weights = previous_log_weights + log_alpha_t

        log_sum_w_t = torch.logsumexp(curr_log_weights, dim=0)
        log_sum_w_t_minus_1 = torch.logsumexp(previous_log_weights, dim=0)
        # logsumexp gives *averaged* weights, and below we do an update to current time step, from previous one.
        # here we just get updated Z_t from the importance weight update, to reduce number of operations
        # but this can be very confusing without noting that [\hat{w}_t / \hat{w}_{t-1}] = Z_t/Z_{t-1}
        curr_log_z_hat = curr_log_z_hat + (log_sum_w_t - log_sum_w_t_minus_1)

        logger.debug(f"  Step {t+1} Log Weights (first 5): {[round(w, 3) for w in curr_log_weights[:min(5, num_particles)].cpu().tolist()]}")
        logger.debug(f"  Step {t+1} Log Alpha_t (first 5): {[round(a, 3) for a in log_alpha_t[:min(5, num_particles)].cpu().tolist()]}")
        logger.debug(f"  Step {t+1} Log Z_hat: {curr_log_z_hat.item():.4f}")


        ess = compute_effective_sample_size(curr_log_weights, logger)
        resampled_particle_indices = None
        # NOTE: current_log_weights is passed as is to future time steps if resampling is not performed!
        # we only set log_weights to zeros (default) *after* resampling step - this is still somewhat arbitrary imo.
        if ess < num_particles / 2 and (t < new_tokens_count - 1):
            logger.debug(f"  Step {t+1}: ESS {ess.item():.2f} < {num_particles / 2}. Resampling.")
            (
                resampled_particle_indices,
                (
                    curr_particle_sequences,
                    curr_log_p_cumulative,
                    curr_log_psi_cumulative,
                )
            ) = resample_particles_and_state(
                curr_particle_sequences,
                curr_log_weights,
                curr_log_p_cumulative,
                curr_log_psi_cumulative,
            )
            # Important Step from paper: reset log_weights to zeros after resampling
            curr_log_weights = torch.zeros(num_particles, device=device)
        elif logger:
            logger.debug(f"  Step {t+1}: ESS {ess.item():.2f} >= {num_particles / 2}. No resampling.")


        if record_log_psi_incrementals:
            if resampled_particle_indices is not None:
                log_psi_incremental_selected = log_psi_incremental_selected[resampled_particle_indices]
            # log_psi_incremental_selected_record[t, :] = log_psi_incremental_selected.detach()
            log_psi_incremental_selected_record.append(log_psi_incremental_selected)


    if record_log_psi_incrementals:
        log_psi_incremental_record = torch.stack(log_psi_incremental_selected_record)  # (T, K)
    else:
        log_psi_incremental_record = None


    logger.info(f"SMC sampling completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Final Log Z_hat: {curr_log_z_hat.item():.4f}")
    if num_particles > 0:
        logger.info(f"Final Log Weights (mean): {curr_log_weights.mean().item():.4f}")

    return (
        curr_particle_sequences,  # tensor
        curr_log_weights,
        log_psi_incremental_record,
        curr_log_z_hat
    )


def compute_effective_sample_size(weights, logger: Optional[logging.Logger] = None) -> float:
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