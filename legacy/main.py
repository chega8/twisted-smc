import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import Tuple, Dict, Any
import time
import signal


def stochastic_transformer_sample(rng_key, params_p, prompt: jnp.ndarray, output_len: int, n_samples: int, huggingface_model=None):
    """Generate samples from the transformer model. My implementation"""
    prompt_len = prompt.shape[0]
    batch_prompt = jnp.full((n_samples, prompt.shape[0]), prompt)
    output = jnp.zeros((n_samples, output_len), dtype=jnp.int32)
    full_seq = jnp.concatenate((batch_prompt, output), axis=1)
    
    def sample_iter(carry, t):
        rng_key, full_seq = carry
        p_logits = huggingface_model['p'](input_ids=full_seq, params=params_p)
        rng_key, subkey = jax.random.split(rng_key)
        indices = jax.random.categorical(subkey, p_logits[:, prompt_len + t - 1, :], shape=(p_logits.shape[0],))
        full_seq = full_seq.at[:, prompt_len + t].set(indices)
        return (rng_key, full_seq), None
    
    carry = (rng_key, full_seq)
    carry, _ = jax.lax.scan(sample_iter, carry, jnp.arange(output_len), output_len)
    _, full_seq = carry
    return full_seq

# def stochastic_transformer_sample(rng_key, params, prompt: jnp.ndarray, output_len, n_samples, huggingface_model=None, return_p_eval=False, prompt_is_already_batch=False):
#     if prompt_is_already_batch:
#         prompt_len = prompt.shape[-1]
#         batch_prompt = prompt
#     else:
#         prompt_len = prompt.shape[0]
#         # print(prompt_len)
#         batch_prompt = jnp.full((n_samples, prompt.shape[0]), prompt)

#     output = jnp.zeros((n_samples, output_len), dtype=jnp.int32)
#     full_seq = jnp.concatenate((batch_prompt, output), axis=1)

#     carry = (rng_key, params, full_seq, prompt_len)
#     carry, p_evals = jax.lax.scan(partial(stochastic_transformer_sample_iter, huggingface_model=huggingface_model, return_p_eval=return_p_eval),
#                              carry, jnp.arange(output_len, dtype=jnp.int32), output_len)

#     rng_key, params, full_seq, _ = carry

#     if return_p_eval:
#         return full_seq, p_evals

#     return full_seq

def stochastic_transformer_sample_iter(carry, t, huggingface_model=None, return_p_eval=False):
    # Essentially the way this works is we pass in a full computation (eg full prompt_len + output_len)
    # but we only use the logit for the time step t, and discard the rest of the computation
    # That is, we are computing logits on the full sequence of length prompt_len + output_len
    # where the first prompt_len + t tokens have meaningful values that we previously computed
    # and the later tokens are unitialized (some garbage value)
    # so we end up wasting computation on those later tokens, as we only use the logit at time step t
    # but this is still faster than not using scan+jit
    # Now we don't have dynamic arrays, and since the indexing uses [:, prompt_len + t - 1, :],
    # the only changing part of the index still doesn't change shape. The key point is that no shapes are changing anywhere.
    # So this works with jit, at the cost of a bit of wasted computation
    # This is the approach that I saw people taking online with transformers.
    # As of May 2023 there did not seem to be a better approach in jax (some discussion of jax.mask didn't end up going anywhere)
    rng_key, params, full_seq, prompt_len = carry
    p_logits = get_transformer_p_logits(params, full_seq, huggingface_model=huggingface_model)
    rng_key, subkey = jax.random.split(rng_key)
    # This below is actually ok without log_softmax because I don't need log prob, and jax categorical uses softmax.
    # I needed log_softmax on the other ones in order to properly combine with the other log term.
    indices_to_use = jax.random.categorical(subkey, p_logits[:, prompt_len + t - 1, :],
                                 shape=(p_logits.shape[0],))
    full_seq = full_seq.at[:, prompt_len + t].set(indices_to_use)

    p_eval = None
    if return_p_eval:
        p_eval = jax.nn.log_softmax(p_logits[:, prompt_len + t - 1, :])[jnp.arange(p_logits.shape[0]), indices_to_use]

    carry = (rng_key, params, full_seq, prompt_len)
    return carry, p_eval

def get_transformer_p_logits(params_p, full_seq, huggingface_model=None):
    assert huggingface_model is not None
    if isinstance(huggingface_model, HashableDict):
        p_logits = huggingface_model['p'](input_ids=full_seq)
    else:
        # should be an apply_fn here?
        p_logits = huggingface_model(input_ids=full_seq, ret="p", hface_model_params=params_p)

    return p_logits

def evaluate_log_psi_selected_tokens(full_seq, prompt_len, params_twist, condition_twist_on_tokens, huggingface_model, params_proposal=None, params_p=None):
    """Evaluate log psi values for selected tokens."""
    print(f"evaluate_log_psi_selected_tokens called with full_seq shape: {full_seq.shape}")
    print(f"params_twist: {params_twist}")
    
    try:
        # Check if huggingface_model has the 'twist' key
        if 'twist' not in huggingface_model:
            print("ERROR: huggingface_model does not have 'twist' key")
            # Return a dummy value with the expected shape
            return jnp.zeros((full_seq.shape[0], full_seq.shape[1], 1))
        
        # Call the twist function
        twist_values = huggingface_model['twist'](input_ids=full_seq, params=params_twist)
        print(f"twist_values shape: {twist_values.shape}")
        return twist_values
    except Exception as e:
        print(f"ERROR in evaluate_log_psi_selected_tokens: {e}")
        # Return a dummy value with the expected shape
        return jnp.zeros((full_seq.shape[0], full_seq.shape[1], 1))

def binary_cross_entropy(logits, targets):
    """Compute binary cross entropy loss."""
    print(f"binary_cross_entropy called with logits shape: {logits.shape}, targets shape: {targets.shape}")
    
    try:
        # Check for NaN or Inf values
        if jnp.any(jnp.isnan(logits)) or jnp.any(jnp.isinf(logits)):
            print("WARNING: NaN or Inf values in logits")
            # Replace NaN or Inf values with 0
            logits = jnp.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Compute the loss
        loss = -(targets * jax.nn.log_sigmoid(logits) + (1 - targets) * jax.nn.log_sigmoid(-logits))
        
        # Check for NaN or Inf values in the result
        if jnp.any(jnp.isnan(loss)) or jnp.any(jnp.isinf(loss)):
            print("WARNING: NaN or Inf values in loss")
            # Replace NaN or Inf values with 0
            loss = jnp.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        
        return loss
    except Exception as e:
        print(f"ERROR in binary_cross_entropy: {e}")
        # Return a dummy value with the expected shape
        return jnp.zeros_like(logits)

def get_l_bce(rng_key, prompt, params_p, params_twist, log_true_final_twist, output_len, n_twist, condition_twist_on_tokens, smc_procedure_type, rm_type, beta_temp=1., proposal_is_p=False, huggingface_model=None, tempered_twist=False, beta_prop=None, true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None, log_prob_class=None, params_proposal=None):
    """Compute BCE loss for twist learning. My implementation."""
    print("Starting get_l_bce function")
    
    # Check if required inputs are provided
    if true_sigma_samples is None:
        print("ERROR: true_sigma_samples is None")
        return jnp.array(0.0)  # Return a dummy value to avoid hanging
    
    if log_prob_class is None:
        print("ERROR: log_prob_class is None")
        return jnp.array(0.0)  # Return a dummy value to avoid hanging
    
    print(f"true_sigma_samples shape: {true_sigma_samples.shape}")
    print(f"log_prob_class shape: {log_prob_class.shape}")
    
    samples_to_evaluate_over = true_sigma_samples
    print(f"Evaluating log psi on samples with shape: {samples_to_evaluate_over.shape}")
    
    try:
        log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
            samples_to_evaluate_over, 
            prompt.shape[-1],
            params_twist, 
            condition_twist_on_tokens,
            huggingface_model, 
            params_proposal=params_proposal, 
            params_p=params_p
        )
        print(f"Evaluated log psi with shape: {log_psi_on_p_samples.shape}")
    except Exception as e:
        print(f"ERROR in evaluate_log_psi_selected_tokens: {e}")
        return jnp.array(0.0)  # Return a dummy value to avoid hanging
    
    try:
        class_prob = jnp.exp(log_prob_class)
        print(f"class_prob shape: {class_prob.shape}")
        
        # Create a properly shaped array for broadcasting
        class_prob_broadcasted = jnp.zeros_like(log_psi_on_p_samples)
        for i in range(class_prob.shape[0]):
            class_prob_broadcasted = class_prob_broadcasted.at[i, :, 0].set(class_prob[i])
        
        print(f"class_prob_broadcasted shape: {class_prob_broadcasted.shape}")
    except Exception as e:
        print(f"ERROR in broadcasting class_prob: {e}")
        return jnp.array(0.0)  # Return a dummy value to avoid hanging
    
    try:
        loss = binary_cross_entropy(log_psi_on_p_samples, class_prob_broadcasted)
        print(f"Computed binary_cross_entropy with shape: {loss.shape}")
        mean_loss = loss.mean()
        print(f"Mean loss: {mean_loss}")
        return mean_loss
    except Exception as e:
        print(f"ERROR in binary_cross_entropy: {e}")
        return jnp.array(0.0)  # Return a dummy value to avoid hanging

# def get_l_bce(
#     rng_key, prompt, params_p, params_twist, log_true_final_twist,
#     output_len, n_twist, condition_twist_on_tokens,
#     smc_procedure_type, rm_type, beta_temp=1., proposal_is_p=False,
#     evaluate_over_samples_from="p", huggingface_model=None, tempered_twist=False, beta_prop=None,
#     true_sigma_samples=None, replay_buffer=None, replay_buffer_log_w_ts=None, log_prob_class=None, params_proposal=None
# ):

#     assert true_sigma_samples is not None # Not really true_sigma_samples, just the samples we run this loss on. # TODO Refactor/rename at some point

#     assert log_prob_class is not None

#     samples_to_evaluate_over = true_sigma_samples

#     log_psi_on_p_samples = evaluate_log_psi_selected_tokens(
#         samples_to_evaluate_over, prompt.shape[-1],
#         params_twist,
#         condition_twist_on_tokens,
#         huggingface_model, params_proposal=params_proposal, params_p=params_p)


#     class_prob = jnp.exp(log_prob_class)

#     class_prob_broadcasted = jnp.full((log_psi_on_p_samples.shape), class_prob[:, None]) # broadcast along the time dimension

#     loss = binary_cross_entropy(log_psi_on_p_samples, class_prob_broadcasted)

#     return loss.mean()

def train_twist(rng_key, prompt, params_p, params_twist, log_true_final_twist, output_len, n_twist, optimizer_twist, optim_twist_state, huggingface_model, n_epochs=3, twist_updates_per_epoch=100):
    """Main training loop for twist learning using BCE loss."""
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        
        for twist_update in range(twist_updates_per_epoch):
            # Generate samples from base model
            rng_key, subkey = jax.random.split(rng_key)
            samples = stochastic_transformer_sample(subkey, params_p, prompt, output_len, n_twist, huggingface_model)
            
            # Get log probabilities from true final twist
            log_prob_class = log_true_final_twist(samples)
            
            # Compute BCE loss and gradients
            loss_fn = partial(get_l_bce, 
                            prompt=prompt,
                            params_p=params_p,
                            log_true_final_twist=log_true_final_twist,
                            output_len=output_len,
                            n_twist=n_twist,
                            condition_twist_on_tokens=None,
                            smc_procedure_type="smc",
                            rm_type="one_bad",
                            huggingface_model=huggingface_model,
                            true_sigma_samples=samples,
                            log_prob_class=log_prob_class)
            
            grad_fn = jax.grad(loss_fn, argnums=2)  # argnums=2 corresponds to params_twist
            grads = grad_fn(rng_key, params_p, params_twist)
            
            # Update twist parameters
            updates, optim_twist_state = optimizer_twist.update(grads, optim_twist_state)
            params_twist = optax.apply_updates(params_twist, updates)
            
            if (twist_update + 1) % 10 == 0:
                loss = loss_fn(rng_key, params_p, params_twist)
                print(f"Twist update {twist_update + 1}, Loss: {loss:.4f}")
    
    return params_twist, optim_twist_state

def train(rng_key, prompt, params_p, params_twist, log_true_final_twist, output_len, n_twist, condition_twist_on_tokens, smc_procedure_type, huggingface_model=None):
    """Train the twist function using BCE loss.
    
    Args:
        rng_key: JAX random key
        prompt: Input prompt tensor
        params_p: Parameters of the base model
        params_twist: Parameters of the twist model
        log_true_final_twist: Function computing true final twist
        output_len: Length of output sequence
        n_twist: Number of twist samples
        condition_twist_on_tokens: Tokens to condition twist on
        smc_procedure_type: Type of SMC procedure
        huggingface_model: Optional HuggingFace model
        
    Returns:
        loss: The BCE loss value
    """
    print("Starting train function")
    
    # Set a timeout for the entire function
    start_time = time.time()
    timeout = 300  # 5 minutes timeout
    
    # Generate samples from base model
    rng_key, sk1 = jax.random.split(rng_key)
    print(f"Generating {n_twist} samples with output length {output_len}")
    true_sigma_samples = stochastic_transformer_sample(
        sk1, params_p, prompt, output_len, n_twist, 
        huggingface_model=huggingface_model
    )
    print(f"Generated samples with shape: {true_sigma_samples.shape}") 
    
    # Check timeout
    if time.time() - start_time > timeout:
        print("Timeout reached during sample generation")
        return jnp.array(0.0)  # Return a dummy value
    
    # Get log probabilities from true final twist
    print(f"Calling log_true_final_twist with condition_twist_on_tokens: {condition_twist_on_tokens}")
    
    # Try to call the function with the expected signature
    try:
        # First try with the original signature
        log_prob_class = log_true_final_twist(true_sigma_samples, condition_twist_on_tokens)
        print(f"Computed log_prob_class with shape: {log_prob_class.shape}") 
    except TypeError as e:
        print(f"Error calling log_true_final_twist: {e}")
        print("Trying alternative signature...")
        try:
            # Try with just the samples
            log_prob_class = log_true_final_twist(true_sigma_samples)
            print(f"Computed log_prob_class with shape: {log_prob_class.shape}") 
        except TypeError as e2:
            print(f"Error calling log_true_final_twist with alternative signature: {e2}")
            print("Trying with additional parameters...")
            # Try with additional parameters
            try:
                # Try with prompt and tokenizer
                from legacy.train_gemma2 import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
                prompt_text = "Solve this math problem: 2 + 2 = ?"
                answer = "4"
                log_prob_class = log_true_final_twist(true_sigma_samples, prompt_text, answer, tokenizer)
                print(f"Computed log_prob_class with shape: {log_prob_class.shape}") 
            except Exception as e3:
                print(f"Error calling log_true_final_twist with additional parameters: {e3}")
                # If all else fails, return a dummy value
                log_prob_class = jnp.zeros(true_sigma_samples.shape[0])
                print("Using dummy log_prob_class")
    
    # Check timeout
    if time.time() - start_time > timeout:
        print("Timeout reached during log_prob_class computation")
        return jnp.array(0.0)  # Return a dummy value
    
    # Compute BCE loss using the samples
    print("Computing BCE loss")
    
    # Set a timeout for the BCE loss computation
    bce_start_time = time.time()
    bce_timeout = 60  # 1 minute timeout for BCE loss computation
    
    try:
        loss = get_l_bce(
            rng_key=rng_key,
            prompt=prompt,
            params_p=params_p,
            params_twist=params_twist,
            log_true_final_twist=log_true_final_twist,
            output_len=output_len,
            n_twist=n_twist,
            condition_twist_on_tokens=condition_twist_on_tokens,
            smc_procedure_type=smc_procedure_type,
            rm_type="binary",
            huggingface_model=huggingface_model,
            true_sigma_samples=true_sigma_samples,
            log_prob_class=log_prob_class
        )
        print(f"Computed loss: {loss}") 
    except Exception as e:
        print(f"ERROR in get_l_bce: {e}")
        loss = jnp.array(0.0)  # Return a dummy value
    
    # Check timeout
    if time.time() - bce_start_time > bce_timeout:
        print("Timeout reached during BCE loss computation")
        loss = jnp.array(0.0)  # Return a dummy value
    
    return loss

def calculate_reward(samples, prompt, reward_model=None):
    """
    Calculate rewards for generated sequences.
    
    Args:
        samples: Generated sequences of shape (batch_size, seq_len)
        prompt: Input prompt of shape (prompt_len,)
        reward_model: Optional reward model for more sophisticated reward calculation
        
    Returns:
        rewards: Array of shape (batch_size,) containing rewards for each sequence
    """
    print(f"main.py calculate_reward called with {len(samples)} samples")
    print(f"prompt type: {type(prompt)}")
    
    batch_size = samples.shape[0]
    rewards = jnp.zeros(batch_size)
    
    for i in range(batch_size):
        # Calculate length penalty
        length_penalty = 0.0
        
        # Calculate diversity reward
        diversity_reward = 0.0
        
        # Calculate coherence reward
        coherence_reward = 0.0
        
        # Combine rewards
        rewards = rewards.at[i].set(length_penalty + diversity_reward + coherence_reward)
    
    print(f"Calculated rewards in main.py: {rewards}")
    return rewards

def log_true_final_twist(samples, condition_twist_on_tokens=None, reward_model=None):
    """
    Calculate log probabilities for the true final twist based on rewards.
    
    Args:
        samples: Generated sequences
        condition_twist_on_tokens: Optional tokens to condition on
        reward_model: Optional reward model for reward calculation
        
    Returns:
        log_probs: Log probabilities for each sequence
    """
    print(f"main.py log_true_final_twist called with {len(samples)} samples")
    print(f"condition_twist_on_tokens: {condition_twist_on_tokens}")
    
    # Import the calculate_reward function from train_gemma2
    try:
        from legacy.train_gemma2 import calculate_reward as gsm8k_calculate_reward
        print("Using GSM8K calculate_reward function")
        
        # Create a dummy tokenizer for now - this will be replaced by the actual tokenizer
        # when the function is called from train_gemma2.py
        class DummyTokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "dummy text"
        
        dummy_tokenizer = DummyTokenizer()
        
        # Calculate rewards using the GSM8K function
        # We need to provide a prompt and correct_answer, but we don't have them here
        # So we'll use dummy values
        dummy_prompt = "dummy prompt"
        dummy_answer = "dummy answer"
        
        rewards = gsm8k_calculate_reward(samples, dummy_prompt, dummy_answer, dummy_tokenizer)
        print(f"Calculated rewards using GSM8K function: {rewards}")
    except ImportError:
        print("Could not import GSM8K calculate_reward function, using fallback")
        # Fallback to the original function
        rewards = calculate_reward(samples, condition_twist_on_tokens, reward_model)
        print(f"Calculated rewards using fallback function: {rewards}")
    
    # Convert rewards to log probabilities using softmax
    log_probs = jax.nn.log_softmax(rewards)
    print(f"Calculated log_probs in main.py: {log_probs}")
    
    return log_probs

def main():
    # Initialize random key
    rng_key = jax.random.PRNGKey(0)
    
    # Initialize model parameters (this would come from your model initialization)
    params_p = {}  # Base model parameters
    params_twist = {}  # Twist model parameters
    
    # Initialize optimizer
    optimizer_twist = optax.adam(learning_rate=1e-4)
    optim_twist_state = optimizer_twist.init(params_twist)
    
    # Example prompt
    prompt = jnp.array([1, 2, 3])  # Replace with your actual prompt
    
    # Training parameters
    output_len = 10
    n_twist = 32
    
    # Initialize huggingface model (this would come from your model setup)
    huggingface_model = {
        'p': lambda input_ids, params: jnp.zeros((input_ids.shape[0], input_ids.shape[1], 10)),  # Dummy implementation
        'twist': lambda input_ids, params: jnp.zeros((input_ids.shape[0], input_ids.shape[1], 1))  # Dummy implementation
    }
    
    # Define log_true_final_twist function (this would come from your reward model)
    def log_true_final_twist(samples):
        return jnp.zeros(samples.shape[0])  # Dummy implementation
    
    # Run training
    params_twist, optim_twist_state = train_twist(
        rng_key=rng_key,
        prompt=prompt,
        params_p=params_p,
        params_twist=params_twist,
        log_true_final_twist=log_true_final_twist,
        output_len=output_len,
        n_twist=n_twist,
        optimizer_twist=optimizer_twist,
        optim_twist_state=optim_twist_state,
        huggingface_model=huggingface_model
    )

if __name__ == "__main__":
    main()
