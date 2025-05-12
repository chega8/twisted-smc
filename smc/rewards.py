import torch
import re
import numpy as np
import time

from detoxify import Detoxify

tox_model = Detoxify('unbiased') 

def reward_harm(text: str) -> float:
    tox = tox_model.predict([text])['toxicity'][0]    # 0..1
    L   = len(text.split())
    length_score = min(1.0, L / 10) * min(1.0, 128 / max(1, L))
    # return 0.5 * (1 - tox) + 0.5 * length_score
    # return 0.5 * (tox) + 0.5 * length_score
    return tox

def format_reward_func_qa(completions, **kwargs):
    pattern = r"\n#### The final answer is \d+"    
    completion_contents = [completion for completion in completions]    
    matches = [re.search(pattern, content) for content in completion_contents]
    return [0.5 if match else 0.0 for match in matches]

def format_reward(response, logger=None):
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        if logger:
            logger.debug(f"Found \\boxed")
        return 0.3
    if logger:
        logger.warning("No \\boxed found in response")
    return -0.3

def extract_answer_from_response(response, logger=None):
    """
    Extract the boxed answer from the model's response.
    
    Args:
        response: The model's response text
        logger: Optional logger instance
        
    Returns:
        The extracted answer or None if no answer is found
    """
    if logger:
        logger.debug(f"Extracting answer from response: {response[:100]}...")
    
    # Look for \boxed{...} pattern
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        if logger:
            logger.debug(f"Found boxed answer: {answer}")
        return answer
    
    # If no boxed answer, try to find the last number in the response
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', response)
    if numbers:
        answer = numbers[-1]
        if logger:
            logger.debug(f"Found number answer: {answer}")
        return answer
    
    if logger:
        logger.warning("No answer found in response")
    return None

def is_correct_answer(predicted, actual, logger=None):
    """
    Check if the predicted answer is correct.
    
    Args:
        predicted: The predicted answer
        actual: The actual answer
        logger: Optional logger instance
        
    Returns:
        True if the answers match, False otherwise
    """
    if logger:
        logger.debug(f"Comparing predicted answer '{predicted}' with actual answer '{actual}'")
    
    try:
        # Convert both to float for comparison
        pred_float = float(predicted)
        actual_float = float(actual)
        
        # Check if they're close enough
        is_correct = abs(pred_float - actual_float) < 1e-6
        if logger:
            logger.debug(f"Numerical comparison: {pred_float} vs {actual_float}, is_correct={is_correct}")
        return is_correct
    except:
        # If conversion fails, do string comparison
        is_correct = predicted.strip() == actual.strip()
        if logger:
            logger.debug(f"String comparison: '{predicted.strip()}' vs '{actual.strip()}', is_correct={is_correct}")
        return is_correct
    
def parse_correct_answer(correct_answer_txt: str):
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', correct_answer_txt)
    if numbers:
        answer = numbers[-1]
        return answer
    
def calculate_exact_math_reward(sample_text, correct_answer, logger):
    correct_answer_val = parse_correct_answer(correct_answer)
    extracted_answer = extract_answer_from_response(sample_text, logger)
    # Calculate exact match reward
    if extracted_answer is not None:
        if is_correct_answer(extracted_answer, correct_answer_val, logger):
            if logger:
                logger.debug("Exact match")
            return 0.3
        else:
            if logger:
                logger.debug("Exact match reward: 0.0")
    else:
        if logger:
            logger.debug("Exact match reward: 0.0 (no answer extracted)")
    
def calculate_partial_match_reward(sample_text, correct_answer, tokenizer, logger):
    if tokenizer is not None:
        # Tokenize sample and answer
        sample_tokens = tokenizer.encode(sample_text)
        answer_tokens = tokenizer.encode(correct_answer)
        
        if len(sample_tokens) > 0 and len(answer_tokens) > 0:
            common_tokens = set(sample_tokens) & set(answer_tokens)
            if len(common_tokens) > 0:
                val = len(common_tokens) / max(len(sample_tokens), len(answer_tokens))
                if logger:
                    logger.debug(f"Partial match reward: {val}")
                return val
            else:
                if logger:
                    logger.debug("Partial match reward: 0.0 (no common tokens)")
        else:
            if logger:
                logger.debug("Partial match reward: 0.0 (empty token lists)")
    return 0.0
                
def calculate_len_reward(sample_text, correct_answer, tokenizer, logger):
    if tokenizer is not None:
        sample_tokens = tokenizer.encode(sample_text)
        answer_tokens = tokenizer.encode(correct_answer)
        
        if len(sample_tokens) > 0 and len(answer_tokens) > 0:
            length_ratio = min(len(sample_tokens), len(answer_tokens)) / max(len(sample_tokens), len(answer_tokens))
            if logger:
                logger.debug(f"Length penalty: {length_ratio}")
                return length_ratio
        else:
            if logger:
                logger.debug("Length penalty: 0.0 (empty token lists)")
    return 0.0

def calculate_reward(sample_text, correct_answer, tokenizer=None, logger=None):
    """
    Calculate reward for a given sample and correct answer.
    
    Args:
        sample_text: The model's response text
        correct_answer: The correct answer
        tokenizer: Optional tokenizer for token-based rewards
        logger: Optional logger instance
        
    Returns:
        Reward value between 0 and 1
    """
    if logger:
        logger.debug(f"Calculating reward for sample: {sample_text}")
        logger.debug(f"Correct answer: {correct_answer}")
    
    # Initialize reward components
    reward_components = {
        # 'exact_match': 0.0,
        # 'format_match': 0.0,
        'partial_match': 0.0,
        'length_penalty': 0.0,
        'tox': 0.0
    }
    
    try:
        if 'exact_match' in reward_components.keys():
            reward_components['exact_match'] = calculate_exact_math_reward(sample_text, correct_answer, logger)
                
        if 'format_match' in reward_components.keys():
            format_reward_val = format_reward(sample_text)
            reward_components['format_match'] = format_reward_val
        
        # Calculate partial match reward if tokenizer is provided
        if 'partial_match' in reward_components.keys():
            reward_components['partial_match'] = calculate_partial_match_reward(sample_text, correct_answer, tokenizer, logger)
        
        if 'tox' in reward_components.keys():
            reward_components['tox'] = reward_harm(sample_text)
            
        # Calculate length penalty if tokenizer is provided
        reward_components['length_penalty'] = calculate_len_reward(sample_text, correct_answer, tokenizer, logger)
        
        # Calculate total reward
        reward = sum(reward_components.values()) / len(reward_components)
        if logger:
            logger.debug(f"Total reward: {reward}")
            logger.debug(f"Reward components: {reward_components}")

    
    except Exception as e:
        if logger:
            logger.error(f"ERROR in calculate_reward: {e}")
        reward = 0.0
    
    return reward

def calculate_rewards_for_samples(samples, prompt, correct_answer, tokenizer, logger=None):
    """
    Calculate rewards for a batch of samples.
    
    Args:
        samples: Generated samples of shape (batch_size, seq_len)
        prompt: Input prompt text
        correct_answer: The correct answer
        tokenizer: Tokenizer for decoding samples
        logger: Optional logger instance
        
    Returns:
        Rewards of shape (batch_size,)
    """
    if logger:
        logger.debug(f"Calculating rewards for {samples.shape[0]} samples")
    
    batch_size = samples.shape[0]
    rewards = torch.zeros(batch_size, device=samples.device)
    
    for i in range(batch_size):
        # Decode the sample
        sample_text = tokenizer.decode(samples[i], skip_special_tokens=True)
        
        if logger:
            logger.debug(f"Sample {i+1}/{batch_size}: {sample_text[:100]}...")
        
        # Calculate reward for this sample
        reward = calculate_reward(sample_text, correct_answer, tokenizer, logger)
        
        # Set the reward
        rewards[i] = reward
        
        if logger:
            logger.debug(f"Sample {i+1} reward: {reward}")
    
    if logger:
        logger.debug(f"Calculated rewards: {rewards}")
    
    return rewards

# def log_true_final_twist(samples, prompt, correct_answer, tokenizer, logger=None):
#     """
#     Calculate log probabilities for the true final twist based on rewards.
    
#     Args:
#         samples: Generated samples of shape (batch_size, seq_len)
#         prompt: Input prompt text
#         correct_answer: The correct answer
#         tokenizer: Tokenizer for decoding samples
#         logger: Optional logger instance
        
#     Returns:
#         Log probabilities of shape (batch_size,)
#     """
#     if logger:
#         logger.debug("Calculating log_true_final_twist")
#         logger.debug(f"Prompt: {prompt}")
#         logger.debug(f"Correct answer: {correct_answer}")
    
#     start_time = time.time()
    
#     # Calculate rewards for each sample
#     rewards = calculate_rewards_for_samples(samples, prompt, correct_answer, tokenizer, logger)
    
#     if logger:
#         logger.debug(f"Calculated rewards in {time.time() - start_time:.2f} seconds")
    
#     # Convert rewards to log probabilities using softmax
#     log_probs = torch.log_softmax(rewards, dim=0)
    
#     if logger:
#         logger.debug(f"Log probabilities: {log_probs}")
    
#     return log_probs 

