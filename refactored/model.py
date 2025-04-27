import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class TwistModel(nn.Module):
    """
    A twist model that applies a linear transformation to token embeddings.
    This model is used to modify the probability distribution of the base model.
    """
    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head  = nn.Linear(hidden_dim, vocab_size)   # → (B, V)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids : (B, L)   – any length
        returns   : (B, V)   – log-psi for EACH vocab token
        """
        last_ids = input_ids[:, -1]          # (B,)
        h        = self.embed(last_ids)      # (B, hidden)
        return self.head(h) 

class ModelWrapper:
    def __init__(self, model_name, device=None, logger=None):
        self.logger = logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if logger:
            logger.info(f"Loading model {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.twist_model = TwistModel(self.tokenizer.vocab_size).to(self.device)
        
        if logger:
            logger.info(f"Model loaded with vocab size: {self.tokenizer.vocab_size}")
    
    def get_base_model_logits_for_sequence(self, sequence):
        """Get logits from base model for a sequence of tokens."""
        input_ids = torch.tensor(sequence, device=self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, return_dict=True)
            return outputs.logits[0, -1, :]
    
    def get_twist_values_for_particles(self, particles):
        """Get twist values for a list of particles."""
        max_len = max(len(p) for p in particles)
        input_ids = torch.zeros(len(particles), max_len, dtype=torch.long, device=self.device)
        for i, p in enumerate(particles):
            input_ids[i, :len(p)] = torch.tensor(p, device=self.device)
        
        # twist_values = self.twist_model(input_ids)
        # psi = twist_values[:, -1, 0].unsqueeze(-1).expand(-1, self.tokenizer.vocab_size)
        
        # log ψ  (B, vocab)
        log_psi = self.twist_model(input_ids)             
        psi     = torch.exp(log_psi)
        return psi  
    
    def save_state(self, path):
        """Save model state."""
        torch.save({
            'base_model': self.base_model.state_dict(),
            'twist_model': self.twist_model.state_dict()
        }, path)
    
    def load_state(self, path):
        """Load model state."""
        state = torch.load(path)
        self.base_model.load_state_dict(state['base_model'])
        self.twist_model.load_state_dict(state['twist_model'])

class HuggingFaceModelWrapper:
    """
    A wrapper for HuggingFace models that provides a consistent interface
    for both the base model and the twist model.
    """
    def __init__(self, model_name, logger=None):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name of the HuggingFace model to load
            logger: Optional logger instance
        """
        if logger:
            logger.info(f"Initializing HuggingFaceModelWrapper with model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if logger:
            logger.info(f"Loaded tokenizer with vocab size: {len(self.tokenizer)}")
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if logger:
            logger.info(f"Loaded model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if logger:
            logger.info(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        
        # Initialize the twist model
        self.twist_model = TwistModel(self.tokenizer.vocab_size)
        self.twist_model = self.twist_model.to(self.device)
        
    def get_base_model_logits(self, input_ids, logger=None):
        """
        Get logits from the base model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            logger: Optional logger instance
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        if logger:
            logger.debug(f"Getting base model logits for input shape: {input_ids.shape}")
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, return_dict=True)
            logits = outputs.logits
        
        if logger:
            logger.debug(f"Base model logits shape: {logits.shape}")
        
        return logits
    
    def get_twist_values(self, input_ids, logger=None):
        """
        Get twist values from the twist model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            logger: Optional logger instance
            
        Returns:
            Twist values of shape (batch_size, seq_len, hidden_size)
        """
        if logger:
            logger.debug(f"Getting twist values for input shape: {input_ids.shape}")
        
        twist_values = self.twist_model(input_ids)
        
        if logger:
            logger.debug(f"Twist values shape: {twist_values.shape}")
        
        return twist_values
    
    def sample_from_base_model(self, prompt, output_length, num_samples=1, temperature=1.0, logger=None):
        """
        Generate samples from the base model (p).
        
        Args:
            prompt: Input prompt text
            output_length: Length of the output sequence
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            logger: Optional logger instance
            
        Returns:
            Generated samples of shape (num_samples, prompt_length + output_length)
        """
        if logger:
            logger.debug(f"Sampling {num_samples} sequences from base model with length {output_length}")
            logger.debug(f"Prompt: {prompt}")
            logger.debug(f"Temperature: {temperature}")
        
        start_time = time.time()
        
        # Tokenize the prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_ids.shape[1]
        
        if logger:
            logger.debug(f"Prompt tokenized to {prompt_length} tokens")
        
        # Create batch of prompts
        batch_prompt = prompt_ids.repeat(num_samples, 1)
        
        # Initialize output tensor
        output = torch.zeros(num_samples, output_length, dtype=torch.long, device=self.device)
        
        # Generate samples
        for t in range(output_length):
            if logger and t % 10 == 0:
                logger.debug(f"Generating token {t+1}/{output_length}")
            
            # Get logits for the current sequence
            full_seq = torch.cat((batch_prompt, output[:, :t]), dim=1)
            logits = self.get_base_model_logits(full_seq, logger)
            
            # Get logits for the next token
            next_token_logits = logits[:, prompt_length + t - 1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update output
            output[:, t] = next_token
        
        # Combine prompt and output
        full_seq = torch.cat((batch_prompt, output), dim=1)
        
        if logger:
            logger.debug(f"Base model sampling completed in {time.time() - start_time:.2f} seconds")
            logger.debug(f"Generated samples shape: {full_seq.shape}")
        
        return full_seq
    
    def sample_from_proposal_model(self, prompt, output_length, num_samples=1, temperature=1.0, logger=None):
        """
        Generate samples from the proposal model (q = p * twist).
        
        Args:
            prompt: Input prompt text
            output_length: Length of the output sequence
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            logger: Optional logger instance
            
        Returns:
            Generated samples of shape (num_samples, prompt_length + output_length)
        """
        if logger:
            logger.debug(f"Sampling {num_samples} sequences from proposal model with length {output_length}")
            logger.debug(f"Prompt: {prompt}")
            logger.debug(f"Temperature: {temperature}")
        
        start_time = time.time()
        
        # Tokenize the prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_ids.shape[1]
        
        if logger:
            logger.debug(f"Prompt tokenized to {prompt_length} tokens")
        
        # Create batch of prompts
        batch_prompt = prompt_ids.repeat(num_samples, 1)
        
        # Initialize output tensor
        output = torch.zeros(num_samples, output_length, dtype=torch.long, device=self.device)
        
        # Generate samples
        for t in range(output_length):
            if logger and t % 10 == 0:
                logger.debug(f"Generating token {t+1}/{output_length}")
            
            # Get logits for the current sequence
            full_seq = torch.cat((batch_prompt, output[:, :t]), dim=1)
            base_logits = self.get_base_model_logits(full_seq, logger)
            
            # Get twist values for the current sequence
            twist_values = self.get_twist_values(full_seq, logger)
            
            # Apply twist to the logits
            # The twist values are of shape (batch_size, seq_len, hidden_size)
            # We need to broadcast them to match the logits shape
            twist_broadcasted = twist_values[:, :, 0].unsqueeze(-1).expand(-1, -1, base_logits.shape[-1])
            
            # Apply twist to the logits
            twisted_logits = base_logits + twist_broadcasted
            
            # Get logits for the next token
            next_token_logits = twisted_logits[:, prompt_length + t - 1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Update output
            output[:, t] = next_token
        
        # Combine prompt and output
        full_seq = torch.cat((batch_prompt, output), dim=1)
        
        if logger:
            logger.debug(f"Proposal model sampling completed in {time.time() - start_time:.2f} seconds")
            logger.debug(f"Generated samples shape: {full_seq.shape}")
        
        return full_seq
    
    def sample(self, prompt, output_length, num_samples=1, temperature=1.0, use_proposal=True, logger=None):
        """
        Generate samples from either the base model or the proposal model.
        
        Args:
            prompt: Input prompt text
            output_length: Length of the output sequence
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            use_proposal: Whether to use the proposal model (q) or base model (p)
            logger: Optional logger instance
            
        Returns:
            Generated samples of shape (num_samples, prompt_length + output_length)
        """
        if use_proposal:
            return self.sample_from_proposal_model(prompt, output_length, num_samples, temperature, logger)
        else:
            return self.sample_from_base_model(prompt, output_length, num_samples, temperature, logger)
    
    def get_twist_parameters(self, logger=None):
        """
        Get the parameters of the twist model.
        
        Args:
            logger: Optional logger instance
            
        Returns:
            Dictionary of twist model parameters
        """
        if logger:
            logger.debug("Getting twist parameters")
        
        params = {
            'transformer': {
                'w': self.twist_model.linear.weight.data,
                'b': self.twist_model.linear.bias.data
            }
        }
        
        if logger:
            logger.debug(f"Twist parameters: w shape={params['transformer']['w'].shape}, b shape={params['transformer']['b'].shape}")
        
        return params
    
    def set_twist_parameters(self, params, logger=None):
        """
        Set the parameters of the twist model.
        
        Args:
            params: Dictionary of twist model parameters
            logger: Optional logger instance
        """
        if logger:
            logger.debug("Setting twist parameters")
        
        if 'transformer' in params:
            self.twist_model.linear.weight.data = params['transformer']['w']
            self.twist_model.linear.bias.data = params['transformer']['b']
            
            if logger:
                logger.debug(f"Set twist parameters: w shape={params['transformer']['w'].shape}, b shape={params['transformer']['b'].shape}")
        else:
            if logger:
                logger.warning("No 'transformer' key in params, twist parameters not set")

    def get_base_model_logits_for_sequence(self, sequence, logger=None):
        """
        Get logits from the base model for a single sequence.
        
        Args:
            sequence: List of token indices
            logger: Optional logger instance
            
        Returns:
            Logits of shape (vocab_size,)
        """
        if logger:
            logger.debug(f"Getting base model logits for sequence of length {len(sequence)}")
        
        # Convert sequence to tensor
        input_ids = torch.tensor(sequence, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, return_dict=True)
            logits = outputs.logits[0, -1, :]  # Get logits for the last token
        
        if logger:
            logger.debug(f"Base model logits shape: {logits.shape}")
        
        return logits
    
    def get_twist_values_for_particles(self, particles, logger=None):
        """
        Get twist values for a list of particles.
        
        Args:
            particles: List of sequences (each a list of token indices)
            logger: Optional logger instance
            
        Returns:
            Twist values of shape (num_particles, vocab_size)
        """
        if logger:
            logger.debug(f"Getting twist values for {len(particles)} particles")
        
        # Convert particles to tensor
        max_len = max(len(p) for p in particles)
        input_ids = torch.zeros(len(particles), max_len, dtype=torch.long, device=self.device)
        for i, p in enumerate(particles):
            input_ids[i, :len(p)] = torch.tensor(p, device=self.device)
        
        # Get twist values
        twist_values = self.twist_model(input_ids)
        
        # Expand twist values to match vocab size
        twist_values = twist_values[:, -1, 0].unsqueeze(-1).expand(-1, self.tokenizer.vocab_size)
        
        if logger:
            logger.debug(f"Twist values shape: {twist_values.shape}")
        
        return twist_values 