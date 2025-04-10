# Twisted SMC for Math Problem Solving

This repository contains a refactored implementation of the Twisted Sequential Monte Carlo (SMC) approach for solving math problems using language models. The codebase has been completely rewritten to use PyTorch instead of JAX, with improved code organization and documentation.

## Project Structure

```
refactored/
├── model.py          # Model definitions and wrappers
├── rewards.py        # Reward calculation functions
├── training.py       # Training loop and utilities
├── main.py          # Entry point and argument parsing
└── README.md        # This file
```

## Key Components

### Model (`model.py`)
- `TwistModel`: A PyTorch module implementing the twist function
- `HuggingFaceModelWrapper`: Wrapper for HuggingFace models with twist functionality

### Rewards (`rewards.py`)
- `calculate_reward`: Computes rewards for math problem solutions
- `extract_answer_from_response`: Extracts numerical answers from model responses

### Training (`training.py`)
- `train_twist_for_math_problems`: Main training loop
- Various utility functions for data loading and loss computation

## Usage

To train the model, run:

```bash
python main.py \
    --model_name google/gemma-2b-it \
    --num_examples 100 \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --output_dir math_twist_model \
    --output_length 256 \
    --num_twist_samples 8
```

### Arguments

- `--model_name`: Name of the HuggingFace model to use
- `--num_examples`: Number of training examples to use
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimization
- `--output_dir`: Directory to save model checkpoints
- `--seed`: Random seed for reproducibility
- `--split`: Dataset split to use ('train' or 'test')
- `--output_length`: Maximum output sequence length
- `--num_twist_samples`: Number of twist samples per example
- `--twist_updates_per_example`: Number of twist updates per example
- `--save_every`: Save checkpoint every N batches

## Implementation Details

The implementation uses a twist function to guide the sampling process towards solutions that satisfy the given math problem. The twist function is trained to maximize the probability of generating correct solutions.

Key features:
- PyTorch-based implementation for better compatibility and debugging
- Modular design with clear separation of concerns
- Comprehensive error handling and debugging output
- Configurable hyperparameters via command-line arguments
- Automatic checkpointing and experiment tracking

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- NumPy
- tqdm 