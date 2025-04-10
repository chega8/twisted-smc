import argparse
import logging
import os
import time
from training import train_twist_for_math_problems
from utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Train twist function for math problems")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the base model")
    parser.add_argument("--num_examples", type=int, default=1, help="Number of examples to use")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--output_length", type=int, default=30, help="Length of output sequences")
    parser.add_argument("--num_twist_samples", type=int, default=2, help="Number of twist samples")
    parser.add_argument("--twist_updates_per_example", type=int, default=1, help="Number of twist updates per example")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N examples")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.debug)
    
    # Train the model
    train_twist_for_math_problems(
        model_name=args.model_name,
        num_examples=args.num_examples,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        seed=args.seed,
        split=args.split,
        output_length=args.output_length,
        num_twist_samples=args.num_twist_samples,
        twist_updates_per_example=args.twist_updates_per_example,
        save_every=args.save_every,
        logger=logger
    )

if __name__ == "__main__":
    main() 