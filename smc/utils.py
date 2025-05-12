import logging
import os
import json
import time
from datetime import datetime

def setup_logging(output_dir, debug=False):
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log files
        debug: Whether to enable debug logging
        
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'training_{timestamp}.log')
    
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('twist_smc')
    logger.info(f"Logging initialized. Debug mode: {debug}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def log_dict(logger, data, prefix='', level=logging.INFO):
    """
    Log a dictionary in a readable format.
    
    Args:
        logger: Logger instance
        data: Dictionary to log
        prefix: Prefix for the log message
        level: Logging level
    """
    if not data:
        logger.log(level, f"{prefix}Empty dictionary")
        return
    
    for key, value in data.items():
        if isinstance(value, dict):
            logger.log(level, f"{prefix}{key}:")
            log_dict(logger, value, prefix + '  ', level)
        else:
            logger.log(level, f"{prefix}{key}: {value}")

def log_tensor(logger, tensor, name, level=logging.DEBUG):
    """
    Log tensor information.
    
    Args:
        logger: Logger instance
        tensor: Tensor to log
        name: Name of the tensor
        level: Logging level
    """
    if tensor is None:
        logger.log(level, f"{name}: None")
        return
    
    shape = tensor.shape if hasattr(tensor, 'shape') else 'unknown'
    dtype = tensor.dtype if hasattr(tensor, 'dtype') else 'unknown'
    device = tensor.device if hasattr(tensor, 'device') else 'unknown'
    
    logger.log(level, f"{name}: shape={shape}, dtype={dtype}, device={device}")
    
    # Log a sample of the tensor values if it's small enough
    if hasattr(tensor, 'numel') and tensor.numel() < 100:
        logger.log(level, f"{name} values: {tensor}")

def log_time(logger, start_time, message, level=logging.INFO):
    """
    Log the time elapsed since start_time.
    
    Args:
        logger: Logger instance
        start_time: Start time in seconds
        message: Message to log
        level: Logging level
    """
    elapsed = time.time() - start_time
    logger.log(level, f"{message}: {elapsed:.2f} seconds")
    return time.time()  # Return current time for chaining 


# utils/log_plot.py
import csv, pathlib, matplotlib.pyplot as plt

class TrainLogger:
    """Light-weight metric logger + png plotter."""
    def __init__(self, out_dir: str):
        self.out_dir   = pathlib.Path(out_dir)
        self.csv_path  = self.out_dir / "train_metrics.csv"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.step      = []
        self.metrics   = {}              # name -> list

        # create csv header
        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"])    # we'll append metric names lazily

    def log(self, step: int, **kwargs):
        """
        Example:  logger.log(step=i, loss=loss.item(), logZ=log_z.item())
        """
        if not self.metrics:             # first call → init lists / header
            for k in kwargs: self.metrics[k] = []
            with self.csv_path.open("r+", newline="") as f:
                lines = list(csv.reader(f))
                lines[0].extend(kwargs.keys())
                f.seek(0); f.truncate()
                csv.writer(f).writerows(lines)

        self.step.append(step)
        for k, v in kwargs.items():
            self.metrics[k].append(float(v))   # ensure JSON-serialisable

        # append row to CSV
        with self.csv_path.open("a", newline="") as f:
            row = [step] + [kwargs[k] for k in self.metrics]
            csv.writer(f).writerow(row)

    # call once at the very end
    def plot_all(self):
        for k, vals in self.metrics.items():
            plt.figure()
            plt.plot(self.step, vals)
            plt.xlabel("step")
            plt.ylabel(k)
            plt.title(k)
            plt.tight_layout()
            plt.savefig(self.out_dir / f"{k}.png")
            plt.close()
