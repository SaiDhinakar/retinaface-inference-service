import logging
import os
from datetime import datetime

def setup_logger(name: str, log_file: str = "./logs", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with console and optional file output.
    
    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, only console logging.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Default logger for the module
logger = setup_logger(__name__, log_file=f"logs/{datetime.now().strftime('%Y-%m-%d')}.log")