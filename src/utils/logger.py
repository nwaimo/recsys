import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with consistent formatting and both file and console handlers.
    
    Args:
        name: The name of the logger (typically __name__).
        log_file: Optional specific log file path. If None, uses default location.
        level: The logging level to use.
    
    Returns:
        A configured logger instance.
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # If no specific log file is provided, use a default name based on the module
    if log_file is None:
        log_file = os.path.join(logs_dir, f'recommender_{datetime.now().strftime("%Y%m%d")}.log')

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create and configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers if any
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
