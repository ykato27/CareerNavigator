import logging
import sys
from typing import Optional
from pythonjsonlogger import jsonlogger
from skillnote_recommendation.config import config

def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Setup a logger with JSON formatting for cloud environments
    and standard formatting for local development.
    """
    logger = logging.getLogger(name)
    
    # Set default level if not provided
    if level is None:
        level = logging.DEBUG if config.DEBUG else logging.INFO
    logger.setLevel(level)
    
    # Check if handler already exists to avoid duplicates
    if logger.handlers:
        return logger
        
    handler = logging.StreamHandler(sys.stdout)
    
    # Use JSON formatter for better parsing in cloud logs, 
    # or standard formatter for readability in local console
    if not config.DEBUG:
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
