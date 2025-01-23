#!/usr/bin/env python3
"""
Entry point script for the Movie Recommender System.
Run this script to start the system: python run.py
"""

from src.main import main
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting Movie Recommender System...")
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Detailed traceback:")
