#!/usr/bin/env python3
"""
Entry point script for the Movie Recommender System.
Run this script to start the system: python run.py
"""

from src.main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check the logs for more details.")
