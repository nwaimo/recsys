from queue import Queue
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from .utils.logger import setup_logger
from datetime import datetime
from typing import Dict, Any, List

class BatchProcessor:
    """
    Processes batches of ratings data, adding them to a queue and processing them in the background.
    Improves efficiency by handling ratings in chunks and validating data.
    """

    def __init__(self):
        """
        Initializes the BatchProcessor without direct dependencies.
        """
        self.recommender = None
        self.batch_size = 100
        self.batch_queue = Queue()
        self.processing_thread = None
        self.is_processing = False
        self.processed_batches = 0
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers as needed
        self.logger = setup_logger(__name__)
        self.lock = threading.Lock()  # Lock for thread-safe operations

    def set_recommender(self, recommender):
        """Set the recommender system via dependency injection"""
        self.recommender = recommender

    def validate_rating(self, rating: Dict[str, Any]) -> bool:
        """
        Validates a single rating entry.

        Args:
            rating: A dictionary representing a rating with 'user_id', 'movie_id', 'rating', and 'timestamp'.

        Returns:
            True if the rating is valid, False otherwise.
        """
        required_fields = {'user_id', 'movie_id', 'rating', 'timestamp'}
        if not all(field in rating for field in required_fields):
            self.logger.error(f"Missing required fields: {required_fields - rating.keys()}")
            return False

        try:
            # Validate data types and ranges
            if not isinstance(rating['user_id'], (int, np.int64)):
                self.logger.error(f"Invalid user_id type: {type(rating['user_id'])} for user_id {rating['user_id']}")
                return False

            if not isinstance(rating['movie_id'], (int, np.int64)):
                self.logger.error(f"Invalid movie_id type: {type(rating['movie_id'])} for movie_id {rating['movie_id']}")
                return False

            if not isinstance(rating['rating'], (int, float, np.number)):
                self.logger.error(f"Invalid rating type: {type(rating['rating'])} for rating {rating['rating']}")
                return False

            if not (0.5 <= float(rating['rating']) <= 5.0):
                self.logger.error(f"Rating out of range: {rating['rating']}")
                return False

            # Validate timestamp (optional, depending on your requirements)
            if not isinstance(rating['timestamp'], (int, np.int64)):
                self.logger.error(f"Invalid timestamp type: {type(rating['timestamp'])}")
                return False
            
            # Check if user_id and movie_id exist in the recommender system (optional)

            return True

        except (ValueError, TypeError) as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def add_to_batch(self, rating_data: Dict[str, Any]) -> bool:
        """
        Adds a new rating to the batch queue.

        Args:
            rating_data: A dictionary representing a rating.

        Returns:
            True if the rating was added successfully, False otherwise.
        """
        with self.lock:
            if not self.validate_rating(rating_data):
                return False

            self.batch_queue.put(rating_data)
            self.logger.debug(f"Added rating to queue: {rating_data}")

            if self.batch_queue.qsize() >= self.batch_size and not self.is_processing:
                self.process_batch()

            return True

    def process_batch(self) -> None:
        """
        Processes a batch of ratings from the queue in a separate thread.
        """
        with self.lock:
            if self.is_processing:
                self.logger.debug("Batch processing already in progress.")
                return

            self.is_processing = True
            self.logger.info("Starting batch processing...")

        # Start processing in a separate thread to avoid blocking
        self.processing_thread = threading.Thread(target=self._process_batch_thread)
        self.processing_thread.start()

    def _process_batch_thread(self) -> None:
        """
        Internal method to handle batch processing in a thread.
        """
        batch = self._get_batch_from_queue()

        if batch:
            try:
                # Split the batch into chunks and process them in parallel
                futures = []
                for rating_chunk in np.array_split(batch, self.executor._max_workers):
                    futures.append(self.executor.submit(self._process_ratings_chunk, rating_chunk))

                # Wait for all chunks to be processed
                for future in futures:
                    future.result()

                with self.lock:
                    self.processed_batches += 1
                self.logger.info(f"Processed batch {self.processed_batches} with {len(batch)} ratings")

            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
            finally:
                with self.lock:
                    self.is_processing = False

    def _get_batch_from_queue(self) -> List[Dict[str, Any]]:
        """
        Retrieves a batch of ratings from the queue.

        Returns:
            A list of rating dictionaries.
        """
        batch = []
        with self.lock:
            while not self.batch_queue.empty() and len(batch) < self.batch_size:
                batch.append(self.batch_queue.get())
        return batch
    
    def _process_ratings_chunk(self, ratings_chunk: List[Dict[str, Any]]) -> bool:
        """
        Processes a chunk of ratings.

        Args:
            ratings_chunk: A list of rating dictionaries.
        
        Returns:
            True if the chunk was processed successfully.
        """
        try:
            new_ratings_df = pd.DataFrame(ratings_chunk)
            self.recommender.update_feature_matrices(new_ratings_df)
            self.logger.debug(f"Processed ratings chunk: {ratings_chunk}")
            return True
        except Exception as e:
            self.logger.error(f"Error processing ratings chunk: {e}")
            return False

    def get_batch_status(self) -> Dict[str, Any]:
        """
        Gets the current status of batch processing.

        Returns:
            A dictionary containing information about the batch processor's state.
        """
        with self.lock:
            return {
                'queue_size': self.batch_queue.qsize(),
                'is_processing': self.is_processing,
                'processed_batches': self.processed_batches,
                'batch_size': self.batch_size,
                'last_check': datetime.now().isoformat()
            }