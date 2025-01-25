import queue # Import queue explicitly for exception handling
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from .utils.logger import setup_logger
from datetime import datetime
from typing import Dict, Any, List

class BatchProcessor:
    """
    Processes batches of ratings data, adding them to a queue and processing them in the background.
    Improves efficiency by handling ratings in chunks and validating data.
    """

    DEFAULT_BATCH_SIZE = 100
    DEFAULT_CHECK_INTERVAL = 1.0  # seconds
    DEFAULT_MAX_WORKERS = 4
    QUEUE_GET_TIMEOUT = 5 # seconds - timeout to wait for items when fetching batch from queue

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE, check_interval: float = DEFAULT_CHECK_INTERVAL, max_workers: int = DEFAULT_MAX_WORKERS):
        """
        Initializes the BatchProcessor without direct dependencies.

        Args:
            batch_size: The number of ratings to process in each batch.
            check_interval: The interval (in seconds) to check the queue for processing.
            max_workers: The maximum number of threads to use for processing batches.
        """
        self.recommender = None
        self.batch_size = batch_size
        self.batch_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        self.processed_batches = 0
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="BatchProcessorWorker") # Naming threads for easier debugging
        self.logger = setup_logger(__name__)
        self.lock = threading.Lock()  # Lock for thread-safe operations
        self.check_interval = check_interval
        self.checker_thread = None
        self.should_run = True
        self._start_checker()

    def _start_checker(self):
        """Starts the background thread that checks for batches to process."""
        if self.checker_thread is None or not self.checker_thread.is_alive(): # Prevent starting multiple checkers
            self.checker_thread = threading.Thread(target=self._check_queue_periodically, daemon=True, name="BatchQueueChecker") # Naming thread
            self.checker_thread.start()
            self.logger.info("Batch checker thread started.")
        else:
            self.logger.warning("Batch checker thread already running, avoiding restart.")

    def _check_queue_periodically(self):
        """Periodically checks the queue and processes batches when ready."""
        while self.should_run:
            try:
                current_size = self.batch_queue.qsize()
                if current_size >= self.batch_size and not self.is_processing:
                    self.process_batch()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in batch checker thread: {e}", exc_info=True)

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
            missing_fields = required_fields - rating.keys()
            self.logger.error(f"Validation failed: Missing fields {missing_fields}. Rating data: {rating}")
            return False

        try:
            # Validate data types and ranges - more specific error messages
            if not isinstance(rating['user_id'], (int, np.int64)):
                self.logger.error(f"Validation failed: user_id must be int, got {type(rating['user_id']).__name__}. Rating data: {rating}")
                return False

            if not isinstance(rating['movie_id'], (int, np.int64)):
                self.logger.error(f"Validation failed: movie_id must be int, got {type(rating['movie_id']).__name__}. Rating data: {rating}")
                return False

            if not isinstance(rating['rating'], (int, float, np.number)):
                self.logger.error(f"Validation failed: rating must be numeric, got {type(rating['rating']).__name__}. Rating data: {rating}")
                return False

            if not (0.5 <= float(rating['rating']) <= 5.0):
                self.logger.error(f"Validation failed: rating {rating['rating']} out of range [0.5, 5.0]. Rating data: {rating}")
                return False

            if not isinstance(rating['timestamp'], (int, np.int64)):
                self.logger.error(f"Validation failed: timestamp must be int, got {type(rating['timestamp']).__name__}. Rating data: {rating}")
                return False

            # Additional validation rules can be added here

            return True

        except (ValueError, TypeError) as e:
            self.logger.error(f"Validation error: {e}. Rating data: {rating}", exc_info=True)
            return False

    def add_to_batch(self, rating_data: Dict[str, Any]) -> bool:
        """
        Adds a new rating to the batch queue.

        Args:
            rating_data: A dictionary representing a rating.

        Returns:
            True if the rating was added successfully, False otherwise.
        """
        if not self.validate_rating(rating_data):
            return False  # Validation failure already logged in validate_rating method

        try:
            self.batch_queue.put(rating_data, block=False) # Non-blocking put to avoid potential delays
            self.logger.debug(f"Rating added to queue. Queue size: {self.batch_queue.qsize()}. Rating data: {rating_data}") # Log queue size
            return True

        except queue.Full: # Should not happen with an unbounded queue, but for robustness
            self.logger.error(f"Batch queue is full (unexpected). Rating data: {rating_data}")
            return False
        except Exception as e:
            self.logger.error(f"Error adding to batch queue: {e}. Rating data: {rating_data}", exc_info=True)
            return False

    def process_batch(self) -> None:
        """
        Processes a batch of ratings from the queue.
        """
        if not self.lock.acquire(blocking=False):
            self.logger.debug("Batch processing already active, skipping trigger.")
            return

        try:
            self.is_processing = True
            self.logger.info("Starting batch processing. Triggered by queue size.")

            # Start processing in a separate thread
            processing_thread = threading.Thread(target=self._process_batch_thread, name="BatchProcessingThread") # Naming thread
            processing_thread.start()

        except Exception as e:
            self.logger.error(f"Error initiating batch processing: {e}", exc_info=True)
            self.is_processing = False
        finally:
            self.lock.release()

    def _process_batch_thread(self) -> None:
        """
        Internal method to handle batch processing in a thread.
        """
        batch_start_time = time.time()  # Time when batch processing started
        batch_size = 0
        processed_chunk_count = 0
        failed_chunk_count = 0

        try:
            batch = self._get_batch_from_queue()
            batch_size = len(batch)
            if not batch:
                self.logger.info("No ratings retrieved from queue for processing in this batch.")
                return

            self.logger.debug(f"Processing batch of {len(batch)} ratings in thread.")

            # Process in chunks using ThreadPoolExecutor
            futures = []
            chunks = np.array_split(batch, self.executor._max_workers) # Split batch into chunks *before* submitting
            for chunk_index, rating_chunk in enumerate(chunks):
                if rating_chunk.size > 0:
                    future = self.executor.submit(self._process_ratings_chunk, rating_chunk.tolist(), chunk_index) # Pass chunk index for logging
                    futures.append(future)

            # Wait for all chunks to complete and check for exceptions
            for future in futures:
                try:
                    future.result() # Will raise exception if _process_ratings_chunk failed
                    processed_chunk_count += 1
                except Exception as e_chunk:
                    self.logger.error(f"Error processing a chunk in batch: {e_chunk}", exc_info=True) # Log chunk processing error
                    failed_chunk_count += 1


            processing_time = time.time() - batch_start_time
            self.processed_batches += 1
            self.logger.info(f"Processed batch {self.processed_batches} of {batch_size} ratings in {processing_time:.4f} seconds. Processed chunks: {processed_chunk_count}, Failed chunks: {failed_chunk_count}.") # Detailed log

        except Exception as e_batch_thread:
            self.logger.error(f"Error in batch processing thread: {e_batch_thread}", exc_info=True)
        finally:
            self.is_processing = False

    def _get_batch_from_queue(self) -> List[Dict[str, Any]]:
        """
        Retrieves a batch of ratings from the queue.

        Returns:
            A list of rating dictionaries.
        """
        batch = []
        batch_start_time = time.time()  # Start time for batch retrieval
        queue_check_interval = 0.1  # Check queue every 100ms while waiting for batch
        start_wait_time = time.time()
        wait_timeout = BatchProcessor.QUEUE_GET_TIMEOUT  # Maximum wait time to fill batch - using class constant

        with self.lock:  # Lock for thread-safe queue operations
            while len(batch) < self.batch_size and (time.time() - start_wait_time) < wait_timeout:
                try:
                    rating = self.batch_queue.get(timeout=queue_check_interval)  # Non-blocking get with timeout
                    batch.append(rating)
                except queue.Empty:  # queue.Empty is raised if timeout occurs
                    pass  # Continue checking if queue is still not full or timeout is not reached
                except Exception as e_get_queue: # Catch any other queue related exceptions
                    self.logger.error(f"Error getting item from batch queue: {e_get_queue}", exc_info=True)
                    break # Exit loop in case of queue error


        queue_get_time = time.time() - start_wait_time  # Time spent retrieving batch from queue
        retrieved_count = len(batch)
        queue_size_after_get = self.batch_queue.qsize()
        self.logger.debug(f"Retrieved batch of {retrieved_count} ratings from queue in {queue_get_time:.4f} seconds. Queue size after retrieval: {queue_size_after_get}.") # More detailed log
        return batch

    def _process_ratings_chunk(self, ratings_chunk: List[Dict[str, Any]], chunk_index: int) -> bool:
        """
        Processes a chunk of ratings.

        Args:
            ratings_chunk: A list of rating dictionaries.
            chunk_index: Index of the chunk in the batch, for logging purposes.

        Returns:
            True if the chunk was processed successfully.
        """
        chunk_size = len(ratings_chunk)
        if chunk_size == 0:
            self.logger.debug(f"Chunk {chunk_index} is empty, skipping processing.")
            return True # Empty chunk is considered successfully processed

        try:
            new_ratings_df = pd.DataFrame(ratings_chunk)
            self.recommender.update_feature_matrices_with_new_data()
            self.logger.debug(f"Processed ratings chunk {chunk_index} of size {chunk_size}.")
            return True
        except Exception as e:
            self.logger.error(f"Error processing ratings chunk {chunk_index} of size {chunk_size}: {e}", exc_info=True)
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
                'check_interval': self.check_interval,
                'max_workers': self.executor._max_workers,
                'queue_get_timeout': BatchProcessor.QUEUE_GET_TIMEOUT,
                'last_check': datetime.now().isoformat()
            }

    def stop(self):
        """Stops the batch processor gracefully."""
        self.logger.info("Stopping batch processor...")
        self.should_run = False
        if self.checker_thread and self.checker_thread.is_alive():
            self.logger.debug("Waiting for batch checker thread to stop...")
            self.checker_thread.join(timeout=5)
            if self.checker_thread.is_alive():
                self.logger.warning("Batch checker thread did not stop gracefully after 5 seconds.")
            else:
                self.logger.info("Batch checker thread stopped.")

        self.logger.debug("Shutting down thread pool executor...")
        self.executor.shutdown(wait=True) # Wait for all tasks to complete
        self.logger.info("Batch processor stopped and shutdown complete.")