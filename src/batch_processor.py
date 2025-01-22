from queue import Queue
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

class BatchProcessor:
    def __init__(self, recommender_system, batch_size=1):
        self.recommender = recommender_system
        self.batch_size = batch_size
        self.batch_queue = Queue()
        self.processing_thread = None
        self.is_processing = False
        self.processed_batches = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)

    def add_to_batch(self, rating_data):
        """Add a new rating to the batch queue"""
        self.batch_queue.put(rating_data)
        
        if self.batch_queue.qsize() >= self.batch_size:
            self.process_batch()

    def process_batch(self):
        """Process a batch of ratings"""
        if self.is_processing:
            return

        self.is_processing = True
        batch = []
        
        while not self.batch_queue.empty() and len(batch) < self.batch_size:
            batch.append(self.batch_queue.get())
            

        if batch:
            try:
                futures = []
                for rating_chunk in np.array_split(batch, 4):
                    futures.append(
                        self.executor.submit(self._process_ratings_chunk, rating_chunk)
                    )

                for future in futures:
                    future.result()

                self.processed_batches += 1
                self.logger.info(f"Processed batch {self.processed_batches} with {len(batch)} ratings")
            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}")
            finally:
                self.is_processing = False

    def _process_ratings_chunk(self, ratings_chunk):
        """Process a chunk of ratings in parallel"""
        new_ratings_df = pd.DataFrame(ratings_chunk)
        self.recommender.update_feature_matrices(new_ratings_df)
        
        return True

    def get_batch_status(self):
        """Get current status of batch processing"""
        return {
            'queue_size': self.batch_queue.qsize(),
            'is_processing': self.is_processing,
            'processed_batches': self.processed_batches,
            'batch_size': self.batch_size
            
        }
