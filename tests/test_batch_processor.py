import unittest
import time
from src.batch_processor import BatchProcessor
from src.recommender import MovieRecommenderSystem

class TestBatchProcessor(unittest.TestCase):
    """Test cases for the BatchProcessor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_processor = BatchProcessor(batch_size=2, check_interval=0.1)
        self.mock_rating = {
            'user_id': 1,
            'movie_id': 1,
            'rating': 4.0,
            'timestamp': int(time.time())
        }

    def test_initialization(self):
        """Test batch processor initialization."""
        self.assertEqual(self.batch_processor.batch_size, 2)
        self.assertFalse(self.batch_processor.is_processing)
        self.assertEqual(self.batch_processor.processed_batches, 0)

    def test_validate_rating(self):
        """Test rating validation."""
        self.assertTrue(self.batch_processor.validate_rating(self.mock_rating))

        # Test invalid ratings
        invalid_rating = self.mock_rating.copy()
        invalid_rating['rating'] = 6.0
        self.assertFalse(self.batch_processor.validate_rating(invalid_rating))

        invalid_rating['rating'] = "not a number"
        self.assertFalse(self.batch_processor.validate_rating(invalid_rating))

    def test_add_to_batch(self):
        """Test adding ratings to batch."""
        self.assertTrue(self.batch_processor.add_to_batch(self.mock_rating))
        self.assertEqual(self.batch_processor.batch_queue.qsize(), 1)

    def test_batch_processing_trigger(self):
        """Test batch processing trigger."""
        recommender = MovieRecommenderSystem()
        self.batch_processor.set_recommender(recommender)
        
        # Add ratings to trigger processing
        self.batch_processor.add_to_batch(self.mock_rating)
        self.batch_processor.add_to_batch(self.mock_rating)
        
        # Wait for processing to start
        time.sleep(0.2)
        self.assertEqual(self.batch_processor.batch_queue.qsize(), 0)

    def test_batch_status(self):
        """Test batch status reporting."""
        self.batch_processor.add_to_batch(self.mock_rating)
        status = self.batch_processor.get_batch_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('queue_size', status)
        self.assertIn('is_processing', status)
        self.assertIn('processed_batches', status)

    def test_graceful_shutdown(self):
        """Test graceful shutdown of batch processor."""
        self.batch_processor.stop()
        self.assertFalse(self.batch_processor.should_run)

    def tearDown(self):
        """Clean up after each test method."""
        self.batch_processor = None

if __name__ == '__main__':
    unittest.main()
