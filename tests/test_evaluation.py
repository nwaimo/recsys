import unittest
import numpy as np
from src.evaluation import RecommenderEvaluator
from src.recommender import MovieRecommenderSystem

class TestRecommenderEvaluator(unittest.TestCase):
    """Test cases for the RecommenderEvaluator class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        recommender = MovieRecommenderSystem()
        recommender.initialize()
        self.evaluator = RecommenderEvaluator(recommender)

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator)
        self.assertIsNotNone(self.evaluator.recommender)
        self.assertIsNotNone(self.evaluator.movie_genres)
        self.assertIsNotNone(self.evaluator.movie_popularity)

    def test_diversity_calculation(self):
        """Test diversity metric calculation."""
        recommendations = [1, 2, 3, 4, 5]  # movie IDs
        diversity = self.evaluator._calculate_diversity(recommendations)
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0)
        self.assertLessEqual(diversity, 1)

    def test_novelty_calculation(self):
        """Test novelty metric calculation."""
        recommendations = [1, 2, 3, 4, 5]  # movie IDs
        novelty = self.evaluator._calculate_novelty(recommendations)
        self.assertIsInstance(novelty, float)
        self.assertGreaterEqual(novelty, 0)

    def test_evaluation_results(self):
        """Test full evaluation process."""
        results = self.evaluator.evaluate(k=5, n_folds=2)
        
        self.assertIsInstance(results, dict)
        required_metrics = ['coverage', 'diversity', 'novelty']
        required_stats = ['mean', 'std', 'ci_lower', 'ci_upper']
        
        for metric in required_metrics:
            self.assertIn(metric, results)
            for stat in required_stats:
                self.assertIn(stat, results[metric])
                self.assertIsInstance(results[metric][stat], float)

    def test_valid_users_filtering(self):
        """Test filtering of valid users for evaluation."""
        valid_users = self.evaluator._get_valid_users(min_ratings=5)
        self.assertIsInstance(valid_users, list)
        self.assertGreater(len(valid_users), 0)
        self.assertTrue(all(isinstance(uid, (int, np.int64)) for uid in valid_users))

    def test_empty_recommendations(self):
        """Test handling of empty recommendations."""
        diversity = self.evaluator._calculate_diversity([])
        self.assertEqual(diversity, 0.0)
        
        novelty = self.evaluator._calculate_novelty([])
        self.assertEqual(novelty, 0.0)

  

    def tearDown(self):
        """Clean up after each test."""
        self.evaluator = None

if __name__ == '__main__':
    unittest.main()
