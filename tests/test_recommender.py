import unittest
import pandas as pd
import numpy as np
from src.recommender import MovieRecommenderSystem
from src.batch_processor import BatchProcessor

class TestRecommenderSystem(unittest.TestCase):
    """Test cases for the MovieRecommenderSystem class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.recommender = MovieRecommenderSystem()
        self.recommender.initialize()

    def test_recommender_initialization(self):
        """Test if recommender system initializes correctly."""
        self.assertIsNotNone(self.recommender)
        self.assertEqual(self.recommender.cache, {})
        self.assertFalse(self.recommender.is_updating)

    def test_get_similar_movies(self):
        """Test getting similar movies."""
        similar_movies = self.recommender.get_similar_movies(1)
        self.assertIsInstance(similar_movies, list)
        if similar_movies:  # If recommendations exist
            self.assertIsInstance(similar_movies[0], dict)
            self.assertIn('movie_id', similar_movies[0])
            self.assertIn('title', similar_movies[0])

    def test_get_user_recommendations(self):
        """Test getting user recommendations."""
        recommendations = self.recommender.get_user_recommendations(1)
        self.assertIsInstance(recommendations, list)
        if recommendations:  # If recommendations exist
            self.assertIsInstance(recommendations[0], dict)
            self.assertIn('movie_id', recommendations[0])
            self.assertIn('title', recommendations[0])

    def test_get_top_popular_movies(self):
        """Test getting popular movies."""
        popular_movies = self.recommender.get_top_popular_movies(n_recommendations=5)
        self.assertIsInstance(popular_movies, list)
        self.assertLessEqual(len(popular_movies), 5)
        if popular_movies:
            self.assertIsInstance(popular_movies[0], dict)

    def test_add_new_rating(self):
        """Test adding a new rating."""
        batch_processor = BatchProcessor()
        self.recommender.set_batch_processor(batch_processor)
        batch_processor.set_recommender(self.recommender)
        
        result = self.recommender.add_new_rating(1, 1, 4.0)
        self.assertTrue(result)

    def test_cache_behavior(self):
        """Test recommendation caching."""
        user_id = 1
        # First request should not be in cache
        self.assertNotIn(user_id, self.recommender.cache)
        recommendations = self.recommender.get_user_recommendations(user_id)
        # After request, should be in cache
        self.assertIn(user_id, self.recommender.cache)
        self.assertEqual(self.recommender.cache[user_id]['recommendations'], recommendations)

    def test_invalid_movie_id(self):
        """Test error handling for invalid movie ID."""
        similar_movies = self.recommender.get_similar_movies(-1)
        self.assertEqual(similar_movies, [])

    def test_invalid_user_id(self):
        """Test error handling for invalid user ID."""
        recommendations = self.recommender.get_user_recommendations(-1)
        self.assertIsInstance(recommendations, list)  # Should return popular movies for invalid users

    def tearDown(self):
        """Clean up after each test."""
        self.recommender = None
