import unittest
import os
import pandas as pd
import numpy as np
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for the DataLoader class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.data_loader = DataLoader()

    def test_initialization(self):
        """Test data loader initialization."""
        self.assertIsNotNone(self.data_loader)
        self.assertIsNone(self.data_loader.ratings_df)
        self.assertIsNone(self.data_loader.movies_df)
        self.assertIsNone(self.data_loader.users_df)

    def test_dataset_download(self):
        """Test dataset download functionality."""
        self.data_loader.download_dataset()
        self.assertTrue(os.path.exists('ml-1m'))
        self.assertTrue(os.path.exists('ml-1m/ratings.dat'))
        self.assertTrue(os.path.exists('ml-1m/movies.dat'))
        self.assertTrue(os.path.exists('ml-1m/users.dat'))

    def test_data_loading(self):
        """Test data loading functionality."""
        self.data_loader.load_data()
        self.assertIsInstance(self.data_loader.ratings_df, pd.DataFrame)
        self.assertIsInstance(self.data_loader.movies_df, pd.DataFrame)
        self.assertIsInstance(self.data_loader.users_df, pd.DataFrame)

    def test_user_movie_matrix(self):
        """Test user-movie matrix creation."""
        self.data_loader.load_data()
        user_movie_matrix, normalized_matrix = self.data_loader.get_user_movie_matrix()
        
        self.assertIsInstance(user_movie_matrix, pd.DataFrame)
        self.assertIsInstance(normalized_matrix, np.ndarray)
        self.assertEqual(user_movie_matrix.shape[1], 
                        len(self.data_loader.ratings_df['user_id'].unique()))

    def test_movie_info_retrieval(self):
        """Test movie information retrieval."""
        self.data_loader.load_data()
        movie_id = self.data_loader.movies_df['movie_id'].iloc[0]
        movie_info = self.data_loader.get_movie_info(movie_id)
        
        self.assertIsInstance(movie_info, dict)
        self.assertIn('movie_id', movie_info)
        self.assertIn('title', movie_info)
        self.assertIn('genres', movie_info)
        self.assertIn('year', movie_info)

    def test_invalid_movie_info_retrieval(self):
        """Test invalid movie ID handling."""
        self.data_loader.load_data()
        movie_info = self.data_loader.get_movie_info(-1)
        self.assertEqual(movie_info, {})

    def test_movie_ids_retrieval(self):
        """Test movie IDs list retrieval."""
        self.data_loader.load_data()
        movie_ids = self.data_loader.get_movie_ids()
        self.assertIsInstance(movie_ids, list)
        self.assertGreater(len(movie_ids), 0)
        self.assertTrue(all(isinstance(mid, (int, np.int64)) for mid in movie_ids))

    def tearDown(self):
        """Clean up after each test method."""
        self.data_loader = None

if __name__ == '__main__':
    unittest.main()
