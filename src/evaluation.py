import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from sklearn.model_selection import KFold
from .utils.logger import setup_logger
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

class RecommenderEvaluator:
    def __init__(self, recommender):
        self.recommender = recommender
        self.logger = setup_logger(__name__)
        # Cache data at initialization
        self.movies_df = recommender.data_loader.movies_df
        self.ratings_df = recommender.data_loader.ratings_df
        # Pre-compute movie genres dictionary
        self.movie_genres = self._prepare_movie_genres()
        # Pre-compute movie popularity
        self.movie_popularity = self._prepare_movie_popularity()

    def _prepare_movie_genres(self):
        """Pre-compute movie genres dictionary."""
        return {
            row['movie_id']: set(row['genres']) 
            for _, row in self.movies_df.iterrows()
        }

    def _prepare_movie_popularity(self):
        """Pre-compute movie popularity scores."""
        counts = self.ratings_df['movie_id'].value_counts()
        total = len(self.ratings_df)
        return {movie_id: count/total for movie_id, count in counts.items()}

    def evaluate(self, k: int = 10, n_folds: int = 1) -> Dict[str, Dict[str, float]]:
        """Performs k-fold cross-validation using ProcessPoolExecutor."""
        self.logger.info(f"Starting {n_folds}-fold cross-validation evaluation...")
        start_time = datetime.now()

        valid_users = self._get_valid_users(min_ratings=10)
        all_metrics = {'coverage': [], 'diversity': [], 'novelty': []}

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        ratings_array = np.array(valid_users)

        for fold, (train_idx, test_idx) in enumerate(kf.split(ratings_array), 1):
            self.logger.info(f"Processing fold {fold}/{n_folds}")
            test_users = ratings_array[test_idx]
            
            # Get recommendations for all test users
            user_recommendations = {}
            for user_id in test_users:
                try:
                    recs = self.recommender.get_user_recommendations(user_id, k)
                    user_recommendations[user_id] = [r['movie_id'] for r in recs]
                except Exception as e:
                    self.logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")

            # Calculate metrics in parallel
            fold_metrics = self._calculate_metrics_parallel(user_recommendations)
            
            for metric, values in fold_metrics.items():
                all_metrics[metric].append(np.mean(values))

        # Calculate final results
        results = {}
        for metric, values in all_metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            ci = 1.96 * std_value / np.sqrt(n_folds)
            
            results[metric] = {
                'mean': mean_value,
                'std': std_value,
                'ci_lower': mean_value - ci,
                'ci_upper': mean_value + ci
            }

        evaluation_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        return results

    def _calculate_metrics_parallel(self, user_recommendations: Dict[int, List[int]]) -> Dict[str, List[float]]:
        """Calculate metrics for all users using parallel processing."""
        metrics = {
            'coverage': [],
            'diversity': [],
            'novelty': []
        }
        
        all_movies = set(self.movies_df['movie_id'])
        recommended_items = set()
        
        # Process recommendations sequentially since they're already computed
        for user_id, rec_items in user_recommendations.items():
            recommended_items.update(rec_items)
            
            # Calculate metrics
            metrics['diversity'].append(self._calculate_diversity(rec_items))
            metrics['novelty'].append(self._calculate_novelty(rec_items))

        # Calculate coverage once for all recommendations
        coverage = len(recommended_items) / len(all_movies)
        metrics['coverage'] = [coverage] * len(user_recommendations)
        
        return metrics

    def _calculate_diversity(self, recommendations: List[int]) -> float:
        """Calculate diversity using pre-computed genre data."""
        if len(recommendations) < 2:
            return 0.0
            
        total_dissimilarity = 0
        count = 0
        
        for i, movie1_id in enumerate(recommendations[:-1]):
            movie1_genres = self.movie_genres.get(movie1_id, set())
            
            for movie2_id in recommendations[i+1:]:
                movie2_genres = self.movie_genres.get(movie2_id, set())
                
                if movie1_genres or movie2_genres:
                    dissimilarity = 1 - len(movie1_genres & movie2_genres) / len(movie1_genres | movie2_genres)
                    total_dissimilarity += dissimilarity
                    count += 1
                
        return total_dissimilarity / count if count > 0 else 0.0

    def _calculate_novelty(self, recommendations: List[int]) -> float:
        """Calculate novelty using pre-computed popularity data."""
        if not recommendations:
            return 0.0
        
        novelties = []
        for item in recommendations:
            popularity = self.movie_popularity.get(item, 0)
            novelty = -np.log2(popularity) if popularity > 0 else 0
            novelties.append(novelty)
            
        return np.mean(novelties)

    def _get_valid_users(self, min_ratings: int = 10) -> List[int]:
        """Returns users with at least min_ratings ratings."""
        user_counts = self.ratings_df['user_id'].value_counts()
        return user_counts[user_counts >= min_ratings].index.tolist()
