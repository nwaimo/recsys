import numpy as np
import faiss
import joblib
import threading
import time
import psutil
import pandas as pd
from datetime import datetime, timedelta
from collections import deque, Counter
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional

# Add imports for the improved classes
from .analytics import AnalyticsMonitor  # Assuming you move the improved AnalyticsMonitor to a file named analytics.py
from .batch_processor import BatchProcessor  # Assuming you move the improved BatchProcessor to a file named batch_processor.py
from .data_loader import DataLoader
from .utils.logger import setup_logger

class MovieRecommenderSystem:
    """
    A movie recommender system that uses collaborative filtering and content-based
    filtering to provide movie recommendations. It incorporates features like:

    - Approximate Nearest Neighbors (ANN) search using FAISS for efficient similarity computation.
    - Popularity-based recommendations.
    - Caching of user recommendations.
    - Batch processing of new ratings for model updates.
    - Real-time analytics monitoring.
    - Periodic model updates.

    """

    def __init__(self):
        """
        Initializes the MovieRecommenderSystem with data loading capabilities.
        Analytics and batch processing are added later via dependency injection.
        """
        self.logger = setup_logger(__name__)
        self.data_loader = DataLoader()
        self.analytics = None
        self.batch_processor = None
        

        # Core data structures - Will be populated during initialization
        self.user_movie_matrix: Optional[pd.DataFrame] = None  # Pivot table of user-movie ratings
        self.movie_features: Optional[np.ndarray] = None      # Feature matrix for movies (e.g., from user-movie matrix)
        self.index: Optional[faiss.IndexFlatL2] = None       # FAISS index for similarity search
        self.movie_ids: Optional[pd.Index] = None            # Index of movie IDs corresponding to rows in movie_features
        self.popularity_scores: Optional[pd.Series] = None   # Popularity scores for each movie

        # Cache and real-time components
        self.cache: Dict[int, Dict] = {}  # Cache for user recommendations: {user_id: {'recommendations': [], 'timestamp': ...}}
        self.update_lock = threading.Lock() # Lock for updating shared resources
        self.last_update_time: Optional[datetime] = None  # Timestamp of the last model update
        self.update_interval = timedelta(minutes=30)  # Interval for periodic model updates
        self.is_updating = False  # Flag to indicate if a model update is in progress

    def set_analytics(self, analytics):
        """Set the analytics monitor via dependency injection"""
        self.analytics = analytics

    def set_batch_processor(self, batch_processor):
        """Set the batch processor via dependency injection"""
        self.batch_processor = batch_processor

    def setup_logging(self):
        """Remove this method as we now use the centralized logger."""
        pass  # Logging is now handled by setup_logger

    def initialize(self):
        """
        Initializes the recommender system by:
        1. Downloading the dataset (if not already downloaded).
        2. Loading data from CSV files into Pandas DataFrames.
        3. Creating feature matrices.
        4. Building the FAISS index for similarity search.
        """
        self.logger.info("Initializing recommender system...")
        self.data_loader.download_dataset()
        self.data_loader.load_data()
        self.create_feature_matrices()
        self.build_ann_index()
        self.logger.info("Recommender system initialized.")

    def create_feature_matrices(self):
        """
        Creates feature matrices for collaborative filtering and content-based recommendations.
        This involves:
        1. Creating a user-movie matrix (pivot table of ratings).
        2. Applying feature scaling (StandardScaler) to the user-movie matrix.
        3. Calculating popularity scores for each movie.
        """
        self.logger.info("Creating feature matrices...")
        
        # Get user-movie matrix and normalized features
        self.user_movie_matrix, self.movie_features = self.data_loader.get_user_movie_matrix()
        self.movie_ids = self.user_movie_matrix.index
        self.logger.info("User-movie matrix and movie features created.")

        # Calculate popularity scores
        self.calculate_popularity_scores()
        self.logger.info("Popularity scores calculated.")

    def calculate_popularity_scores(self):
        """
        Calculates popularity scores for each movie based on:

        - Number of ratings
        - Average rating
        - Time decay (more recent ratings are weighted higher)

        The popularity score is a weighted combination of these factors.
        """
        self.logger.info("Calculating popularity scores...")
        current_timestamp = self.data_loader.ratings_df['timestamp'].max()

        rating_stats = self.data_loader.ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean'],
            'timestamp': 'max'
        }).reset_index()

        rating_stats.columns = ['movie_id', 'rating_count', 'rating_mean', 'last_rated']
        
        # Time decay: Reduce the weight of older ratings
        rating_stats['time_weight'] = 1 / (1 + np.log1p((current_timestamp - rating_stats['last_rated']) / (24 * 60 * 60)))
        
        # Popularity score formula (can be customized)
        rating_stats['popularity_score'] = (
                rating_stats['rating_count'] * rating_stats['rating_mean'] * rating_stats['time_weight']
        )

        self.popularity_scores = rating_stats.set_index('movie_id')['popularity_score']
        self.logger.info("Popularity scores calculated.")

    def build_ann_index(self):
        """
        Builds a FAISS index for efficient approximate nearest neighbor (ANN) search.
        This index is used to find movies similar to a given movie based on their feature vectors.
        """
        self.logger.info("Building FAISS index...")
        n_features = self.movie_features.shape[1]

        # Convert features to float32 (FAISS requirement) and ensure contiguous array
        self.movie_features = np.ascontiguousarray(self.movie_features.astype('float32'))

        # Create and build the index (using L2 distance here)
        self.index = faiss.IndexFlatL2(n_features)
        self.index.add(self.movie_features)
        self.logger.info(f"FAISS index built with {self.index.ntotal} movies.")

    def get_similar_movies(self, movie_id: int, n_neighbors: int = 5) -> List[Dict[str, Any]]:
        """
        Finds movies similar to a given movie using the FAISS index.

        Args:
            movie_id: The ID of the movie for which to find similar movies.
            n_neighbors: The number of similar movies to return.

        Returns:
            A list of dictionaries, where each dictionary represents a similar movie and contains its ID, title, and other information.
        """
        start_time = time.time()
        self.analytics.record_recommendation_request()

        try:
            # Get the index of the movie in the movie_ids
            movie_idx = self.movie_ids.get_loc(movie_id)

            # Get the movie vector (features) and reshape for FAISS
            movie_vector = self.movie_features[movie_idx].reshape(1, -1)

            # Search using FAISS
            distances, indices = self.index.search(movie_vector, n_neighbors + 1)  # +1 to include the movie itself

            # Get the movie IDs of the similar movies (excluding the query movie itself)
            similar_movie_indices = indices[0][1:]
            similar_movie_ids = self.movie_ids[similar_movie_indices]

            # Get movie information for the similar movies
            similar_movies = []
            for mid in similar_movie_ids:
                similar_movies.append(self.data_loader.get_movie_info(mid))

            response_time = time.time() - start_time
            self.analytics.record_response_time(response_time)
            self.logger.debug(f"Similar movies for movie ID {movie_id} found in {response_time:.4f} seconds.")
            return similar_movies

        except KeyError:
            self.analytics.record_error('movie_not_found')
            self.logger.warning(f"Movie ID {movie_id} not found in the database.")
            return []  # Return an empty list if movie not found
        except Exception as e:
            self.analytics.record_error('similar_movies_error')
            self.logger.error(f"Error getting similar movies for movie ID {movie_id}: {e}")
            return []  # Return an empty list on error

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Generates personalized movie recommendations for a given user.

        The recommendation process involves:

        1. **Checking the cache:** If recommendations for the user are already in the cache and are not stale, return them.
        2. **Finding similar movies to the user's top-rated movies:** Get the user's highest-rated movies and find other movies similar to them using `get_similar_movies`.
        3. **Filtering out movies the user has already seen.**
        4. **Using popularity scores to rank or filter recommendations** (if popularity scores are available).
        5. **Caching the generated recommendations.**

        Args:
            user_id: The ID of the user for whom to generate recommendations.
            n_recommendations: The number of recommendations to generate.

        Returns:
            A list of dictionaries, where each dictionary represents a recommended movie.
        """
        start_time = time.time()
        self.analytics.record_recommendation_request()

        try:
            # 1. Check the cache
            if user_id in self.cache and time.time() - self.cache[user_id]['timestamp'] < self.update_interval.total_seconds():
                self.analytics.record_cache_hit()
                response_time = time.time() - start_time
                self.analytics.record_response_time(response_time)
                self.logger.debug(f"Recommendations for user {user_id} found in cache in {response_time:.4f} seconds")
                return self.cache[user_id]['recommendations']

            self.analytics.record_cache_miss()

            # 2. Get user's rated movies
            user_ratings = self.data_loader.ratings_df[self.data_loader.ratings_df['user_id'] == user_id]
            
            # Handle cases where the user has no ratings
            if user_ratings.empty:
                self.logger.info(f"User {user_id} has no ratings. Returning top popular movies.")
                return self.get_top_popular_movies(n_recommendations)

            # 3. Find the user's top-rated movies and get similar movies
            user_avg_rating = user_ratings['rating'].mean()
            rated_movies = user_ratings['movie_id'].values
            top_rated_movies = user_ratings[user_ratings['rating'] >= user_avg_rating]['movie_id'].values

            similar_movie_candidates: List[Dict] = []
            for movie_id in top_rated_movies:
                similar_movies = self.get_similar_movies(movie_id, n_neighbors=3) # Get top 3 similar movies for each top-rated movie
                if similar_movies:
                    similar_movie_candidates.extend(similar_movies)

            # 4. Filter out movies the user has already rated
            
            recommendations = [movie for movie in similar_movie_candidates if movie['movie_id'] not in rated_movies]

            # Remove duplicate recommendations (if any)
            recommendations = list({movie['movie_id']: movie for movie in recommendations}.values())

            # 5. Sort by popularity (if available) and take top N
            if self.popularity_scores is not None:
                recommendations.sort(key=lambda movie: self.popularity_scores.get(movie['movie_id'], 0), reverse=True)

            recommendations = recommendations[:n_recommendations]

            # 6. Cache the recommendations
            self.cache[user_id] = {
                'recommendations': recommendations,
                'timestamp': time.time()
            }

            response_time = time.time() - start_time
            self.analytics.record_response_time(response_time)
            self.logger.info(f"Recommendations for user {user_id} generated in {response_time:.4f} seconds")
            return recommendations

        except Exception as e:
            self.analytics.record_error('recommendation_error')
            self.logger.error(f"Error getting recommendations for user {user_id}: {e}")
            return []

    def get_top_popular_movies(self, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the top N most popular movies based on their popularity scores.

        Args:
            n_recommendations: The number of popular movies to return.

        Returns:
            A list of dictionaries, where each dictionary represents a popular movie.
        """
        if self.popularity_scores is None:
            self.logger.warning("Popularity scores not calculated. Returning empty list.")
            return []

        top_movie_ids = self.popularity_scores.nlargest(n_recommendations).index
        top_movies = [self.data_loader.get_movie_info(movie_id) for movie_id in top_movie_ids]
        return top_movies
    
    
    def get_genre_recommendations(self, genres: List[str], n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Gets movie recommendations based on specified genres.

        Args:
            genres: A list of genre names (e.g., "Action", "Comedy").
            n_recommendations: The number of recommendations to return.

        Returns:
            A list of dictionaries, where each dictionary represents a recommended movie.
        """
        self.logger.info(f"Getting genre recommendations for genres: {', '.join(genres)}")
        start_time = time.time()
        self.analytics.record_recommendation_request()
        
        try:
            # Filter movies that have at least one of the specified genres
            genre_movies = self.data_loader.movies_df[
                self.data_loader.movies_df['genres'].apply(lambda x: any(g in x for g in genres))
            ]

            # If no movies found for the specified genre, return an empty list
            if genre_movies.empty:
                self.logger.warning(f"No movies found for genres: {', '.join(genres)}")
                return []
            
            # Merge with movie stats to get rating count and mean for each movie
            movie_stats = self.data_loader.ratings_df.groupby('movie_id').agg({
                'rating': ['count', 'mean']
            })
            movie_stats.columns = ['rating_count', 'rating_mean']
            recommendations = genre_movies.merge(movie_stats, on='movie_id')
            
            # If popularity scores exist, merge those as well
            if self.popularity_scores is not None:
                recommendations = recommendations.merge(self.popularity_scores, how='left', left_on='movie_id', right_index=True)
                recommendations.fillna({'popularity_score': 0}, inplace=True)
                
                # Sort by popularity score, then rating count and mean (in descending order)
                recommendations = recommendations.sort_values(
                    by=['popularity_score', 'rating_count', 'rating_mean'],
                    ascending=[False, False, False]
                )
            else:
                # Sort by rating count and mean (in descending order) if no popularity scores are available
                recommendations = recommendations.sort_values(
                    by=['rating_count', 'rating_mean'],
                    ascending=[False, False]
                )

            # Get the top N recommendations
            top_recommendations = recommendations.head(n_recommendations)

            # Convert to list of dictionaries
            recommendation_list = []
            for _, row in top_recommendations.iterrows():
                recommendation_list.append({
                    'movie_id': row['movie_id'],
                    'title': row['title'],
                    'genres': row['genres'],
                    'year': row['year'],
                    'rating_count': row['rating_count'],
                    'rating_mean': row['rating_mean']
                })
            
            response_time = time.time() - start_time
            self.analytics.record_response_time(response_time)
            self.logger.info(f"Genre recommendations generated in {response_time:.4f} seconds")
            
            return recommendation_list

        except Exception as e:
            self.analytics.record_error('genre_recommendation_error')
            self.logger.error(f"Error getting genre recommendations: {e}")
            return []

    def add_new_rating(self, user_id: int, movie_id: int, rating: float, timestamp: Optional[int] = None):
        """
        Adds a new rating to the system. The rating is:

        1. Added to the `BatchProcessor` for later model updates.
        2. Used to update analytics metrics.

        Args:
            user_id: The ID of the user who gave the rating.
            movie_id: The ID of the movie being rated.
            rating: The rating value (e.g., 4.5).
            timestamp: The timestamp of the rating (optional, defaults to current time).
        """
        self.logger.debug(f"Adding new rating: user_id={user_id}, movie_id={movie_id}, rating={rating}")

        if timestamp is None:
            timestamp = int(time.time())

        new_rating = {
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp
        }

        # Add the new rating to the batch processor
        if not self.batch_processor.add_to_batch(new_rating):
            self.logger.error(f"Failed to add rating to batch processor: {new_rating}")

        # Update analytics metrics
        self.analytics.record_rating(user_id, movie_id, rating)

    def update_feature_matrices_with_new_data(self):
        """
        Updates the feature matrices with new data from the batch processor.
        
        This involves:
        1. Getting the new ratings from the batch processor.
        2. Creating a new pivot table from the new ratings.
        3. Concatenating the new pivot table with the existing user-movie matrix.
        4. Re-applying feature scaling (StandardScaler).
        5. Rebuilding the FAISS index.
        6. Recalculating popularity scores.
        """
        if not self.update_lock.acquire(blocking=False):
            self.logger.info("Update already in progress. Skipping.")
            return

        try:
            self.is_updating = True
            self.logger.info("Updating feature matrices with new data...")
            start_time = time.time()

            # Get new ratings from the batch processor (this would ideally be a method in BatchProcessor)
            new_ratings_df = pd.DataFrame(self.batch_processor._get_batch_from_queue())

            if not new_ratings_df.empty:
                # Create a new pivot table from the new ratings
                new_ratings_pivot = new_ratings_df.pivot_table(
                    index='movie_id',
                    columns='user_id',
                    values='rating'
                ).fillna(0)

                # Concatenate the new pivot table with the existing user-movie matrix
                self.user_movie_matrix = pd.concat([
                    self.user_movie_matrix,
                    new_ratings_pivot
                ]).groupby(level=0).sum()  # Aggregate duplicate entries by summing

                # Re-apply feature scaling
                scaler = StandardScaler()
                self.movie_features = scaler.fit_transform(self.user_movie_matrix)
                self.movie_ids = self.user_movie_matrix.index

                # Rebuild the FAISS index
                self.build_ann_index()

                # Recalculate popularity scores
                self.calculate_popularity_scores()

                self.last_update_time = datetime.now()
                self.logger.info(f"Feature matrices updated in {time.time() - start_time:.2f} seconds.")
            else:
                self.logger.info("No new ratings to update.")

        except Exception as e:
            self.analytics.record_error('update_error')
            self.logger.error(f"Error updating feature matrices: {e}")
        finally:
            self.is_updating = False
            self.update_lock.release()

    def evaluate_recommendations(self, test_users: Optional[List[int]] = None, k: int = 10) -> Dict[str, float]:
        """
        Evaluates the recommender system's performance using metrics like NDCG, Precision@k, Recall@k, and MAP@k.

        Args:
            test_users: A list of user IDs to use for evaluation. If None, uses a random sample of 1000 users (or all users if fewer than 1000).
            k: The number of recommendations to generate for evaluation.

        Returns:
            A dictionary containing the average evaluation metrics.
        """
        self.logger.info("Evaluating recommendations...")

        if test_users is None:
            all_users = self.data_loader.ratings_df['user_id'].unique()
            test_users = np.random.choice(all_users, min(1000, len(all_users)), replace=False)

        metrics = {
            'ndcg': [],
            'precision': [],
            'recall': [],
            'map': []
        }

        for user_id in test_users:
            user_ratings = self.data_loader.ratings_df[self.data_loader.ratings_df['user_id'] == user_id].sort_values('timestamp')

            if len(user_ratings) < 10:  # Need at least k+1 ratings for evaluation
                continue

            train_ratings = user_ratings.iloc[:-5]  # Hold out the last 5 ratings for testing
            test_ratings = user_ratings.iloc[-5:]
            test_movies = set(test_ratings['movie_id'])

            # Temporarily add the train ratings to the recommender system's data
            self.data_loader.ratings_df = pd.concat([self.data_loader.ratings_df, train_ratings])
            self.create_feature_matrices()
            self.build_ann_index()

            # Get recommendations
            recommendations = self.get_user_recommendations(user_id, n_recommendations=k)
            
            # Remove the temporarily added train ratings
            self.data_loader.ratings_df = self.data_loader.ratings_df.drop(train_ratings.index)
            self.create_feature_matrices()
            self.build_ann_index()

            if not recommendations:
                continue

            rec_movies = [r['movie_id'] for r in recommendations]

            # Calculate metrics
            metrics['ndcg'].append(self._calculate_ndcg(test_movies, rec_movies, k))
            precision, recall = self._calculate_precision_recall(test_movies, rec_movies, k)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['map'].append(self._calculate_map(test_movies, rec_movies, k))

        # Average metrics across users
        avg_metrics = {
            'ndcg@k': np.mean(metrics['ndcg']),
            'precision@k': np.mean(metrics['precision']),
            'recall@k': np.mean(metrics['recall']),
            'map@k': np.mean(metrics['map'])
        }
        self.logger.info(f"Evaluation complete: {avg_metrics}")
        return avg_metrics

    def _calculate_ndcg(self, actual: set, predicted: list, k: int) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) at k.

        Args:
            actual: A set of the actual relevant movie IDs.
            predicted: A list of the predicted movie IDs (in order).
            k: The position up to which to calculate NDCG.

        Returns:
            The NDCG@k value.
        """
        dcg = 0.0
        idcg = 0.0

        for i, movie_id in enumerate(predicted[:k]):
            if movie_id in actual:
                dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0

        for i in range(min(len(actual), k)):
            idcg += 1 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_precision_recall(self, actual: set, predicted: list, k: int) -> Tuple[float, float]:
        """
        Calculates Precision@k and Recall@k.

        Args:
            actual: A set of the actual relevant movie IDs.
            predicted: A list of the predicted movie IDs.
            k: The position up to which to calculate precision and recall.

        Returns:
            A tuple containing the Precision@k and Recall@k values.
        """
        relevant_and_retrieved = len(actual.intersection(set(predicted[:k])))
        precision = relevant_and_retrieved / k if k > 0 else 0.0
        recall = relevant_and_retrieved / len(actual) if len(actual) > 0 else 0.0
        return precision, recall

    def _calculate_map(self, actual: set, predicted: list, k: int) -> float:
        """
        Calculates the Mean Average Precision (MAP) at k.

        Args:
            actual: A set of the actual relevant movie IDs.
            predicted: A list of the predicted movie IDs.
            k: The position up to which to calculate MAP.

        Returns:
            The MAP@k value.
        """
        ap = 0.0
        num_hits = 0.0

        for i, movie_id in enumerate(predicted[:k]):
            if movie_id in actual:
                num_hits += 1.0
                ap += num_hits / (i + 1.0)

        return ap / min(len(actual), k) if len(actual) > 0 else 0.0

    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """
        Generates a dictionary containing analytics data for a dashboard.

        Returns:
            A dictionary with analytics report, batch processing status, and system status.
        """
        report = self.analytics.get_analytics_report()
        batch_status = self.batch_processor.get_batch_status()
        system_status = self.get_system_status()

        dashboard = {
            'analytics': report,
            'batch_processing': batch_status,
            'system_status': system_status
        }
        self.logger.info("Analytics dashboard generated.")
        return dashboard

    def get_system_status(self) -> Dict[str, Any]:
        """
        Gets the current system status.

        Returns:
            A dictionary containing information about the system's current state.
        """
        return {
            'total_ratings': len(self.data_loader.ratings_df),
            'total_users': len(self.data_loader.ratings_df['user_id'].unique()),
            'total_movies': len(self.data_loader.movies_df),
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else "Not yet updated",
            'is_updating': self.is_updating,
            'cache_size': len(self.cache),
            'batch_queue_size': self.batch_processor.batch_queue.qsize()
        }

    def start_update_process(self):
        """
        Starts the process of updating feature matrices in a separate thread if an update is not already in progress.
        """
        if not self.is_updating:
            self.logger.info("Starting scheduled update process...")
            threading.Thread(target=self.update_feature_matrices_with_new_data, daemon=True).start()
        else:
            self.logger.info("Update already in progress.")

    def save_model(self, filepath: str):
        """
        Saves the current state of the recommender system to disk.

        Args:
            filepath: The path to save the model to.
        """
        try:
            # Prepare the model data for saving
            model_data = {
                'user_movie_matrix': self.user_movie_matrix,
                'movie_features': self.movie_features,
                'index': self.index,
                'movie_ids': self.movie_ids,
                'popularity_scores': self.popularity_scores,
                'cache': self.cache,
                'last_update_time': self.last_update_time
            }

            # Save the model using joblib
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, filepath: str):
        """
        Loads a saved recommender system from disk.

        Args:
            filepath: The path to load the model from.

        Returns:
            An instance of MovieRecommenderSystem loaded with the saved data.
        """
        try:
            # Load the model using joblib
            model_data = joblib.load(filepath)

            # Create a new instance of the recommender system
            instance = cls()

            # Update the instance attributes with the loaded data
            instance.user_movie_matrix = model_data['user_movie_matrix']
            instance.movie_features = model_data['movie_features']
            instance.index = model_data['index']
            instance.movie_ids = model_data['movie_ids']
            instance.popularity_scores = model_data['popularity_scores']
            instance.cache = model_data['cache']
            instance.last_update_time = model_data['last_update_time']
            instance.logger.info(f"Model loaded from {filepath}")
            
            return instance
        except Exception as e:
            instance = cls()
            instance.logger.error(f"Error loading model from {filepath}: {e}")
            return instance