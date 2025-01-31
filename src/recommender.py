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

    def set_batch_processor(self, batch_processor):
        """Set the batch processor via dependency injection"""
        self.batch_processor = batch_processor

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
        """Modified to remove analytics"""
        try:
            movie_idx = self.movie_ids.get_loc(movie_id)
            movie_vector = self.movie_features[movie_idx].reshape(1, -1)
            distances, indices = self.index.search(movie_vector, n_neighbors + 1)
            
            # Get content-based recommendations using TF-IDF
            movie_tfidf_idx = self.data_loader.movies_df[self.data_loader.movies_df['movie_id'] == movie_id].index[0]
            movie_tfidf = self.data_loader.tfidf_matrix[movie_tfidf_idx]
            tfidf_similarities = self.data_loader.tfidf_matrix.dot(movie_tfidf.T).toarray().flatten()
            tfidf_similar_indices = tfidf_similarities.argsort()[-n_neighbors-1:-1][::-1]
            
            similar_movie_indices = list(set(indices[0][1:]).union(set(tfidf_similar_indices)))[:n_neighbors]
            similar_movie_ids = [self.data_loader.movies_df.iloc[idx]['movie_id'] for idx in similar_movie_indices]
            similar_movies = [self.data_loader.get_movie_info(mid) for mid in similar_movie_ids]
            return similar_movies

        except Exception as e:
            self.logger.error(f"Error getting similar movies: {e}")
            return []

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Fixed user recommendations method"""
        try:
            # Check cache first
            if user_id in self.cache and time.time() - self.cache[user_id]['timestamp'] < self.update_interval.total_seconds():
                return self.cache[user_id]['recommendations']

            # Get user's ratings
            user_ratings = self.data_loader.ratings_df[self.data_loader.ratings_df['user_id'] == user_id]
            
            if user_ratings.empty:
                # If no ratings, return popular movies
                return self.get_top_popular_movies(n_recommendations)

            # Get movies similar to user's highly rated movies
            rated_movies = user_ratings.sort_values('rating', ascending=False)
            top_rated = rated_movies.head(3)  # Use top 3 rated movies
            
            similar_movies = []
            for _, movie in top_rated.iterrows():
                similar = self.get_similar_movies(movie['movie_id'], n_neighbors=5)
                similar_movies.extend(similar)

            # Remove duplicates and already rated movies
            rated_movie_ids = set(user_ratings['movie_id'])
            unique_recommendations = []
            seen_ids = set()

            for movie in similar_movies:
                movie_id = movie['movie_id']
                if movie_id not in rated_movie_ids and movie_id not in seen_ids:
                    unique_recommendations.append(movie)
                    seen_ids.add(movie_id)

            recommendations = unique_recommendations[:n_recommendations]

            # Cache the results
            self.cache[user_id] = {
                'recommendations': recommendations,
                'timestamp': time.time()
            }

            return recommendations

        except Exception as e:
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
        
        try:
            # Filter movies that have at least one of the specified genres
            genre_movies = self.data_loader.movies_df[
                self.data_loader.movies_df['genres'].apply(lambda x: any(g in x for g in genres))
            ]

            # If no movies found for the specified genre, return an empty list
            if (genre_movies.empty):
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
            self.logger.info(f"Genre recommendations generated in {response_time:.4f} seconds")
            
            return recommendation_list

        except Exception as e:
            self.logger.error(f"Error getting genre recommendations: {e}")
            return []

    def add_new_rating(self, user_id: int, movie_id: int, rating: float) -> bool:
        """
        Add a new rating to the system.
        
        Args:
            user_id: The ID of the user
            movie_id: The ID of the movie
            rating: The rating value (between 0.5 and 5.0)
            
        Returns:
            bool: True if rating was successfully added, False otherwise
        """
        try:
            # Validate inputs
            if not isinstance(rating, (int, float)) or rating < 0.5 or rating > 5.0:
                self.logger.error(f"Invalid rating value: {rating}")
                return False
                
            if movie_id not in self.data_loader.get_movie_ids():
                self.logger.error(f"Invalid movie ID: {movie_id}")
                return False

            # Create rating data
            rating_data = {
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': float(rating),
                'timestamp': int(time.time())
            }

            # Add to batch processor
            if self.batch_processor:
                self.batch_processor.add_to_batch(rating_data)
                self.logger.info(f"New rating added to batch: User {user_id}, Movie {movie_id}, Rating {rating}")
                return True
            else:
                self.logger.error("Batch processor not initialized")
                return False

        except Exception as e:
            self.logger.error(f"Error adding new rating: {e}", exc_info=True)
            return False

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
            self.logger.error(f"Error updating feature matrices: {e}")
        finally:
            self.is_updating = False
            self.update_lock.release()

 
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