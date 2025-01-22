import numpy as np
import faiss
import joblib
import json
import logging
import threading
import time
import psutil
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
from sklearn.preprocessing import StandardScaler

from .analytics import AnalyticsMonitor
from .batch_processor import BatchProcessor
from .data_loader import DataLoader

class MovieRecommenderSystem:
    def __init__(self):
        # Initialize components
        self.data_loader = DataLoader()
        self.analytics = AnalyticsMonitor()
        self.batch_processor = BatchProcessor(self)
        
        # Core data structures
        self.user_movie_matrix = None
        self.movie_features = None
        self.index = None
        self.movie_ids = None
        self.popularity_scores = None
        
        # Cache and real-time components
        self.cache = {}
        self.new_ratings_queue = deque(maxlen=1000)
        self.update_lock = threading.Lock()
        self.last_update_time = None
        self.update_interval = timedelta(minutes=30)
        self.is_updating = False
        
        self.setup_logging()
        self.start_monitoring()

    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('recommender.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """Start system monitoring thread"""
        def monitor_system():
            while True:
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                }
                self.analytics.metrics['system_metrics'].append(metrics)
                time.sleep(60)

        threading.Thread(target=monitor_system, daemon=True).start()

    def initialize(self):
        """Initialize the recommender system"""
        self.data_loader.download_dataset()
        self.data_loader.load_data()
        self.create_feature_matrices()
        self.build_ann_index()

    def create_feature_matrices(self):
        """Create feature matrices for recommendations"""
        self.logger.info("Creating feature matrices...")
        
        # Get user-movie matrix and normalized features
        self.user_movie_matrix, self.movie_features = self.data_loader.get_user_movie_matrix()
        self.movie_ids = self.user_movie_matrix.index
        
        # Calculate popularity scores
        self.calculate_popularity_scores()

    def calculate_popularity_scores(self):
        """Calculate movie popularity scores"""
        current_timestamp = self.data_loader.ratings_df['timestamp'].max()
        
        rating_stats = self.data_loader.ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean'],
            'timestamp': 'max'
        }).reset_index()
        
        rating_stats.columns = ['movie_id', 'rating_count', 'rating_mean', 'last_rated']
        rating_stats['time_weight'] = 1 / (1 + np.log1p((current_timestamp - rating_stats['last_rated']) / (24 * 60 * 60)))
        rating_stats['popularity_score'] = (
            rating_stats['rating_count'] * rating_stats['rating_mean'] * rating_stats['time_weight']
        )
        
        self.popularity_scores = rating_stats.set_index('movie_id')['popularity_score']

    def build_ann_index(self):
        """Build FAISS index for similarity search"""
        self.logger.info("Building FAISS index...")
        n_features = self.movie_features.shape[1]
        
        # Convert features to float32 as required by FAISS
        self.movie_features = np.ascontiguousarray(self.movie_features.astype('float32'))
        
        # Create and build the index
        self.index = faiss.IndexFlatL2(n_features)
        self.index.add(self.movie_features)

    def get_similar_movies(self, movie_id, n_neighbors=5):
        """Get similar movies based on collaborative filtering"""
        start_time = time.time()
        
        try:
            movie_idx = self.movie_ids.get_loc(movie_id)
            # Get movie vector and reshape for FAISS
            movie_vector = self.movie_features[movie_idx].reshape(1, -1)
            
            # Search using FAISS
            D, I = self.index.search(movie_vector, n_neighbors + 1)
            similar_idx = I[0][1:]  # Skip the first result as it's the query movie itself
            similar_movies = self.movie_ids[similar_idx]

            results = []
            for mid in similar_movies:
                results.append(self.data_loader.get_movie_info(mid))

            self.analytics.update_metric('response_times', time.time() - start_time)
            return results
        except Exception as e:
            self.analytics.update_metric('error_counts', 'similar_movies_error')
            self.logger.error(f"Error getting similar movies: {str(e)}")
            return None

    def evaluate_recommendations(self, test_users=None, k=10):
        """Evaluate recommender system performance using multiple metrics
        
        Args:
            test_users: List of user IDs to evaluate on. If None, uses all users
            k: Number of recommendations to generate for evaluation
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if test_users is None:
            test_users = self.data_loader.ratings_df['user_id'].unique()
            # Sample 1000 users for efficiency if dataset is large
            if len(test_users) > 1000:
                test_users = np.random.choice(test_users, 1000, replace=False)
        
        metrics = {
            'ndcg': [],
            'precision': [],
            'recall': [],
            'map': []
        }
        
        for user_id in test_users:
            # Split user ratings into train/test
            user_ratings = self.data_loader.ratings_df[
                self.data_loader.ratings_df['user_id'] == user_id
            ].sort_values('timestamp')
            
            if len(user_ratings) < 5:  # Skip users with too few ratings
                continue
                
            train_ratings = user_ratings.iloc[:-5]  # Hold out last 5 ratings
            test_ratings = user_ratings.iloc[-5:]
            test_movies = set(test_ratings['movie_id'])
            
            # Get recommendations based on training data
            recommendations = self.get_user_recommendations(user_id, n_recommendations=k)
            if not recommendations:
                continue
            
            rec_movies = [r['movie_id'] for r in recommendations]
            
            # Calculate metrics
            metrics['ndcg'].append(self._calculate_ndcg(test_movies, rec_movies, k))
            precision, recall = self._calculate_precision_recall(test_movies, rec_movies)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['map'].append(self._calculate_map(test_movies, rec_movies))
        
        # Average metrics across users
        return {
            'ndcg@k': np.mean(metrics['ndcg']),
            'precision@k': np.mean(metrics['precision']),
            'recall@k': np.mean(metrics['recall']),
            'map@k': np.mean(metrics['map'])
        }
    
    def _calculate_ndcg(self, actual, predicted, k):
        """Calculate Normalized Discounted Cumulative Gain"""
        dcg = 0
        idcg = 0
        
        for i, item in enumerate(predicted[:k]):
            if item in actual:
                dcg += 1 / np.log2(i + 2)
        
        for i in range(min(len(actual), k)):
            idcg += 1 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0
    
    def _calculate_precision_recall(self, actual, predicted):
        """Calculate Precision and Recall"""
        n_rel = len(actual)
        n_rec = len(predicted)
        n_rel_and_rec = len(actual.intersection(set(predicted)))
        
        precision = n_rel_and_rec / n_rec if n_rec > 0 else 0
        recall = n_rel_and_rec / n_rel if n_rel > 0 else 0
        
        return precision, recall
    
    def _calculate_map(self, actual, predicted):
        """Calculate Mean Average Precision"""
        hits = 0
        sum_prec = 0
        
        for i, item in enumerate(predicted):
            if item in actual:
                hits += 1
                sum_prec += hits / (i + 1)
                
        return sum_prec / len(actual) if len(actual) > 0 else 0

    def get_user_recommendations(self, user_id, n_recommendations=5):
        """Get personalized recommendations for a user"""
        try:
            if user_id in self.cache:
                self.analytics.metrics['cache_hits'] += 1
                return self.cache[user_id]['recommendations']
            
            self.analytics.metrics['cache_misses'] += 1
            
            user_ratings = self.data_loader.ratings_df[
                self.data_loader.ratings_df['user_id'] == user_id
            ]
            if len(user_ratings) == 0:
                return None

            user_avg_rating = user_ratings['rating'].mean()
            rated_movies = user_ratings['movie_id'].values
            top_rated = user_ratings[user_ratings['rating'] >= user_avg_rating]['movie_id'].values

            recommendations = []
            for movie_id in top_rated:
                similar_movies = self.get_similar_movies(movie_id, n_neighbors=3)
                if similar_movies:
                    recommendations.extend(similar_movies)

            recommendations = [r for r in recommendations if r['movie_id'] not in rated_movies]
            recommendations = list({r['movie_id']: r for r in recommendations}.values())

            # Cache the results
            self.cache[user_id] = {
                'recommendations': recommendations[:n_recommendations],
                'timestamp': time.time()
            }

            return recommendations[:n_recommendations]

        except Exception as e:
            self.analytics.update_metric('error_counts', 'recommendation_error')
            self.logger.error(f"Error getting recommendations: {str(e)}")
            return None

    def get_genre_recommendations(self, genres, n_recommendations=5):
        """Get recommendations based on genres"""
        genre_movies = self.data_loader.movies_df[
            self.data_loader.movies_df['genres'].apply(lambda x: any(g in x for g in genres))
        ]

        movie_stats = self.data_loader.ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).reset_index()

        movie_stats.columns = ['movie_id', 'rating_count', 'rating_mean']
        recommendations = genre_movies.merge(movie_stats, on='movie_id')
        recommendations = recommendations.sort_values(
            ['rating_count', 'rating_mean'],
            ascending=[False, False]
        )

        return recommendations.head(n_recommendations)

    def add_new_rating(self, user_id, movie_id, rating, timestamp=None):
        """Add a new rating with analytics and batch processing"""
        start_time = time.time()

        if timestamp is None:
            timestamp = int(time.time())

        new_rating = {
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': timestamp
        }

        # Update analytics
        self.analytics.update_metric('rating_counts', 1)
        self.analytics.update_metric('user_activity', user_id)
        self.analytics.update_metric('movie_popularity', movie_id)
        self.analytics.update_metric('rating_distribution', rating)

        # Add to batch processor
        self.batch_processor.add_to_batch(new_rating)

        # Update response time
        self.analytics.update_metric('response_times', time.time() - start_time)

    def update_feature_matrices(self, new_ratings_df):
        """Update feature matrices with new ratings"""
        new_pivot = new_ratings_df.pivot(
            index='movie_id',
            columns='user_id',
            values='rating'
        ).fillna(0)

        self.user_movie_matrix = pd.concat([
            self.user_movie_matrix,
            new_pivot
        ]).groupby(level=0).sum()

        scaler = StandardScaler()
        self.movie_features = scaler.fit_transform(self.user_movie_matrix)
        self.movie_ids = self.user_movie_matrix.index

    def get_analytics_dashboard(self):
        """Generate analytics dashboard"""
        report = self.analytics.get_analytics_report()
        batch_status = self.batch_processor.get_batch_status()
        system_status = self.get_system_status()

        dashboard = {
            'analytics': report,
            'batch_processing': batch_status,
            'system_status': system_status
        }

        return dashboard

    def get_system_status(self):
        """Get current system status"""
        return {
            'total_ratings': len(self.data_loader.ratings_df),
            'total_users': len(self.data_loader.ratings_df['user_id'].unique()),
            'total_movies': len(self.data_loader.ratings_df['movie_id'].unique()),
            'pending_updates': len(self.new_ratings_queue),
            'last_update_time': self.last_update_time,
            'is_updating': self.is_updating,
            'cache_size': len(self.cache)
        }

    def save_model(self, filepath):
        """Save the model to disk"""
        joblib.dump(self, filepath)

    def cross_validate(self, k_folds=5, n_recommendations=10):
        """Perform k-fold cross validation
        
        Args:
            k_folds: Number of folds for cross validation
            n_recommendations: Number of recommendations to generate
            
        Returns:
            dict: Average metrics across folds
        """
        users = self.data_loader.ratings_df['user_id'].unique()
        np.random.shuffle(users)
        fold_size = len(users) // k_folds
        
        metrics_per_fold = []
        
        for i in range(k_folds):
            self.logger.info(f"Evaluating fold {i+1}/{k_folds}")
            
            # Split users into test and train
            test_users = users[i * fold_size:(i + 1) * fold_size]
            
            # Evaluate on this fold
            fold_metrics = self.evaluate_recommendations(
                test_users=test_users,
                k=n_recommendations
            )
            metrics_per_fold.append(fold_metrics)
        
        # Calculate average metrics across folds
        avg_metrics = {}
        for metric in metrics_per_fold[0].keys():
            values = [fold[metric] for fold in metrics_per_fold]
            avg_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return avg_metrics

    def save_analytics(self, filepath):
        """Save analytics data to file"""
        dashboard = self.get_analytics_dashboard()
        with open(filepath, 'w') as f:
            json.dump(dashboard, f, default=str)

    @classmethod
    def load_model(cls, filepath):
        """Load the model from disk"""
        return joblib.load(filepath)
