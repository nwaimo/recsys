import threading
from datetime import datetime
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import psutil
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional
from .utils.logger import setup_logger

class AnalyticsMonitor:
    """
    Monitors various aspects of the Movie Recommender System, including user activity,
    system performance, and batch processing. Collects and analyzes data to provide
    insights and generate reports and visualizations.
    """

    def __init__(self):
        """
        Initializes the AnalyticsMonitor without direct dependencies.
        """
        self.recommender = None
        self.batch_processor = None
        self.metrics: Dict[str, Any] = {
            'rating_counts': Counter(),  # Count of ratings given
            'user_activity': {},  # {user_id: [timestamp1, timestamp2, ...]}
            'movie_popularity': Counter(),  # {movie_id: rating_count}
            'rating_distribution': Counter(),  # {rating_value: count}
            'response_times': [],  # List of response times for recommendations
            'system_metrics': [],  # List of {timestamp, cpu_usage, memory_usage}
            'error_counts': Counter(),  # {error_type: count}
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_processing_times': [], # List of batch processing times
            'batch_queue_lengths': [],  # List of batch queue lengths over time
            'recommendation_requests_count': 0 # Count of recommendation requests
        }
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        self.logger = setup_logger(__name__)
        
        self.monitoring_active = False
        self.monitoring_thread = None

    def set_recommender(self, recommender):
        """Set the recommender system via dependency injection"""
        self.recommender = recommender

    def set_batch_processor(self, batch_processor):
        """Set the batch processor via dependency injection"""
        self.batch_processor = batch_processor

    def update_metric(self, metric_name: str, value: Any, sub_key: Optional[str] = None):
        """
        Updates a metric with a new value.

        Args:
            metric_name: The name of the metric to update.
            value: The new value.
            sub_key: Optional sub-key for dictionary-based metrics.
        """
        with self.lock:
            if metric_name in self.metrics:
                if isinstance(self.metrics[metric_name], Counter):
                    if sub_key:
                        self.metrics[metric_name][sub_key] += value
                    else:
                        self.metrics[metric_name][value] += 1
                elif isinstance(self.metrics[metric_name], list):
                    self.metrics[metric_name].append(value)
                elif isinstance(self.metrics[metric_name], dict):
                    if sub_key:
                        if sub_key not in self.metrics[metric_name]:
                            self.metrics[metric_name][sub_key] = []
                        self.metrics[metric_name][sub_key].append(value)
                    else:
                        self.logger.warning(f"Metric '{metric_name}' is a dictionary, but no sub_key was provided.")
                else:
                    self.metrics[metric_name] = value
            else:
                self.logger.warning(f"Metric '{metric_name}' not found.")

    def record_rating(self, user_id: int, movie_id: int, rating: float):
        """
        Records a new rating, updating relevant metrics.

        Args:
            user_id: The ID of the user who gave the rating.
            movie_id: The ID of the movie that was rated.
            rating: The rating value.
        """
        timestamp = datetime.now()
        self.update_metric('rating_counts', 1)
        self.update_metric('user_activity', timestamp, sub_key=user_id)
        self.update_metric('movie_popularity', 1, sub_key=movie_id)
        self.update_metric('rating_distribution', rating)
        self.logger.debug(f"User {user_id} rated movie {movie_id} with {rating}")

    def record_response_time(self, response_time: float):
        """
        Records the response time of a recommendation request.

        Args:
            response_time: The response time in seconds.
        """
        self.update_metric('response_times', response_time)
        self.logger.debug(f"Recommendation response time: {response_time:.4f} seconds")
        
    def record_recommendation_request(self):
        """
        Records a recommendation request.
        """
        self.update_metric('recommendation_requests_count', 1)
        self.logger.debug("Recommendation request recorded.")

    def record_cache_hit(self):
        """Records a cache hit."""
        self.update_metric('cache_hits', 1)
        self.logger.debug("Cache hit")

    def record_cache_miss(self):
        """Records a cache miss."""
        self.update_metric('cache_misses', 1)
        self.logger.debug("Cache miss")

    def record_error(self, error_type: str):
        """
        Records an error.

        Args:
            error_type: The type of error that occurred.
        """
        self.update_metric('error_counts', 1, sub_key=error_type)
        self.logger.error(f"Error recorded: {error_type}")

    def record_batch_processing_time(self, processing_time: float):
        """
        Records the time taken to process a batch.

        Args:
            processing_time: The processing time in seconds.
        """
        self.update_metric('batch_processing_times', processing_time)
        self.logger.debug(f"Batch processing time: {processing_time:.4f} seconds")

    def record_batch_queue_length(self, queue_length: int):
        """
        Records the length of the batch queue.

        Args:
            queue_length: The length of the queue.
        """
        self.update_metric('batch_queue_lengths', queue_length)
        self.logger.debug(f"Batch queue length: {queue_length}")

    def start_monitoring(self):
        """Starts the system monitoring if not already active."""
        with self.lock:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitoring_thread = threading.Thread(target=self._monitor_system_metrics, daemon=True)
                self.monitoring_thread.start()
                self.logger.info("Analytics monitoring started")
            else:
                self.logger.debug("Monitoring already active")

    def _monitor_system_metrics(self):
        """
        Monitors system resource usage (CPU and memory) in a separate thread.
        """
        while self.monitoring_active:
            with self.lock:
                self.metrics['system_metrics'].append({
                    'timestamp': datetime.now(),
                    'cpu_usage': psutil.cpu_percent(interval=1),
                    'memory_usage': psutil.virtual_memory().percent
                })
            time.sleep(5)  # Collect metrics every 5 seconds

    def stop_monitoring(self):
        """Stops the system monitoring."""
        with self.lock:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=6)  # Wait up to 6 seconds for the thread to finish
                self.logger.info("Analytics monitoring stopped")

    def get_analytics_report(self) -> Dict[str, Any]:
        """
        Generates an analytics report.

        Returns:
            A dictionary containing various analytics metrics.
        """
        with self.lock:
            # Calculate some derived metrics
            total_ratings = sum(self.metrics['rating_counts'].values())
            active_users = len(self.metrics['user_activity'])
            popular_movies = self.metrics['movie_popularity'].most_common(10)
            avg_response_time = np.mean(self.metrics['response_times'][-1000:]) if self.metrics['response_times'] else 0
            cache_hit_ratio = self.cache_hit_ratio()
            error_rate = sum(self.metrics['error_counts'].values()) / total_ratings if total_ratings > 0 else 0
            avg_batch_processing_time = np.mean(self.metrics['batch_processing_times']) if self.metrics['batch_processing_times'] else 0
            avg_batch_queue_length = np.mean(self.metrics['batch_queue_lengths']) if self.metrics['batch_queue_lengths'] else 0

            report = {
                'uptime': str(datetime.now() - self.start_time),
                'total_ratings': total_ratings,
                'recommendation_requests': self.metrics['recommendation_requests_count'],
                'active_users': active_users,
                'popular_movies': dict(popular_movies),
                'rating_distribution': dict(self.metrics['rating_distribution']),
                'avg_response_time': avg_response_time,
                'cache_hit_ratio': cache_hit_ratio,
                'error_rate': error_rate,
                'avg_batch_processing_time': avg_batch_processing_time,
                'avg_batch_queue_length': avg_batch_queue_length
            }
            self.logger.info("Analytics report generated.")
            return report

    def cache_hit_ratio(self) -> float:
        """
        Calculates the cache hit ratio.

        Returns:
            The cache hit ratio (0.0 to 1.0).
        """
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        return self.metrics['cache_hits'] / total if total > 0 else 0.0

    def plot_analytics(self):
        """
        Generates and displays plots for various analytics metrics.
        """
        with self.lock:
            plt.figure(figsize=(18, 12))

            # Rating distribution
            plt.subplot(3, 3, 1)
            ratings = self.metrics['rating_distribution']
            if ratings:
                plt.bar(ratings.keys(), ratings.values())
                plt.title('Rating Distribution')
                plt.xlabel('Rating')
                plt.ylabel('Count')
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                plt.title('Rating Distribution')

            # User activity over time (simplified for plotting)
            plt.subplot(3, 3, 2)
            if self.metrics['user_activity']:
                user_activity_counts = Counter()
                for user_id, timestamps in self.metrics['user_activity'].items():
                    for ts in timestamps:
                        user_activity_counts[ts.strftime('%Y-%m-%d')] += 1

                dates, counts = zip(*sorted(user_activity_counts.items()))

                plt.plot(dates, counts)
                plt.title('User Activity (Daily)')
                plt.xlabel('Date')
                plt.ylabel('Number of Active Users')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                plt.title('User Activity (Daily)')

            # System metrics
            plt.subplot(3, 3, 3)
            if self.metrics['system_metrics']:
                system_metrics = pd.DataFrame(self.metrics['system_metrics'])
                system_metrics.set_index('timestamp', inplace=True)
                system_metrics[['cpu_usage', 'memory_usage']].plot(ax=plt.gca())
                plt.title('System Resource Usage')
                plt.xlabel('Time')
                plt.ylabel('Usage (%)')
            else:
                 plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                 plt.title('System Resource Usage')

            # Response times
            plt.subplot(3, 3, 4)
            if self.metrics['response_times']:
                plt.hist(self.metrics['response_times'], bins=50)
                plt.title('Response Time Distribution')
                plt.xlabel('Response Time (seconds)')
                plt.ylabel('Count')
            else:
                 plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                 plt.title('Response Time Distribution')

            # Movie popularity
            plt.subplot(3, 3, 5)
            if self.metrics['movie_popularity']:
                top_movies = self.metrics['movie_popularity'].most_common(10)
                movie_ids, counts = zip(*top_movies)
                
                # Assuming you can get movie titles from movie_ids...
                try:
                    movie_titles = [self.recommender.data_loader.movies_df.loc[self.recommender.data_loader.movies_df['movie_id'] == mid, 'title'].iloc[0] for mid in movie_ids]
                except:
                    movie_titles = movie_ids
                
                plt.barh(movie_titles, counts)
                plt.title('Top 10 Most Popular Movies')
                plt.xlabel('Number of Ratings')
                plt.ylabel('Movie')
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                plt.title('Top 10 Most Popular Movies')

            # Error counts
            plt.subplot(3, 3, 6)
            if self.metrics['error_counts']:
                error_types, counts = zip(*self.metrics['error_counts'].items())
                plt.bar(error_types, counts)
                plt.title('Error Counts')
                plt.xlabel('Error Type')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                plt.title('Error Counts')
                
            
            # Batch processing times
            plt.subplot(3, 3, 7)
            if self.metrics['batch_processing_times']:
                plt.hist(self.metrics['batch_processing_times'], bins=20)
                plt.title('Batch Processing Time Distribution')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Count')
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                plt.title('Batch Processing Time Distribution')

            # Batch queue lengths
            plt.subplot(3, 3, 8)
            if self.metrics['batch_queue_lengths']:
                plt.plot(self.metrics['batch_queue_lengths'])
                plt.title('Batch Queue Length Over Time')
                plt.xlabel('Time')
                plt.ylabel('Queue Length')
            else:
                plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                plt.title('Batch Queue Length Over Time')
            
            # Recommendation request counts
            plt.subplot(3, 3, 9)
            plt.bar(['Recommendation Requests'], [self.metrics['recommendation_requests_count']])
            plt.title('Recommendation Requests Count')
            plt.ylabel('Count')

            plt.tight_layout()
            self.logger.info("Analytics plots generated.")
            plt.show()

    def reset_metrics(self):
        """Resets all metrics to their initial state."""
        with self.lock:
            self.metrics = {
                'rating_counts': Counter(),
                'user_activity': {},
                'movie_popularity': Counter(),
                'rating_distribution': Counter(),
                'response_times': [],
                'system_metrics': [],
                'error_counts': Counter(),
                'cache_hits': 0,
                'cache_misses': 0,
                'batch_processing_times': [],
                'batch_queue_lengths': [],
                'recommendation_requests_count': 0
            }
            self.start_time = datetime.now()
            self.logger.info("Analytics metrics reset.")
