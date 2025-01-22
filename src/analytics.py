import threading
from datetime import datetime
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import psutil
import pandas as pd

class AnalyticsMonitor:
    def __init__(self):
        self.metrics = {
            'rating_counts': Counter(),
            'user_activity': Counter(),
            'movie_popularity': Counter(),
            'rating_distribution': Counter(),
            'response_times': [],
            'system_metrics': [],
            'error_counts': Counter(),
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.start_time = datetime.now()
        self.lock = threading.Lock()

    def update_metric(self, metric_name, value):
        with self.lock:
            if isinstance(self.metrics[metric_name], Counter):
                self.metrics[metric_name].update([value])
            elif isinstance(self.metrics[metric_name], list):
                self.metrics[metric_name].append(value)

    def get_analytics_report(self):
        with self.lock:
            return {
                'uptime': str(datetime.now() - self.start_time),
                'total_ratings': sum(self.metrics['rating_counts'].values()),
                'active_users': len(self.metrics['user_activity']),
                'popular_movies': dict(self.metrics['movie_popularity'].most_common(10)),
                'rating_distribution': dict(self.metrics['rating_distribution']),
                'avg_response_time': np.mean(self.metrics['response_times'][-1000:]) if self.metrics['response_times'] else 0,
                'cache_hit_ratio': self.cache_hit_ratio(),
                'error_rate': sum(self.metrics['error_counts'].values())
            }

    def cache_hit_ratio(self):
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        return self.metrics['cache_hits'] / total if total > 0 else 0

    def plot_analytics(self):
        plt.figure(figsize=(15, 10))

        # Rating distribution
        plt.subplot(2, 2, 1)
        ratings = self.metrics['rating_distribution']
        plt.bar(ratings.keys(), ratings.values())
        plt.title('Rating Distribution')

        # User activity over time
        plt.subplot(2, 2, 2)
        user_activity = pd.Series(self.metrics['user_activity'])
        user_activity.plot(kind='line')
        plt.title('User Activity Over Time')

        # System metrics
        plt.subplot(2, 2, 3)
        system_metrics = pd.DataFrame(self.metrics['system_metrics'])
        system_metrics[['cpu_usage', 'memory_usage']].plot()
        plt.title('System Resource Usage')

        # Response times
        plt.subplot(2, 2, 4)
        plt.hist(self.metrics['response_times'], bins=50)
        plt.title('Response Time Distribution')

        plt.tight_layout()
        return plt
