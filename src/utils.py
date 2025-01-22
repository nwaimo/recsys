import json
import numpy as np
from datetime import datetime, timedelta

def display_user_menu():
    """Display the user menu options"""
    print("\nMovie Recommender System - User Menu")
    print("1. Get similar movies")
    print("2. Get personalized recommendations")
    print("3. Get genre recommendations")
    print("4. Add new rating")
    print("5. Switch to admin mode")
    print("6. Exit")

def display_admin_menu():
    """Display the admin menu options"""
    print("\nMovie Recommender System - Admin Menu")
    print("1. View analytics dashboard")
    print("2. View analytics plots")
    print("3. Save analytics")
    print("4. View batch status")
    print("5. Evaluate system")
    print("6. Save model")
    print("7. Switch to user mode")
    print("8. Exit")

def verify_admin_password(password):
    """Verify admin password"""
    # In a real system, this should use proper password hashing
    ADMIN_PASSWORD = "admin123"  # This should be stored securely in practice
    return password == ADMIN_PASSWORD

def evaluate_recommendations(recommender, test_users, n_recommendations=5):
    """Evaluate recommendation quality with improved metrics"""
    metrics = {
        'precision': [],
        'recall': [],
        'diversity': [],
        'novelty': [],
        'coverage': set(),
        'serendipity': []
    }
    
    total_movies = set(recommender.data_loader.movies_df['movie_id'].values)
    popular_movies = set(recommender.popularity_scores.nlargest(100).index)
    
    for user_id in test_users:
        # Get user's training and test sets
        user_ratings = recommender.data_loader.ratings_df[
            recommender.data_loader.ratings_df['user_id'] == user_id
        ]
        if len(user_ratings) < 5:  # Skip users with too few ratings
            continue
            
        # Split ratings by timestamp for more realistic evaluation
        user_ratings = user_ratings.sort_values('timestamp')
        train_ratings = user_ratings.iloc[:-5]
        test_ratings = user_ratings.iloc[-5:]
        
        test_items = set(test_ratings['movie_id'].values)
        train_items = set(train_ratings['movie_id'].values)
        
        # Get recommendations
        recs = recommender.get_user_recommendations(user_id, n_recommendations)
        if not recs:
            continue
            
        rec_ids = set(r['movie_id'] for r in recs)
        metrics['coverage'].update(rec_ids)
        
        # Calculate basic metrics
        hits = len(rec_ids & test_items)
        metrics['precision'].append(hits / len(rec_ids))
        metrics['recall'].append(hits / len(test_items))
        
        # Calculate diversity
        rec_genres = [set(r['genres'].split('|')) for r in recs]
        if rec_genres:
            genre_diversity = len(set.union(*rec_genres)) / len(set.intersection(*rec_genres)) if len(set.intersection(*rec_genres)) > 0 else 1.0
            metrics['diversity'].append(genre_diversity)
        
        # Calculate novelty and serendipity
        for mid in rec_ids:
            novelty = 1 / np.log2(2 + recommender.popularity_scores.get(mid, 0))
            metrics['novelty'].append(novelty)
            
            # Serendipity: recommendations that are both unexpected and relevant
            if mid in test_items and mid not in popular_movies:
                metrics['serendipity'].append(1)
            else:
                metrics['serendipity'].append(0)
    
    # Calculate final metrics
    evaluation_results = {
        'precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
        'recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
        'diversity': np.mean(metrics['diversity']) if metrics['diversity'] else 0,
        'novelty': np.mean(metrics['novelty']) if metrics['novelty'] else 0,
        'coverage': len(metrics['coverage']) / len(total_movies),
        'serendipity': np.mean(metrics['serendipity']) if metrics['serendipity'] else 0
    }
    
    # Add F1 score
    if evaluation_results['precision'] + evaluation_results['recall'] > 0:
        evaluation_results['f1_score'] = 2 * (evaluation_results['precision'] * evaluation_results['recall']) / (evaluation_results['precision'] + evaluation_results['recall'])
    else:
        evaluation_results['f1_score'] = 0
        
    return evaluation_results

def save_analytics_to_file(analytics_data, filepath):
    """Save analytics data to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(analytics_data, f, default=str, indent=2)

def format_time_delta(td):
    """Format a timedelta object into a readable string"""
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)
