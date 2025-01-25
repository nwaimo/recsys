import sys
import logging
import pandas as pd
from fuzzywuzzy import fuzz, process  
from typing import List, Dict
from .recommender import MovieRecommenderSystem
from .batch_processor import BatchProcessor
from .utils.logger import setup_logger
from datetime import datetime
from .evaluation import RecommenderEvaluator


# Configure logging using the centralized logger
logger = setup_logger(__name__)

class MovieRecommenderCLI:
    """
    Command-line interface for the Movie Recommender System.
    Provides functionalities for both users and administrators.
    """

    USER_MODE = "user" # Define constants for modes
    ADMIN_MODE = "admin"

    def __init__(self):
        """
        Initializes the MovieRecommenderCLI with proper dependency injection.
        """
        self.logger = setup_logger(__name__)
        self.logger.info("Initializing recommender system CLI...")

        # Create instances
        self.recommender = MovieRecommenderSystem()
        self.batch_processor = BatchProcessor()

        # Set up dependencies
        self.recommender.set_batch_processor(self.batch_processor)
        self.batch_processor.set_recommender(self.recommender)

        # Initialize the system
        try:
            self.recommender.initialize()
        except Exception as e:
            self.logger.critical(f"Failed to initialize recommender system: {e}", exc_info=True) # Log critical error with traceback
            print("Failed to initialize the recommender system. Check the logs for details.")
            sys.exit(1) # Exit if initialization fails

        self.mode = MovieRecommenderCLI.USER_MODE # Use constants for modes
        logger.info("Recommender system CLI initialized successfully in user mode.")

    def display_user_menu(self):
        """Updated menu options"""
        print(f"\nMovie Recommender System - User Mode ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})") # Added timestamp
        print("1. Search for a movie")
        print("2. Get similar movies")
        print("3. Get personalized recommendations")
        print("4. View top popular movies")
        print("5. Rate a movie")
        print("6. Switch to admin mode")
        print("7. Exit")
        print("\nEnter your choice (1-7): ")

    def display_admin_menu(self):
        """
        Displays the menu options for administrators.
        """
        print(f"\nMovie Recommender System - Admin Mode ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print("1. System status")
        print("2. Save model")
        print("3. Evaluate model")
        print("4. Clear cache")
        print("5. Switch to user mode")
        print("6. Exit")
        print("\nEnter your choice (1-6): ")

    def search_movies(self):
        """
        Enhanced movie search with multiple search options and fuzzy matching.
        Allows searching by title, genre, year, or a combination.
        """
        print("\nSearch Options:")
        print("1. Search by title")
        print("2. Search by genre")
        print("3. Search by year")
        print("4. Advanced search (combine criteria)")
        
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                self._search_by_title()
            elif choice == '2':
                self._search_by_genre()
            elif choice == '3':
                self._search_by_year()
            elif choice == '4':
                self._advanced_search()
            else:
                print("Invalid choice. Please try again.")
                
        except Exception as e:
            self.logger.error(f"Error during movie search: {e}", exc_info=True)
            print("An error occurred during search. Please try again.")

    def _search_by_title(self):
        """Search movies using fuzzy string matching on titles."""
        query = input("\nEnter movie title to search: ").strip()
        if not query:
            print("Please enter a valid search query.")
            return

        movies_df = self.recommender.data_loader.movies_df
        
        # Use fuzzy matching to find similar titles
        movie_titles = movies_df['title'].tolist()
        matches = process.extract(query, movie_titles, scorer=fuzz.token_sort_ratio, limit=10)
        
        matching_movies = movies_df[movies_df['title'].isin([m[0] for m in matches])]
        self._display_search_results(matching_movies, [m[1] for m in matches])

    def _search_by_genre(self):
        """Search movies by genre."""
        movies_df = self.recommender.data_loader.movies_df
        all_genres = set([genre for genres in movies_df['genres'] for genre in genres])
        
        print("\nAvailable genres:", ', '.join(sorted(all_genres)))
        genre = input("\nEnter genre: ").strip().capitalize()
        
        if genre not in all_genres:
            print(f"Genre '{genre}' not found. Please choose from the available genres.")
            return
            
        matching_movies = movies_df[movies_df['genres'].apply(lambda x: genre in x)]
        self._display_search_results(matching_movies)

    def _search_by_year(self):
        """Search movies by year or year range."""
        print("\nEnter year range (e.g., '1995' for single year or '1995-2000' for range)")
        year_input = input("Year(s): ").strip()
        
        try:
            if '-' in year_input:
                start_year, end_year = map(int, year_input.split('-'))
                matching_movies = self.recommender.data_loader.movies_df[
                    (self.recommender.data_loader.movies_df['year'] >= start_year) &
                    (self.recommender.data_loader.movies_df['year'] <= end_year)
                ]
            else:
                year = int(year_input)
                matching_movies = self.recommender.data_loader.movies_df[
                    self.recommender.data_loader.movies_df['year'] == year
                ]
            
            self._display_search_results(matching_movies)
            
        except ValueError:
            print("Invalid year format. Please use YYYY or YYYY-YYYY format.")

    def _advanced_search(self):
        """Combine multiple search criteria."""
        movies_df = self.recommender.data_loader.movies_df
        filtered_df = movies_df.copy()
        
        # Title search
        title = input("\nEnter title (or press Enter to skip): ").strip()
        if title:
            movie_titles = filtered_df['title'].tolist()
            matches = process.extract(title, movie_titles, scorer=fuzz.token_sort_ratio, limit=50)
            filtered_df = filtered_df[filtered_df['title'].isin([m[0] for m in matches])]
        
        # Genre search
        all_genres = set([genre for genres in movies_df['genres'] for genre in genres])
        print("\nAvailable genres:", ', '.join(sorted(all_genres)))
        genre = input("Enter genre (or press Enter to skip): ").strip().capitalize()
        if genre and genre in all_genres:
            filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: genre in x)]
        
        # Year search
        year_input = input("\nEnter year or range (e.g., 1995 or 1995-2000, or press Enter to skip): ").strip()
        if year_input:
            try:
                if '-' in year_input:
                    start_year, end_year = map(int, year_input.split('-'))
                    filtered_df = filtered_df[
                        (filtered_df['year'] >= start_year) &
                        (filtered_df['year'] <= end_year)
                    ]
                else:
                    year = int(year_input)
                    filtered_df = filtered_df[filtered_df['year'] == year]
            except ValueError:
                print("Invalid year format, skipping year filter.")
        
        self._display_search_results(filtered_df)

    def _display_search_results(self, matching_movies, similarity_scores: List[int] = None):
        """
        Display search results with optional similarity scores.
        
        Args:
            matching_movies: DataFrame containing the matched movies
            similarity_scores: Optional list of similarity scores for fuzzy matching results
        """
        if matching_movies.empty:
            print("\nNo movies found matching your search criteria.")
            return

        print(f"\nFound {len(matching_movies)} matching movies:")
        
        for idx, (_, movie) in enumerate(matching_movies.iterrows()):
            score_str = f" (Match: {similarity_scores[idx]}%)" if similarity_scores else ""
            print(
                f"ID: {movie['movie_id']} - "
                f"{movie['title']} "
                f"({int(movie['year']) if pd.notna(movie['year']) else 'N/A'}) - "
                f"Genres: {', '.join(movie['genres'])}"
                f"{score_str}"
            )

        if len(matching_movies) > 10:
            print("\nShowing top 10 results. Please refine your search for more specific results.")

    def evaluate_model(self):
        """Evaluates the model using comprehensive metrics."""
        try:
            print("\nStarting comprehensive evaluation...")
            evaluator = RecommenderEvaluator(self.recommender)
            results = evaluator.evaluate(k=10, n_folds=5)
            
            print("\nEvaluation Results:")
            for metric, values in results.items():
                print(f"\n{metric.upper()}:")
                print(f"  Mean: {values['mean']:.4f}")
                print(f"  Std:  {values['std']:.4f}")
                print(f"  95% CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]")
                
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}", exc_info=True)
            print("An error occurred during model evaluation. Check the logs for details.")

    def get_movie_recommendations(self):
        """
        Provides movie recommendations based on a given movie ID.
        """
        try:
            movie_id_input = input("\nEnter movie ID to get recommendations: ").strip()
            if not movie_id_input.isdigit(): # Input validation for movie ID
                print("Please enter a valid movie ID (number).")
                return
            movie_id = int(movie_id_input)

            # Check if movie_id exists
            if movie_id not in self.recommender.data_loader.get_movie_ids():
                print(f"Movie ID {movie_id} not found in the dataset.") # Inform user if movie ID is invalid
                return

            similar_movies = self.recommender.get_similar_movies(movie_id)

            if not similar_movies:
                print("No recommendations found for this movie.")
                return

            print("\nRecommended Movies (Similar to Movie ID {}):".format(movie_id)) # Added base movie id to output
            for movie in similar_movies:
                print(f"ID: {movie['movie_id']} - {movie['title']} ({int(movie['year']) if pd.notna(movie['year']) else 'N/A'}) - Genres: {','.join(movie['genres'])}") # Added year and genre display, handled NaN year

        except ValueError:
            print("Invalid input. Please enter a valid movie ID (number).")
        except Exception as e:
            logger.error(f"Error getting movie recommendations: {e}", exc_info=True) # Log traceback
            print("An unexpected error occurred while getting movie recommendations. Please check the logs.")

    def get_personalized_recommendations(self):
        """Gets personalized recommendations for a user"""
        try:
            user_id_input = input("\nEnter user ID: ").strip()
            if not user_id_input.isdigit(): # Input validation for user ID
                print("Please enter a valid user ID (number).")
                return
            user_id = int(user_id_input)

            recommendations = self.recommender.get_user_recommendations(user_id)

            if recommendations:
                print(f"\nPersonalized Recommendations for User ID {user_id}:") # Added user id to output
                for movie in recommendations:
                    print(f"ID: {movie['movie_id']} - {movie['title']} ({int(movie['year']) if pd.notna(movie['year']) else 'N/A'}) - "
                          f"Genres: {', '.join(movie['genres'])}") # Added year and genre display, handled NaN year, formatted genres
            else:
                print("\nNo personalized recommendations found. Displaying popular movies instead:")
                self.view_popular_movies()

        except ValueError:
            print("Invalid input. Please enter a valid user ID (number).")
        except Exception as e:
            self.logger.error(f"Error getting personalized recommendations: {e}", exc_info=True) # Log traceback
            print("An unexpected error occurred while getting personalized recommendations. Please check the logs.")

    def get_system_status(self):
        """
        Displays the current status of the recommender system (admin only).
        """
        status = self.recommender.get_system_status()
        print("\nSystem Status:")
        for key, value in status.items():
            print(f"{key}: {value}")
        batch_status = self.batch_processor.get_batch_status()
        print("\nBatch Processor Status:")
        for key, value in batch_status.items():
            print(f"{key}: {value}")

    def save_model(self):
        """
        Saves the current state of the recommender model (admin only).
        """
        filepath = input("\nEnter filepath to save the model (e.g., model.joblib): ").strip()
        if not filepath:
            print("Filepath cannot be empty.")
            return
        try:
            self.recommender.save_model(filepath)
            print(f"Model saved successfully to {filepath}!")
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {e}", exc_info=True) # Log filepath and traceback
            print(f"An unexpected error occurred while saving the model to {filepath}. Please check the logs.")

    def rate_movie(self):
        """
        Allows users to rate a movie.
        """
        try:
            user_id_input = input("\nEnter user ID: ").strip()
            movie_id_input = input("Enter movie ID: ").strip()
            rating_input = input("Enter rating (0.5-5.0): ").strip()

            if not (user_id_input.isdigit() and movie_id_input.isdigit()):
                print("User ID and Movie ID must be valid numbers.")
                return

            try:
                rating = float(rating_input)
            except ValueError:
                print("Rating must be a valid number between 0.5 and 5.0.")
                return

            if rating < 0.5 or rating > 5.0:
                print("Rating must be between 0.5 and 5.0.")
                return

            user_id = int(user_id_input)
            movie_id = int(movie_id_input)

            # Check if movie_id exists
            if movie_id not in self.recommender.data_loader.get_movie_ids():
                print(f"Movie ID {movie_id} not found in the dataset.") # Inform user if movie ID is invalid
                return

            self.recommender.add_new_rating(user_id, movie_id, rating)
            print("Rating added successfully!")

        except ValueError:
            print("Invalid input. Please enter valid numbers for user ID, movie ID, and rating.")
        except Exception as e:
            logger.error(f"Error adding rating: {e}", exc_info=True) # Log traceback
            print("An unexpected error occurred while adding the rating. Please check the logs.")

    def view_popular_movies(self):
        """Displays popular movies using the enhanced ranking method"""
        try:
            popular_movies = self.recommender.get_top_popular_movies(n_recommendations=10)

            if popular_movies:
                print("\nTop Popular Movies:")
                for movie in popular_movies:
                    print(f"ID: {movie['movie_id']} - {movie['title']} ({int(movie['year']) if pd.notna(movie['year']) else 'N/A'}) - "
                          f"Genres: {', '.join(movie['genres'])}") # Added year and genre display, handled NaN year, formatted genres
            else:
                print("No popular movies found.")

        except Exception as e:
            self.logger.error(f"Error displaying popular movies: {e}", exc_info=True) # Log traceback
            print("An unexpected error occurred while fetching popular movies. Please check the logs.")

    def clear_cache(self):
        """Clears the recommendation cache."""
        try:
            cache_size = len(self.recommender.cache)
            self.recommender.cache.clear()
            print(f"\nCache cleared successfully. {cache_size} items removed.")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}", exc_info=True)
            print("An error occurred while clearing the cache.")

    def run(self):
        """Updated main loop with new menu structure"""
        while True:
            try:
                if self.mode == MovieRecommenderCLI.USER_MODE: # Use constants for modes
                    self.display_user_menu()
                    choice = input().strip()

                    if choice == '1':
                        self.search_movies()
                    elif choice == '2':
                        self.get_movie_recommendations()
                    elif choice == '3':
                        self.get_personalized_recommendations()
                    elif choice == '4':
                        self.view_popular_movies()
                    elif choice == '5':
                        self.rate_movie()
                    elif choice == '6':
                        self.mode = MovieRecommenderCLI.ADMIN_MODE # Use constants for modes
                        print("\nSwitching to admin mode...")
                    elif choice == '7':
                        print("\nThank you for using the Movie Recommender System!")
                        sys.exit(0)
                    else:
                        print("\nInvalid choice. Please enter a number between 1 and 7.")
                elif self.mode == MovieRecommenderCLI.ADMIN_MODE: # Use constants for modes
                    self.display_admin_menu()
                    choice = input().strip()

                    if choice == '1':
                        self.get_system_status()
                    elif choice == '2':
                        self.save_model()
                    elif choice == '3':
                        self.evaluate_model()
                    elif choice == '4':
                        self.clear_cache()
                    elif choice == '5':
                        self.mode = MovieRecommenderCLI.USER_MODE # Use constants for modes
                        print("\nSwitching to user mode...")
                    elif choice == '6':
                        print("\nThank you for using the Movie Recommender System!")
                        sys.exit(0)
                    else:
                        print("\nInvalid choice. Please enter a number between 1 and 6.") # Updated choice range for admin menu
            except KeyboardInterrupt:
                 print("\nExiting the program.")
                 sys.exit(0)
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True) # Log traceback
                print(f"\nAn unexpected error occurred. Please check the logs for more details.")

def main():
    """
    Entry point for the Movie Recommender CLI application.
    """
    cli = MovieRecommenderCLI()
    cli.run()

if __name__ == "__main__":
    main()