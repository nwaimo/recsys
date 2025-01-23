import sys
import logging
import pandas as pd
from .recommender import MovieRecommenderSystem
from .analytics import AnalyticsMonitor
from .batch_processor import BatchProcessor
from .utils.logger import setup_logger

# Configure logging using the centralized logger
logger = setup_logger(__name__)

class MovieRecommenderCLI:
    """
    Command-line interface for the Movie Recommender System.
    Provides functionalities for both users and administrators.
    """

    def __init__(self):
        """
        Initializes the MovieRecommenderCLI with proper dependency injection.
        """
        self.logger = setup_logger(__name__)
        self.logger.info("Initializing recommender system...")
        
        # Create instances
        self.recommender = MovieRecommenderSystem()
        self.analytics = AnalyticsMonitor()
        self.batch_processor = BatchProcessor()
        
        # Set up dependencies
        self.recommender.set_analytics(self.analytics)
        self.recommender.set_batch_processor(self.batch_processor)
        self.analytics.set_recommender(self.recommender)
        self.analytics.set_batch_processor(self.batch_processor)
        self.batch_processor.set_recommender(self.recommender)
        
        # Initialize the system
        self.recommender.initialize()
        
        # Start monitoring automatically
        self.analytics.start_monitoring()
        self.logger.info("Analytics monitoring started automatically")
        
        self.mode = "user"
        logger.info("Recommender system initialized successfully")

    def display_user_menu(self):
        """
        Displays the main menu options for users.
        """
        print("\nMovie Recommender System - User Mode")
        print("1. Search for a movie")
        print("2. Get movie recommendations")
        print("3. Get personalized recommendations")
        print("4. Get genre recommendations")
        print("5. Rate a movie")
        print("6. View popular movies")
        print("7. Switch to admin mode")
        print("8. Exit")
        print("\nEnter your choice (1-8): ")

    def display_admin_menu(self):
        """
        Displays the menu options for administrators.
        """
        print("\nMovie Recommender System - Admin Mode")
        print("1. System status")
        print("2. Save model")
        print("3. Get analytics report")
        print("4. Plot analytics")
        print("5. Start monitoring")
        print("6. Switch to user mode")
        print("7. Exit")
        print("\nEnter your choice (1-7): ")

    def search_movies(self):
        """
        Allows users to search for movies by title.
        """
        query = input("\nEnter movie title to search: ").strip()
        if not query:
            print("Please enter a valid search query.")
            return

        movies_df = self.recommender.data_loader.movies_df
        matching_movies = movies_df[movies_df['title'].str.lower().str.contains(query.lower())]

        if matching_movies.empty:
            print("No movies found matching your search.")
            return

        print("\nSearch Results:")
        for idx, movie in matching_movies.iterrows():
            print(f"ID: {movie['movie_id']} - {movie['title']} ({movie['year']}) - Genres: {','.join(movie['genres'])}")

    def get_movie_recommendations(self):
        """
        Provides movie recommendations based on a given movie ID.
        """
        try:
            movie_id = int(input("\nEnter movie ID to get recommendations: "))
            similar_movies = self.recommender.get_similar_movies(movie_id)

            if not similar_movies:
                print("No recommendations found for this movie.")
                return

            print("\nRecommended Movies:")
            for movie in similar_movies:
                print(f"ID: {movie['movie_id']} - {movie['title']} ({movie['year']}) - Genres: {','.join(movie['genres'])}")

        except ValueError:
            print("Please enter a valid movie ID (number).")
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            print(f"An unexpected error occurred while getting recommendations. Please check the logs for more details.")

    def get_personalized_recommendations(self):
        """
        Provides personalized movie recommendations for a given user ID.
        """
        try:
            user_id = int(input("\nEnter user ID: "))
            recommendations = self.recommender.get_user_recommendations(user_id)

            if not recommendations:
                print("No personalized recommendations found for this user. Please make sure the user has rated some movies.")
                return

            print("\nPersonalized Recommendations:")
            for movie in recommendations:
                print(f"ID: {movie['movie_id']} - {movie['title']} ({movie['year']}) - Genres: {','.join(movie['genres'])}")

        except ValueError:
            print("Please enter a valid user ID (number).")
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            print(f"An unexpected error occurred while getting personalized recommendations. Please check the logs for more details.")

    def get_genre_recommendations(self):
        """
        Provides movie recommendations based on a selected genre.
        """
        print("\nAvailable genres:")
        all_genres = set()
        for genres in self.recommender.data_loader.movies_df['genres']:
            all_genres.update(genres)
        print(", ".join(sorted(all_genres)))

        genre = input("\nEnter genre: ").strip()
        recommendations = self.recommender.get_genre_recommendations([genre])

        if recommendations.empty:
            print("No recommendations found for this genre.")
            return

        print("\nTop Movies in Genre:")
        for _, movie in recommendations.iterrows():
            print(f"ID: {movie['movie_id']} - {movie['title']} ({movie['year']}) - Genres: {','.join(movie['genres'])}")

    def get_system_status(self):
        """
        Displays the current status of the recommender system (admin only).
        """
        status = self.recommender.get_system_status()
        print("\nSystem Status:")
        for key, value in status.items():
            print(f"{key}: {value}")

    def save_model(self):
        """
        Saves the current state of the recommender model (admin only).
        """
        try:
            self.recommender.save_model()
            print("Model saved successfully!")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            print(f"An unexpected error occurred while saving the model. Please check the logs for more details.")

    def rate_movie(self):
        """
        Allows users to rate a movie.
        """
        try:
            user_id = int(input("\nEnter user ID: "))
            movie_id = int(input("Enter movie ID: "))
            rating = float(input("Enter rating (0.5-5.0): "))

            if rating < 0.5 or rating > 5.0:
                print("Rating must be between 0.5 and 5.0.")
                return

            self.recommender.add_new_rating(user_id, movie_id, rating)
            print("Rating added successfully!")

        except ValueError:
            print("Please enter valid numbers for user ID, movie ID, and rating.")
        except Exception as e:
            logger.error(f"Error adding rating: {e}")
            print(f"An unexpected error occurred while adding the rating. Please check the logs for more details.")

    def view_popular_movies(self):
        """
        Displays the top 10 popular movies based on average ratings and rating count.
        """
        try:
            ratings_df = self.recommender.data_loader.ratings_df
            movies_df = self.recommender.data_loader.movies_df

            # Calculate average ratings and number of ratings
            movie_stats = ratings_df.groupby('movie_id').agg({
                'rating': ['mean', 'count']
            })
            movie_stats.columns = ['avg_rating', 'rating_count']  # Flatten the multi-level column index
            movie_stats = movie_stats.reset_index()

            # Filter movies with at least 100 ratings
            popular_movies = movie_stats[movie_stats['rating_count'] >= 100]

            # Sort by average rating
            popular_movies = popular_movies.sort_values('avg_rating', ascending=False)

            # Merge with movies_df to get movie titles
            popular_movies = pd.merge(popular_movies, movies_df[['movie_id', 'title', 'year']], on='movie_id')

            print("\nTop 10 Popular Movies (with at least 100 ratings):")
            for index, row in popular_movies.head(10).iterrows():
                print(f"ID: {row['movie_id']} - {row['title']} ({row['year']}) - "
                      f"Average Rating: {row['avg_rating']:.2f} ({row['rating_count']} ratings)")

        except Exception as e:
            logger.error(f"Error displaying popular movies: {e}")
            print(f"An unexpected error occurred while fetching popular movies. Please check the logs for more details.")

    def start_monitoring(self):
        """
        Starts the monitoring process for the recommender system (admin only).
        """
        try:
            self.recommender.start_monitoring()
            print("Monitoring started successfully!")
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            print(f"An unexpected error occurred while starting monitoring. Please check the logs for more details.")

    def get_analytics_report(self):
        """
        Retrieves and displays an analytics report (admin only).
        """
        analytics = AnalyticsMonitor()
        report = analytics.get_analytics_report()
        print("\nAnalytics Report:")
        for key, value in report.items():
            print(f"{key}: {value}")

    def plot_analytics(self):
        """
        Generates and displays plots for analytics data (admin only).
        """
        analytics = AnalyticsMonitor()
        analytics.plot_analytics()

    def run(self):
        """
        Main loop for the CLI, handling user input and dispatching actions.
        """
        while True:
            try:
                if self.mode == "user":
                    self.display_user_menu()
                    choice = input().strip()

                    if choice == '1':
                        self.search_movies()
                    elif choice == '2':
                        self.get_movie_recommendations()
                    elif choice == '3':
                        self.get_personalized_recommendations()
                    elif choice == '4':
                        self.get_genre_recommendations()
                    elif choice == '5':
                        self.rate_movie()
                    elif choice == '6':
                        self.view_popular_movies()
                    elif choice == '7':
                        self.mode = "admin"
                        print("\nSwitching to admin mode...")
                    elif choice == '8':
                        print("\nThank you for using the Movie Recommender System!")
                        sys.exit(0)
                    else:
                        print("\nInvalid choice. Please enter a number between 1 and 8.")
                elif self.mode == "admin":
                    self.display_admin_menu()
                    choice = input().strip()

                    if choice == '1':
                        self.get_system_status()
                    elif choice == '2':
                        self.save_model()
                    elif choice == '3':
                        self.get_analytics_report()
                    elif choice == '4':
                        self.plot_analytics()
                    elif choice == '5':
                        self.start_monitoring()
                    elif choice == '6':
                        self.mode = "user"
                        print("\nSwitching to user mode...")
                    elif choice == '7':
                        print("\nThank you for using the Movie Recommender System!")
                        sys.exit(0)
                    else:
                        print("\nInvalid choice. Please enter a number between 1 and 7.")
            except KeyboardInterrupt:
                 print("\nExiting the program.")
                 sys.exit(0)
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                print(f"\nAn unexpected error occurred. Please check the logs for more details.")

def main():
    """
    Entry point for the Movie Recommender CLI application.
    """
    cli = MovieRecommenderCLI()
    cli.run()

if __name__ == "__main__":
    main()