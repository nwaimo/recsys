import sys
import logging
from .recommender import MovieRecommenderSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommenderCLI:
    def __init__(self):
        logger.info("Initializing recommender system...")
        self.recommender = MovieRecommenderSystem()
        self.recommender.initialize()
      
        logger.info("Recommender system initialized successfully")

    def print_menu(self):
        print("\nMovie Recommender System")
        print("1. Search for a movie")
        print("2. Get movie recommendations")
        print("3. Get personalized recommendations")
        print("4. Get genre recommendations")
        print("5. Rate a movie")
        print("6. View popular movies")
        print("7. Exit")
        print("\nEnter your choice (1-7): ")

    def search_movies(self):
        query = input("\nEnter movie title to search: ").strip()
        if not query:
            print("Please enter a valid search query")
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
            print("Please enter a valid movie ID (number)")
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")

    def get_personalized_recommendations(self):
        try:
            user_id = int(input("\nEnter user ID: "))
            recommendations = self.recommender.get_user_recommendations(user_id)
            
            if not recommendations:
                print("No personalized recommendations found for this user.")
                return

            print("\nPersonalized Recommendations:")
            for movie in recommendations:
                print(f"ID: {movie['movie_id']} - {movie['title']} ({movie['year']}) - Genres: {','.join(movie['genres'])}")
        
        except ValueError:
            print("Please enter a valid user ID (number)")
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")

    def get_genre_recommendations(self):
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

    def rate_movie(self):
        try:
            user_id = int(input("\nEnter user ID: "))
            movie_id = int(input("Enter movie ID: "))
            rating = float(input("Enter rating (0.5-5.0): "))
            
            if rating < 0.5 or rating > 5.0:
                print("Rating must be between 0.5 and 5.0")
                return

            self.recommender.add_new_rating(user_id, movie_id, rating)
            print("Rating added successfully!")
        
        except ValueError:
            print("Please enter valid numbers for user ID, movie ID, and rating")
        except Exception as e:
            print(f"Error adding rating: {str(e)}")

    def view_popular_movies(self):
        ratings_df = self.recommender.data_loader.ratings_df
        movies_df = self.recommender.data_loader.movies_df
        
        # Calculate average ratings and number of ratings
        movie_stats = ratings_df.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        # Filter movies with at least 100 ratings
        popular_movies = movie_stats[movie_stats[('rating', 'count')] >= 100]
        
        # Sort by average rating
        popular_movies = popular_movies.sort_values(('rating', 'mean'), ascending=False)
        
        print("\nPopular Movies:")
        for movie_id in popular_movies['movie_id'][:10]:
            movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
            avg_rating = popular_movies[popular_movies['movie_id'] == movie_id][('rating', 'mean')].iloc[0]
            num_ratings = popular_movies[popular_movies['movie_id'] == movie_id][('rating', 'count')].iloc[0]
            print(f"ID: {movie['movie_id']} - {movie['title']} ({movie['year']}) - "
                  f"Average Rating: {avg_rating:.2f} ({num_ratings} ratings)")

    def run(self):
        while True:
            self.print_menu()
            try:
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
                    print("\nThank you for using the Movie Recommender System!")
                    sys.exit(0)
                else:
                    print("\nInvalid choice. Please enter a number between 1 and 7.")
            
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                logger.error(f"Error in main loop: {str(e)}")

def main():
    cli = MovieRecommenderCLI()
    cli.run()

if __name__ == "__main__":
    main()
