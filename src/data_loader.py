import os
import requests
import zipfile
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

class DataLoader:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        self.genre_matrix = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.logger = logging.getLogger(__name__)

    def download_dataset(self):
        """Download MovieLens 1M dataset"""
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        if not os.path.exists("ml-1m"):
            self.logger.info("Downloading MovieLens 1M dataset...")
            r = requests.get(url, stream=True)
            with open("ml-1m.zip", 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192)):
                    if chunk:
                        f.write(chunk)

            self.logger.info("Extracting files...")
            with zipfile.ZipFile("ml-1m.zip", 'r') as zip_ref:
                zip_ref.extractall()
            os.remove("ml-1m.zip")
        else:
            self.logger.info("Dataset already exists")

    def load_data(self):
        """Load and preprocess the MovieLens 1M dataset"""
        self.logger.info("Loading data...")

        self.ratings_df = pd.read_csv('ml-1m/ratings.dat',
                                    sep='::',
                                    engine='python',
                                    names=['user_id', 'movie_id', 'rating', 'timestamp'])

        self.movies_df = pd.read_csv('ml-1m/movies.dat',
                                   sep='::',
                                   engine='python',
                                   encoding='latin-1',
                                   names=['movie_id', 'title', 'genres'])

        self.users_df = pd.read_csv('ml-1m/users.dat',
                                  sep='::',
                                  engine='python',
                                  names=['user_id', 'gender', 'age', 'occupation', 'zip_code'])

        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess the loaded data"""
        # Process movies data
        self.movies_df['genres'] = self.movies_df['genres'].str.split('|')
        self.movies_df['year'] = self.movies_df['title'].str.extract('\((\d{4})\)').astype(float)
        self.movies_df['title_clean'] = self.movies_df['title'].str.replace('\(\d{4}\)', '').str.strip()

        # Create genre matrix
        mlb = MultiLabelBinarizer()
        self.genre_matrix = mlb.fit_transform(self.movies_df['genres'])

        # Create TF-IDF matrix for content-based filtering
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies_df['title_clean'] + ' ' + self.movies_df['genres'].apply(lambda x: ' '.join(x))
        )

    def get_user_movie_matrix(self):
        """Create and return the user-movie rating matrix"""
        user_movie_matrix = self.ratings_df.pivot(
            index='movie_id',
            columns='user_id',
            values='rating'
        ).fillna(0)

        # Normalize ratings
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(user_movie_matrix)

        return user_movie_matrix, normalized_matrix

    def get_movie_ids(self):
        """Get list of movie IDs"""
        return self.movies_df['movie_id'].values

    def get_movie_info(self, movie_id):
        """Get information about a specific movie"""
        movie = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
        return {
            'movie_id': movie_id,
            'title': movie['title'],
            'genres': '|'.join(movie['genres']),
            'year': movie['year']
        }
