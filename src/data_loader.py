import os
import requests
import zipfile
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List, Dict, Any, Optional
from .utils.logger import setup_logger

class DataLoader:
    """
    Handles downloading, loading, and preprocessing the MovieLens 1M dataset.

    Provides methods to:
    - Download the dataset from a URL.
    - Extract the dataset from a ZIP archive.
    - Load the dataset into Pandas DataFrames.
    - Preprocess the data, including:
        - One-hot encoding of genres.
        - TF-IDF vectorization of movie titles and genres.
    - Create user-movie rating matrix.
    - Get movie information by ID.
    """

    def __init__(self):
        """
        Initializes the DataLoader and sets up logging.
        """
        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        self.genre_matrix: Optional[np.ndarray] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.logger = setup_logger(__name__)

    def download_dataset(self, url: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                         download_dir: str = ".", force_download: bool = False):
        """
        Downloads the MovieLens 1M dataset if it doesn't exist locally.

        Args:
            url: The URL of the dataset.
            download_dir: The directory to download the dataset to.
            force_download: If True, re-download the dataset even if it exists.
        """
        dataset_dir = os.path.join(download_dir, "ml-1m")
        zip_filepath = os.path.join(download_dir, "ml-1m.zip")

        if not os.path.exists(dataset_dir) or force_download:
            if force_download:
                self.logger.info("Forcing re-download of the dataset.")
                if os.path.exists(dataset_dir):
                    os.system(f"rm -rf {dataset_dir}")
                if os.path.exists(zip_filepath):
                    os.remove(zip_filepath)

            self.logger.info(f"Downloading MovieLens 1M dataset from {url} to {download_dir}...")

            # Download the dataset with progress bar
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()  # Raise an exception for bad status codes
                    total_size = int(r.headers.get('content-length', 0))
                    block_size = 8192
                    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

                    with open(zip_filepath, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=block_size):
                            progress_bar.update(len(chunk))
                            f.write(chunk)
                    progress_bar.close()

                if total_size != 0 and progress_bar.n != total_size:
                    self.logger.error("Error downloading dataset: Incomplete download.")
                    raise Exception("Incomplete download.")

                self.logger.info("Dataset downloaded successfully.")

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error downloading dataset: {e}")
                raise

            self.logger.info(f"Extracting files from {zip_filepath}...")

            # Extract the dataset
            try:
                with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
            except zipfile.BadZipFile as e:
                self.logger.error(f"Error extracting dataset: {e}")
                self.logger.error(f"Please check the integrity of {zip_filepath}.")
                raise

            os.remove(zip_filepath)
            self.logger.info("Dataset extracted and ZIP file removed.")

        else:
            self.logger.info("Dataset already exists locally. Skipping download.")

    def load_data(self):
        """
        Loads the MovieLens 1M dataset from CSV files into Pandas DataFrames.
        Also preprocesses the data.
        """
        self.logger.info("Loading data from CSV files...")

        try:
            # Load the data into DataFrames
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

            self.logger.info("Data loaded successfully.")

            self._preprocess_data()

        except FileNotFoundError as e:
            self.logger.error(f"Error loading data: {e}")
            self.logger.error("Make sure the dataset is downloaded and extracted to the correct directory.")
            raise
        except pd.errors.ParserError as e:
            self.logger.error(f"Error parsing data: {e}")
            self.logger.error("Check the format of the data files.")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading data: {e}")
            raise

    def _preprocess_data(self):
        """
        Preprocesses the loaded data:

        - Cleans movie titles.
        - Extracts the year from movie titles.
        - Creates a genre matrix using one-hot encoding.
        - Creates a TF-IDF matrix from movie titles and genres.
        """
        self.logger.info("Preprocessing data...")

        # Process movies data
        self.movies_df['genres'] = self.movies_df['genres'].str.split('|')
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        self.movies_df['title_clean'] = self.movies_df['title'].str.replace(r'\((\d{4})\)', '', regex=True).str.strip()
        self.logger.info("Movie data preprocessed.")

        # Create genre matrix
        mlb = MultiLabelBinarizer()
        self.genre_matrix = mlb.fit_transform(self.movies_df['genres'])
        self.logger.info("Genre matrix created.")

        # Create TF-IDF matrix for content-based filtering
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies_df['title_clean'] + ' ' + self.movies_df['genres'].apply(lambda x: ' '.join(x))
        )
        self.logger.info("TF-IDF matrix created.")
        self.logger.info("Data preprocessing complete.")

    def get_user_movie_matrix(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Creates and returns the user-movie rating matrix and its normalized version.

        Returns:
            A tuple containing:
            - user_movie_matrix: A DataFrame where rows are movies, columns are users, and values are ratings.
            - normalized_matrix: A NumPy array that is the standardized version of the user-movie matrix.
        """
        self.logger.info("Creating user-movie matrix...")

        # Create the user-movie matrix
        user_movie_matrix = self.ratings_df.pivot(
            index='movie_id',
            columns='user_id',
            values='rating'
        ).fillna(0)

        # Normalize the matrix using StandardScaler
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(user_movie_matrix)

        self.logger.info("User-movie matrix created and normalized.")
        return user_movie_matrix, normalized_matrix

    def get_movie_ids(self) -> List[int]:
        """
        Returns a list of all unique movie IDs.

        Returns:
             A list of movie IDs.
        """
        return self.movies_df['movie_id'].tolist()

    def get_movie_info(self, movie_id: int) -> Dict[str, Any]:
        """
        Returns information about a specific movie.

        Args:
            movie_id: The ID of the movie.

        Returns:
            A dictionary containing the movie's title, genres, and year.
            Returns an empty dictionary if the movie ID is not found.
        """
        try:
            movie = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            return {
                'movie_id': movie_id,
                'title': movie['title'],
                'genres': '|'.join(movie['genres']),
                'year': movie['year']
            }
        except IndexError:
            self.logger.warning(f"Movie ID {movie_id} not found.")
              # Return empty dict if movie not found