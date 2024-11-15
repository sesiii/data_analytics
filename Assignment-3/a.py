import pandas as pd
import requests
import time
import re
import logging
import os
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('script.log')
    ]
)
logger = logging.getLogger(__name__)

class MovieDataFetcher:
    def __init__(self, api_key: str, input_file: str, output_file: str):
        self.api_key = api_key
        self.input_file = input_file
        self.output_file = output_file
        self.checkpoint_file = 'checkpoint.txt'
        self.movies_df = pd.read_csv(input_file)
        self.start_index = self._load_checkpoint()
        
        # Initialize or load the output DataFrame
        if os.path.exists(self.output_file):
            self.processed_df = pd.read_csv(self.output_file)
            logger.info(f"Loaded existing output file with {len(self.processed_df)} entries")
        else:
            self.processed_df = pd.DataFrame(columns=self.movies_df.columns.tolist() + ['top_actors'])
            logger.info("Created new output DataFrame")

    def _load_checkpoint(self) -> int:
        """Load the checkpoint or return 0 if no checkpoint exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                index = int(f.read().strip())
                logger.info(f"Resuming from index {index}")
                return index
        return 0

    def _save_checkpoint(self, index: int) -> None:
        """Save the current processing index to checkpoint file."""
        with open(self.checkpoint_file, 'w') as f:
            f.write(str(index))

    @staticmethod
    def extract_year(title: str) -> Optional[str]:
        """Extract year from movie title."""
        match = re.search(r'\((\d{4})\)', title)
        return match.group(1) if match else None

    def get_top_actors(self, movie_title: str, year: Optional[str]) -> List[str]:
        """Fetch top actors for a movie from TMDB API."""
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': self.api_key,
            'query': movie_title,
            'year': year
        }

        try:
            search_response = requests.get(search_url, params=params).json()
            
            if not search_response.get('results'):
                logger.warning(f"No results found for {movie_title} ({year})")
                return []

            movie_id = search_response['results'][0]['id']
            credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
            credits_params = {'api_key': self.api_key}
            
            credits_response = requests.get(credits_url, params=credits_params).json()
            top_actors = [actor['name'] for actor in credits_response.get('cast', [])[:5]]
            
            return top_actors

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {movie_title} ({year}): {e}")
            return []
        except KeyError as e:
            logger.error(f"Unexpected API response format for {movie_title} ({year}): {e}")
            return []

    def update_and_save_progress(self, index: int, row: pd.Series, top_actors: List[str]) -> None:
        """Update the processed DataFrame and save progress."""
        # Create a new row with all the original data plus top_actors
        new_row = row.to_dict()
        new_row['top_actors'] = ', '.join(top_actors)
        
        # If the index exists in processed_df, update it; otherwise append
        if index < len(self.processed_df):
            self.processed_df.loc[index] = new_row
        else:
            self.processed_df = pd.concat([self.processed_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to CSV
        self.processed_df.to_csv(self.output_file, index=False)
        logger.info(f"Saved progress for movie {index}: {row['title']}")
        
        # Update checkpoint
        self._save_checkpoint(index)

    def process_movies(self) -> None:
        """Process all movies and update the CSV file."""
        total_movies = len(self.movies_df)
        
        for index, row in self.movies_df.iterrows():
            if index < self.start_index:
                continue

            movie_title = row['title']
            year = self.extract_year(movie_title)
            movie_title_clean = re.sub(r'\(\d{4}\)', '', movie_title).strip()

            logger.info(f"Processing {index + 1}/{total_movies}: {movie_title_clean} ({year})")

            # Implement retry logic
            retries = 3
            top_actors = []
            
            for attempt in range(retries):
                try:
                    top_actors = self.get_top_actors(movie_title_clean, year)
                    if top_actors:
                        logger.info(f"Found actors for {movie_title_clean}: {', '.join(top_actors)}")
                        break
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(5)
            
            # Update the processed DataFrame and save progress
            self.update_and_save_progress(index, row, top_actors)
            
            # Respect API rate limits
            time.sleep(0.25)

def main():
    # Configuration
    API_KEY = '5fa8e2a358a75b3adf88bb6aaf916598'
    INPUT_FILE = 'movies_set1.csv'
    OUTPUT_FILE = 'movies_with_actors.csv'

    try:
        fetcher = MovieDataFetcher(API_KEY, INPUT_FILE, OUTPUT_FILE)
        fetcher.process_movies()
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise

if __name__ == "__main__":
    main()