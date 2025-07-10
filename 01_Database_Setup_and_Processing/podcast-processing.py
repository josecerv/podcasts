#!/usr/bin/env python3
"""
Podcast RSS Feed Processor
A script to process all podcast RSS feeds, collecting episodes from 2024 or 2025.
Episodes without a non-empty description are skipped.
Duplicate episodes (based on podcast_id and episode_id) are removed (keeping the first one).
Only podcasts with at least 8 unique episodes (after deduplication) are saved.
The resulting episodes are written as SQL DDL and INSERT statements to a timestamped .sql file,
optionally loaded into a PostgreSQL database, and then compressed using gzip.
"""

import feedparser
import pandas as pd
import logging
from datetime import datetime
import hashlib
import time
import os
import re
import multiprocessing
from functools import partial
import gzip
from typing import List, Dict, Optional
import concurrent.futures
import signal
from urllib.request import urlopen
import socket
import subprocess

# -------------------------------
# Database credentials (embedded)
# -------------------------------
DB_USER = 'jcervantez'
DB_PASSWORD = 'Cervantez12'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'podcast_episodes'

# Set the PostgreSQL environment variables for psql.
os.environ['PGUSER'] = DB_USER
os.environ['PGPASSWORD'] = DB_PASSWORD
os.environ['PGHOST'] = DB_HOST
os.environ['PGPORT'] = DB_PORT
os.environ['PGDATABASE'] = DB_NAME

# Constants
TIMEOUT = 10  # seconds
BATCH_SIZE = 10000  # Show progress every 10,000 podcasts

# Configure minimal logging for errors only
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('podcast_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Process-safe counters for progress tracking
from multiprocessing import Value
processed_count = Value('i', 0)
total_podcasts = Value('i', 0)
successful_podcasts = Value('i', 0)
batch_successful = Value('i', 0)

def set_feedparser_timeout():
    """Set socket timeout for feedparser"""
    socket.setdefaulttimeout(TIMEOUT)

def clean_text(text):
    """
    Clean text to be UTF-8 compliant and remove problematic characters.
    """
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.encode('utf-8', 'replace').decode('utf-8', 'replace')
        text = re.sub(r'[\x80-\x9F]', '', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'[\u2028\u2029\u0085\u000A\u000B\u000C\u000D\u2028\u2029]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = re.sub(r'&#?\w+;', ' ', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ''

def get_robust_description(entry):
    """
    Build a robust description from available metadata in the RSS entry.
    It concatenates multiple potential sources: description, summary, itunes_summary,
    summary_detail, content, and media:description.
    """
    parts = []

    # Common keys: description, summary, and itunes_summary.
    for key in ['description', 'summary', 'itunes_summary']:
        value = entry.get(key)
        if value:
            parts.append(value)

    # Check summary_detail if available
    summary_detail = entry.get('summary_detail', {}).get('value')
    if summary_detail:
        parts.append(summary_detail)

    # Check if content is available (often a list of dictionaries)
    content_list = entry.get('content', [])
    if content_list and isinstance(content_list, list):
        for content_item in content_list:
            value = content_item.get('value', '')
            if value:
                parts.append(value)

    # Check for alternative media description fields that some namespaces might use.
    media_description = entry.get('media:description')
    if media_description:
        parts.append(media_description)

    # Concatenate all parts ensuring a space separator between them.
    robust_description = ' '.join(parts).strip()
    return robust_description

def verify_unique_episodes(episodes):
    """
    Verify that all episodes are unique based on podcast_id and episode_id.
    Returns a list of unique episodes.
    """
    unique_episodes = {}
    duplicate_count = 0
    for episode in episodes:
        key = (episode['podcast_id'], episode['episode_id'])
        if key not in unique_episodes:
            unique_episodes[key] = episode
        else:
            duplicate_count += 1
    if duplicate_count > 0:
        print(f"Removed {duplicate_count:,} duplicate episodes")
    return list(unique_episodes.values())

def enforce_minimum_episode_count(episodes, min_count=8):
    """
    Enforces that each podcast has at least `min_count` unique episodes.
    Groups episodes by podcast_id, and only returns episodes from podcasts
    that have min_count or more episodes.
    """
    podcasts = {}
    for ep in episodes:
        pid = ep['podcast_id']
        podcasts.setdefault(pid, []).append(ep)
    
    valid_episodes = []
    removed = 0
    for pid, eps in podcasts.items():
        if len(eps) >= min_count:
            valid_episodes.extend(eps)
        else:
            removed += 1
    print(f"Removed {removed} podcasts with fewer than {min_count} unique episodes after deduplication.")
    return valid_episodes

class PodcastProcessor:
    """Handles the processing of podcast feeds."""
    
    @staticmethod
    def generate_episode_id(podcast_id: str, clean_title: str, clean_description: str) -> str:
        """
        Generate a unique hash for an episode using the podcast_id,
        cleaned title, and cleaned description.
        """
        components = [podcast_id, clean_title, clean_description]
        unique_str = ''.join(components).encode('utf-8', 'ignore')
        return hashlib.sha256(unique_str).hexdigest()

    def process_entry(self, entry: dict, podcast_id: str) -> Optional[dict]:
        """Process a single feed entry."""
        try:
            published_parsed = entry.get('published_parsed')
            published_date = datetime(*published_parsed[:6]) if published_parsed else None
            if not published_date or published_date.year not in (2024, 2025):
                return None

            # Use robust description extraction function
            raw_description = get_robust_description(entry)
            clean_description = clean_text(raw_description)
            if not clean_description:
                return None

            raw_title = entry.get('title', '')
            if isinstance(raw_title, dict):
                raw_title = str(raw_title)
            clean_title = clean_text(raw_title)

            raw_link = entry.get('link', '')
            if isinstance(raw_link, dict):
                raw_link = str(raw_link)
            clean_link = clean_text(raw_link)

            raw_duration = entry.get('itunes_duration', '')
            if isinstance(raw_duration, dict):
                raw_duration = str(raw_duration)
            clean_duration = clean_text(raw_duration)

            episode_id = self.generate_episode_id(podcast_id, clean_title, clean_description)
            return {
                'podcast_id': podcast_id,
                'episode_id': episode_id,
                'episode_title': clean_title,
                'episode_description': clean_description,
                'episode_published': published_date,
                'episode_link': clean_link,
                'episode_duration': clean_duration
            }
        except Exception as e:
            logger.error(f"Error processing entry for podcast '{podcast_id}': {str(e)}")
            return None

    def process_feed_return(self, rss_url: str, podcast_id: str) -> List[dict]:
        """
        Process a single podcast feed with timeout.
        """
        try:
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(TIMEOUT)
            try:
                feed = feedparser.parse(rss_url)
            finally:
                socket.setdefaulttimeout(original_timeout)
            if not feed.entries:
                return []
            local_episodes = []
            for entry in feed.entries:
                episode_data = self.process_entry(entry, podcast_id)
                if episode_data:
                    local_episodes.append(episode_data)
            return local_episodes
        except Exception as e:
            logger.error(f"Error processing feed {podcast_id}: {str(e)}")
            return []

def process_single_podcast(row_dict, processor: PodcastProcessor) -> List[dict]:
    """
    Process a single podcast feed with simplified progress tracking.
    """
    try:
        rss_url = row_dict.get('rss')
        podcast_id = row_dict.get('podcast_id')
        if not rss_url:
            return []
        episodes = processor.process_feed_return(rss_url, podcast_id)
        with processed_count.get_lock():
            processed_count.value += 1
            current_count = processed_count.value
            if len(episodes) >= 8:
                with successful_podcasts.get_lock(), batch_successful.get_lock():
                    successful_podcasts.value += 1
                    batch_successful.value += 1
            if current_count % BATCH_SIZE == 0:
                with batch_successful.get_lock():
                    batch_rate = (batch_successful.value / BATCH_SIZE) * 100
                    print(f"\n{current_count:,} podcasts processed:")
                    print(f" - Current batch successful podcasts: {batch_successful.value:,} ({batch_rate:.1f}% of {BATCH_SIZE})")
                    print(f" - Total successful podcasts so far: {successful_podcasts.value:,}")
                    batch_successful.value = 0
        return episodes if len(episodes) >= 8 else []
    except Exception as e:
        logger.error(f"Error processing podcast {podcast_id}: {str(e)}")
        return []

def sql_escape(value):
    """
    Escape a value for SQL insertion.
    """
    if value is None or value == '':
        return 'NULL'
    if isinstance(value, datetime):
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    if isinstance(value, str):
        value = value.replace('\x00', '')
        value = clean_text(value).replace("'", "''")
    return f"'{value}'"

def load_sql_file(sql_filename, psql_path=r"C:\Program Files\PostgreSQL\17\bin\psql.exe"):
    """
    Load an SQL file into PostgreSQL by calling the psql command-line tool.
    """
    cmd = [
        psql_path,
        "-h", os.environ.get("PGHOST"),
        "-U", os.environ.get("PGUSER"),
        "-d", os.environ.get("PGDATABASE"),
        "-f", sql_filename
    ]
    print(f"Loading SQL file {sql_filename} into PostgreSQL...")
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        print("Error loading SQL file into PostgreSQL:")
        print(result.stderr)
    else:
        print("SQL file loaded successfully into PostgreSQL!")
        print(result.stdout)

def write_compressed_sql(episodes: List[dict], base_filename: str = 'podcast_episodes', load_to_postgres: bool = False):
    """
    Write the episodes data to a timestamped .sql file.
    Optionally, load the SQL file into PostgreSQL before compressing the file.
    Then compress the SQL file using gzip.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sql_filename = f'{base_filename}_{timestamp}.sql'
    compressed_filename = f'{sql_filename}.gz'
    
    # Create table name with today's date (or reminder_one if preferred)
    today_date = datetime.now().strftime('%Y_%m_%d')
    # You can use either of these table names:
    table_name = f"episodes_{today_date}"  # e.g., episodes_2025_03_02
    # Or uncomment the line below to use reminder_one instead
    # table_name = "episodes_reminder_one"

    print(f"\nWriting SQL file: {sql_filename}")
    with open(sql_filename, 'w', encoding='utf-8') as f:
        ddl = (
            f"DROP TABLE IF EXISTS {table_name};\n"
            f"CREATE TABLE {table_name} (\n"
            "    id SERIAL PRIMARY KEY,\n"
            "    podcast_id VARCHAR(255),\n"
            "    episode_id VARCHAR(64),\n"
            "    episode_title TEXT,\n"
            "    episode_description TEXT,\n"
            "    episode_published TIMESTAMP,\n"
            "    episode_link VARCHAR(2048),\n"
            "    episode_duration VARCHAR(50),\n"
            "    UNIQUE (podcast_id, episode_id)\n"
            ");\n\n"
        )
        f.write(ddl)
        total_episodes = len(episodes)
        for i, episode in enumerate(episodes, 1):
            if i % 10000 == 0:
                print(f"Writing episode {i:,} of {total_episodes:,}")
            columns = [
                'podcast_id', 'episode_id', 'episode_title', 'episode_description',
                'episode_published', 'episode_link', 'episode_duration'
            ]
            values = [sql_escape(episode.get(col)) for col in columns]
            insert_statement = (
                f"INSERT INTO {table_name} ({', '.join(columns)}) "
                f"VALUES ({', '.join(values)}) "
                f"ON CONFLICT (podcast_id, episode_id) DO NOTHING;\n"
            )
            f.write(insert_statement)
    # Optionally load the SQL file into PostgreSQL:
    if load_to_postgres:
        load_sql_file(sql_filename)
    print("Compressing SQL file...")
    with open(sql_filename, 'rb') as f_in, gzip.open(compressed_filename, 'wb', compresslevel=9) as f_out:
        f_out.writelines(f_in)
    os.remove(sql_filename)
    print(f"SQL file compressed and saved as {compressed_filename}")


def process_podcast_wrapper(args):
    """
    Wrapper function to unpack arguments for process_single_podcast.
    """
    row_dict, processor = args
    return process_single_podcast(row_dict, processor)

def process_podcasts(csv_path: str = 'podcasts_final_sample.csv', load_to_postgres: bool = False):
    """
    Main processing function with improved progress tracking, deduplication,
    and minimum episode enforcement.
    If load_to_postgres is True, the SQL file will be loaded into PostgreSQL.
    """
    start_time = time.time()
    try:
        set_feedparser_timeout()
        print("Loading podcast data...")
        podcasts_df = pd.read_csv(csv_path, low_memory=False)
        total_count = len(podcasts_df)
        print(f"Found {total_count:,} podcasts to process")
        processed_count.value = 0
        successful_podcasts.value = 0
        batch_successful.value = 0
        processor = PodcastProcessor()
        print("Preparing podcast data...")
        podcast_data = [
            {
                'rss': row.rss if pd.notna(row.rss) else None,
                'podcast_id': row.podcast_id if pd.notna(row.podcast_id) else None
            }
            for row in podcasts_df.itertuples()
        ]
        num_workers = multiprocessing.cpu_count() * 2
        print(f"Processing with {num_workers} worker processes")
        process_args = [(row_dict, processor) for row_dict in podcast_data]
        all_episodes = []
        print("Starting podcast processing...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for batch_results in executor.map(process_podcast_wrapper, process_args):
                if batch_results:
                    all_episodes.extend(batch_results)
        print(f"\nTotal episodes collected: {len(all_episodes):,}")
        print("Verifying unique episodes...")
        unique_episodes = verify_unique_episodes(all_episodes)
        print(f"Total unique episodes after verification: {len(unique_episodes):,}")
        unique_episodes = enforce_minimum_episode_count(unique_episodes, min_count=8)
        print(f"Total episodes after enforcing minimum episodes per podcast: {len(unique_episodes):,}")
        write_compressed_sql(unique_episodes, load_to_postgres=load_to_postgres)
        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Final successful podcasts count: {successful_podcasts.value:,}")
        print(f"Final unique episodes count: {len(unique_episodes):,}")
    except Exception as e:
        logger.error(f"Critical error during processing: {e}")
        raise

def main():
    """
    Entry point for script execution.
    Set load_to_postgres=True to automatically load the SQL file into PostgreSQL.
    """
    try:
        print(f"Starting podcast RSS feed processing at: {datetime.now()}")
        def signal_handler(signum, frame):
            print("\nReceived interrupt signal. Finishing current batch before exiting...")
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, signal_handler)
        process_podcasts(csv_path='podcasts_final_sample.csv', load_to_postgres=True)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    finally:
        print(f"Process ended at: {datetime.now()}")

if __name__ == "__main__":
    main()
