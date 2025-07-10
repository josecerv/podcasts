#!/usr/bin/env python3
"""
Podcast 12-Month Episode Counter
Counts episodes in podcasts published between February 23, 2024 and February 23, 2025.
Filters the podcasts to only those whose podcast IDs are found in email_one_survey.csv,
email_two_survey.csv, and email_three_survey.csv.
Only processes podcast IDs that are not already in podcast_episodes_last_12_months.csv.
"""

import feedparser
import pandas as pd
import logging
from datetime import datetime
import time
import socket
import csv
from concurrent.futures import ThreadPoolExecutor
import threading
import os

# Constants
TIMEOUT = 10  # seconds
# Define the 12-month window: from February 23, 2024 to February 23, 2025
WINDOW_START = datetime(2024, 2, 23, 0, 0, 0)
WINDOW_END = datetime(2025, 2, 23, 23, 59, 59)
MAX_THREADS = 250  # High thread count for I/O-bound tasks
PROGRESS_INTERVAL = 500  # Report progress every 500 podcasts
OUTPUT_FILENAME = 'podcast_episodes_last_12_months.csv'

# Minimal logging
logging.basicConfig(level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-safe counters
lock = threading.Lock()
processed_count = 0
success_count = 0

def count_episodes_in_window(podcast):
    """
    Count episodes published in the 12-month window for a single podcast.
    Returns a tuple: (podcast_id, episodes_last_12_months)
    """
    global processed_count, success_count
    
    podcast_id = podcast.get('podcast_id', 'unknown')
    rss_url = podcast.get('rss')
    
    if not rss_url:
        with lock:
            processed_count += 1
            if processed_count % PROGRESS_INTERVAL == 0:
                print(f"Processed {processed_count:,} podcasts | Successful: {success_count:,}")
        return podcast_id, 0
    
    try:
        # Set timeout for this request
        socket.setdefaulttimeout(TIMEOUT)

        # Parse the feed
        feed = feedparser.parse(rss_url)
        
        if not feed or not hasattr(feed, 'entries') or not feed.entries:
            with lock:
                processed_count += 1
                if processed_count % PROGRESS_INTERVAL == 0:
                    print(f"Processed {processed_count:,} podcasts | Successful: {success_count:,}")
            return podcast_id, 0

        count = 0
        # Loop through the episodes in the feed
        for entry in feed.entries:
            try:
                published_parsed = entry.get('published_parsed')
                if not published_parsed:
                    continue
                # Convert published_parsed to datetime
                pub_date = datetime(*published_parsed[:6])
                # Check if the publication date is within the specified window
                if WINDOW_START <= pub_date <= WINDOW_END:
                    count += 1
            except Exception as e:
                continue

        # Update thread-safe counters
        with lock:
            processed_count += 1
            if count > 0:
                success_count += 1

            if processed_count % PROGRESS_INTERVAL == 0:
                print(f"Processed {processed_count:,} podcasts | Successful: {success_count:,}")
                
        return podcast_id, count

    except Exception as e:
        with lock:
            processed_count += 1
            if processed_count % PROGRESS_INTERVAL == 0:
                print(f"Processed {processed_count:,} podcasts | Successful: {success_count:,}")
        return podcast_id, 0

def main():
    print(f"Starting podcast episode counter at: {datetime.now()}")
    start_time = time.time()
    
    try:
        # Load podcast IDs from survey CSV files
        print("Loading survey podcast IDs...")
        survey1 = pd.read_csv('email_one_survey.csv', low_memory=False)
        survey2 = pd.read_csv('email_two_survey.csv', low_memory=False)
        survey3 = pd.read_csv('email_three_survey.csv', low_memory=False)
        
        # Ensure IDs are of type str and get unique values from all surveys
        survey_ids = set(survey1['podcastID'].astype(str).unique()).union(
                      set(survey2['podcastID'].astype(str).unique()),
                      set(survey3['podcastID'].astype(str).unique()))
        print(f"Found {len(survey_ids):,} unique podcasts in surveys")
        
        # Check if the output file already exists and load existing data
        existing_results = {}
        if os.path.exists(OUTPUT_FILENAME):
            print(f"Loading existing processed podcast data from {OUTPUT_FILENAME}...")
            try:
                existing_df = pd.read_csv(OUTPUT_FILENAME, low_memory=False)
                for _, row in existing_df.iterrows():
                    podcast_id = str(row['podcast_id'])
                    episode_count = row['episodes_last_12_months']
                    existing_results[podcast_id] = episode_count
                print(f"Found {len(existing_results):,} previously processed podcasts")
            except Exception as e:
                print(f"Error reading existing file: {str(e)}")
                print("Will create a new file.")
        
        # Calculate which podcast IDs are new and need processing
        existing_podcast_ids = set(existing_results.keys())
        new_podcast_ids = survey_ids - existing_podcast_ids
        print(f"Found {len(new_podcast_ids):,} new podcasts to process")
        
        if not new_podcast_ids:
            print("No new podcasts to process. Exiting.")
            return
        
        # Load final sample CSV file
        print("Loading podcast data from podcasts_final_sample.csv...")
        podcasts_df = pd.read_csv('podcasts_final_sample.csv', low_memory=False)
        original_count = len(podcasts_df)
        print(f"Original podcasts count in sample: {original_count:,}")
        
        # Filter podcasts to only those in the survey lists and not already processed
        podcasts_df['podcast_id_str'] = podcasts_df['podcast_id'].astype(str)
        podcasts_df = podcasts_df[podcasts_df['podcast_id_str'].isin(new_podcast_ids)]
        filtered_count = len(podcasts_df)
        print(f"Filtered podcasts to process: {filtered_count:,}")
        
        # Convert the data to list of dictionaries for faster processing
        podcasts = [{'podcast_id': str(row.podcast_id), 'rss': row.rss} 
                    for row in podcasts_df.itertuples() 
                    if hasattr(row, 'rss') and pd.notna(row.rss)]
        
        if not podcasts:
            print("No podcasts to process after filtering. Exiting.")
            return
        
        print(f"Using {MAX_THREADS} threads for parallel processing")
        print(f"Counting episodes published between {WINDOW_START.strftime('%Y-%m-%d')} "
              f"and {WINDOW_END.strftime('%Y-%m-%d')}")
        
        # Process feeds in parallel using threads
        new_results = []
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            new_results = list(executor.map(count_episodes_in_window, podcasts))
        
        # Add new results to existing results
        for podcast_id, episode_count in new_results:
            existing_results[str(podcast_id)] = episode_count
        
        # Write combined results back to CSV
        print(f"Writing {len(existing_results):,} total results to CSV ({OUTPUT_FILENAME})...")
        with open(OUTPUT_FILENAME, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['podcast_id', 'episodes_last_12_months'])
            for podcast_id, episode_count in existing_results.items():
                writer.writerow([podcast_id, episode_count])
        
        # Print final stats
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time/60:.2f} minutes")
        print(f"Total new podcasts processed: {processed_count:,}")
        print(f"Successful new podcasts: {success_count:,}")
        print(f"Total podcasts in output file: {len(existing_results):,}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Process ended at: {datetime.now()}")

if __name__ == "__main__":
    main()
