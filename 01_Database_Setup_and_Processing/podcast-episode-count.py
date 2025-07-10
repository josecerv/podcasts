#!/usr/bin/env python3
"""
Fast Podcast Episode Counter Update

This script updates the episode count by checking each RSS feed for newer 
episodes (assuming the feed entries are sorted newest-first). It only 
iterates over the feed until it encounters a publication date on or before 
the March 4, 2025 cutoff, then stops processing that feed.

For each podcast:
- The original count through March 4 (from podcast_episode_counts.csv) is used.
- The script counts how many new episodes (with pubDate > March 4, 2025) appear.
- The new total up to March 11 is computed by adding the new episode count to the 
  stored March 4 count.
  
The output CSV (podcast_episode_counts.csv) will now include a new column:
- episodes_to_mar11_2025
"""

import feedparser
import pandas as pd
import logging
from datetime import datetime
import time
import socket
import csv
import threading
import os
from concurrent.futures import ThreadPoolExecutor

# Constants
TIMEOUT = 10  # seconds
# Cutoff for counting new episodes is March 4, 2025 (inclusive)
CUTOFF_DATE_MAR4 = datetime(2025, 3, 4, 23, 59, 59)
MAX_THREADS = 250  # High thread count for I/O-bound tasks
PROGRESS_INTERVAL = 500  # Report progress every 500 podcasts

# Minimal logging configuration
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-safe counters
lock = threading.Lock()
processed_count = 0
success_count = 0

def update_podcast_episode_count(podcast):
    """
    For a single podcast, fetch the RSS feed and count new episodes that have been
    added after March 4, 2025. Because we assume the feeds are in reverse chronological
    order (newest first), the loop stops once an episode with pubDate <= March 4 is found.

    Args:
        podcast (dict): Contains keys 'podcast_id', 'rss', and 'episodes_to_mar4_2025'
                        (the baseline count stored previously).

    Returns:
        tuple: (podcast_id, updated_episodes_to_mar11_2025)
    """
    global processed_count, success_count

    podcast_id = podcast.get('podcast_id', 'unknown')
    rss_url = podcast.get('rss')
    baseline_mar4 = podcast.get('episodes_to_mar4_2025', 0)

    new_episode_count = 0

    if not rss_url:
        with lock:
            processed_count += 1
            if processed_count % PROGRESS_INTERVAL == 0:
                print(f"Processed {processed_count:,} podcasts | Successful: {success_count:,}")
        # Without an RSS URL we cannot update the count
        return podcast_id, baseline_mar4

    try:
        socket.setdefaulttimeout(TIMEOUT)
        feed = feedparser.parse(rss_url)

        if not (feed and hasattr(feed, 'entries') and feed.entries):
            with lock:
                processed_count += 1
                if processed_count % PROGRESS_INTERVAL == 0:
                    print(f"Processed {processed_count:,} podcasts | Successful: {success_count:,}")
            return podcast_id, baseline_mar4

        # Iterate through feed entries until we reach one at or before Mar 4.
        for entry in feed.entries:
            try:
                published_parsed = entry.get('published_parsed')
                if not published_parsed:
                    continue

                pub_date = datetime(*published_parsed[:6])
                # Check if the episode is a new one (i.e. pub_date > March 4)
                if pub_date > CUTOFF_DATE_MAR4:
                    new_episode_count += 1
                else:
                    # Since feed is chronological (newest first), break out once we pass Mar 4.
                    break
            except Exception:
                continue

        with lock:
            processed_count += 1
            # Count as a success if we got any new count or if the baseline exists
            if new_episode_count > 0 or baseline_mar4 > 0:
                success_count += 1
            if processed_count % PROGRESS_INTERVAL == 0:
                print(f"Processed {processed_count:,} podcasts | Successful: {success_count:,}")

        updated_count = baseline_mar4 + new_episode_count
        return podcast_id, updated_count

    except Exception as e:
        with lock:
            processed_count += 1
            if processed_count % PROGRESS_INTERVAL == 0:
                print(f"Processed {processed_count:,} podcasts | Successful: {success_count:,}")
        return podcast_id, baseline_mar4

def main():
    print(f"Starting podcast episode counter update at: {datetime.now()}")
    start_time = time.time()
    
    try:
        # Load the RSS feed sources from the sample CSV (which contains podcast_id, rss, etc.)
        print("Loading podcast source data from podcasts_final_sample.csv...")
        podcasts_df = pd.read_csv('podcasts_final_sample.csv', low_memory=False)
        print(f"Source Podcasts: {len(podcasts_df):,}")

        # Load the previously saved episode counts (assumed to have March4 counts) from CSV.
        print("Loading baseline episode counts from podcast_episode_counts.csv...")
        counts_df = pd.read_csv('podcast_episode_counts.csv', low_memory=False)
        print(f"Baseline count data: {len(counts_df):,} podcasts")

        # Merge on podcast_id to bring in the rss feed url and the baseline mar4 count.
        merged_df = pd.merge(
            podcasts_df[['podcast_id', 'rss']],
            counts_df[['podcast_id', 'episodes_to_feb26_2025', 'episodes_to_mar4_2025']],
            on='podcast_id',
            how='inner'
        )
        
        # Prepare list of podcast dicts for processing.
        podcasts = merged_df.to_dict('records')
        
        print(f"Using {MAX_THREADS} threads for parallel processing")
        print("Processing feeds to update episode counts for new episodes (after March 4, 2025)...")

        results = []
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            results = list(executor.map(update_podcast_episode_count, podcasts))
        
        # Create a DataFrame from the results; it has podcast_id and updated episodes_to_mar11_2025.
        updated_df = pd.DataFrame(results, columns=['podcast_id', 'episodes_to_mar11_2025'])
        
        # Merge updated counts into the original baseline counts.
        final_df = pd.merge(counts_df, updated_df, on='podcast_id', how='left')
        
        # Save the updated counts back into CSV.
        output_filename = 'podcast_episode_counts.csv'
        final_df.to_csv(output_filename, index=False)
        
        print(f"\nUpdated CSV '{output_filename}' is successfully written.")
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time/60:.2f} minutes")
        print(f"Total podcasts processed: {processed_count:,}")
        print(f"Successful updates: {success_count:,}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print(f"Process ended at: {datetime.now()}")

if __name__ == "__main__":
    main()
