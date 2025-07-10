import json
from tqdm.auto import tqdm
import os
from collections import defaultdict

def append_new_episodes(original_file, incremental_file, output_file):
    """
    Append only new episodes from incremental file to original file.
    Does not update any existing episodes.
    Only includes episodes from podcasts that exist in the original file.
    """
    print(f"Loading original guest extract from: {original_file}")
    
    # Create sets to track existing episodes and podcasts
    existing_episodes = set()  # Will store "podcast_id:episode_id" keys
    existing_podcasts = set()  # Will store podcast_ids
    original_data = []
    original_count = 0
    
    # First scan the original file
    with open(original_file, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                original_data.append(line)  # Keep original line exactly as is
                
                pid = obj.get('podcast_id')
                eid = obj.get('episode_id')
                if pid and eid:
                    key = f"{pid}:{eid}"
                    existing_episodes.add(key)
                    existing_podcasts.add(pid)
                    original_count += 1
            except json.JSONDecodeError as e:
                print(f"Invalid JSON at line {lineno} in {original_file}: {e}")
                continue
    
    print(f"Original file contains {original_count:,} records from {len(existing_podcasts):,} podcasts")
    
    # Process the incremental file and identify new episodes for existing podcasts
    incremental_count = 0
    new_episodes = []
    
    print(f"Processing incremental file: {incremental_file}")
    with open(incremental_file, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get('podcast_id')
                eid = obj.get('episode_id')
                
                if pid and eid:
                    key = f"{pid}:{eid}"
                    incremental_count += 1
                    
                    # Only add if:
                    # 1. This episode doesn't exist in original file
                    # 2. The podcast exists in the original file
                    if key not in existing_episodes and pid in existing_podcasts:
                        new_episodes.append(line)
                        
            except json.JSONDecodeError as e:
                print(f"Invalid JSON at line {lineno} in {incremental_file}: {e}")
                continue
    
    # Write the combined data to the output file
    print(f"Writing combined data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as out_file:
        # First write all original records (unchanged)
        for line in original_data:
            out_file.write(line + '\n')
        
        # Then append new episodes
        for line in new_episodes:
            out_file.write(line + '\n')
    
    # Generate a summary of what episodes were added per podcast
    podcast_new_episodes = defaultdict(int)
    for line in new_episodes:
        obj = json.loads(line)
        pid = obj.get('podcast_id')
        podcast_new_episodes[pid] += 1
    
    print(f"\nAppend completed:")
    print(f"Original file: {original_count:,} records")
    print(f"Incremental file: {incremental_count:,} records processed")
    print(f"Added {len(new_episodes):,} new episodes for {len(podcast_new_episodes):,} existing podcasts")
    print(f"Combined file contains {original_count + len(new_episodes):,} total records")
    print(f"Saved to: {output_file}")
    
    # Show sample of podcasts with new episodes
    if podcast_new_episodes:
        print("\nSample of podcasts with new episodes:")
        for i, (pid, count) in enumerate(sorted(podcast_new_episodes.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"  Podcast {pid}: +{count} new episodes")
        if len(podcast_new_episodes) > 10:
            print(f"  ...and {len(podcast_new_episodes) - 10} more podcasts")

if __name__ == "__main__":
    # File paths
    original_file = "guests-extract_names_URM.jsonl"
    incremental_file = "2025-03-03-guests-extract_names_mentioned-incremental.jsonl"  # Rename this to match your actual filename
    output_file = "guests-extract_names_URM-e2.jsonl"
    
    # Check if files exist
    if not os.path.exists(original_file):
        print(f"Error: Original file '{original_file}' not found!")
    elif not os.path.exists(incremental_file):
        print(f"Error: Incremental file '{incremental_file}' not found!")
    else:
        append_new_episodes(original_file, incremental_file, output_file)
