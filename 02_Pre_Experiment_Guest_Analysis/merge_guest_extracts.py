import ujson as json
from tqdm.auto import tqdm
import os
import time

def valid_json_lines(file_path):
    """Parse and yield valid JSON lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON at line {lineno} in {file_path}: {e}")
                continue

def merge_guest_extract_files(original_file, incremental_file, output_file):
    """Merge original and incremental guest extract files into a combined file."""
    start_time = time.time()
    print(f"Loading original guest extract from: {original_file}")
    
    # Create a dictionary to track existing records by podcast_id + episode_id
    original_entries = {}
    original_count = 0
    
    # First scan the original file
    for obj in tqdm(valid_json_lines(original_file), desc="Scanning original file"):
        pid = obj.get('podcast_id')
        eid = obj.get('episode_id')
        if pid and eid:
            key = f"{pid}:{eid}"
            original_entries[key] = obj
            original_count += 1
    
    print(f"Original file contains {original_count:,} records")
    
    # Process the incremental file and identify new records
    incremental_count = 0
    new_count = 0
    
    print(f"Processing incremental file: {incremental_file}")
    for obj in tqdm(valid_json_lines(incremental_file), desc="Processing incremental records"):
        pid = obj.get('podcast_id')
        eid = obj.get('episode_id')
        if pid and eid:
            key = f"{pid}:{eid}"
            incremental_count += 1
            
            # If this is a new record or has updated guest information
            if key not in original_entries:
                original_entries[key] = obj
                new_count += 1
    
    # Write the combined records to the output file
    print(f"Writing combined guest data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for obj in tqdm(original_entries.values(), desc="Writing combined records"):
            out_file.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    elapsed_time = time.time() - start_time
    print(f"\nMerge completed in {elapsed_time:.2f} seconds")
    print(f"Incremental file contained {incremental_count:,} records")
    print(f"Added {new_count:,} new records from incremental file")
    print(f"Combined file contains {len(original_entries):,} total records")
    print(f"Saved combined guest extract to: {output_file}")

if __name__ == "__main__":
    # File paths
    original_file = "guests-extract-combined.jsonl"  # Use last week's combined file
    incremental_file = "20205-03-09-guests-extract_names_mentioned-podcast_episodes_incremental.jsonl"  # New data from latest RSS refresh
    output_file = "guests-extract-combined-v2.jsonl"  # New combined file with v2 suffix

    
    # Check if files exist
    if not os.path.exists(original_file):
        print(f"Error: Original file '{original_file}' not found!")
    elif not os.path.exists(incremental_file):
        print(f"Error: Incremental file '{incremental_file}' not found!")
    else:
        merge_guest_extract_files(original_file, incremental_file, output_file)
