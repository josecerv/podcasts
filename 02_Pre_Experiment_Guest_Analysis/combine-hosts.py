import json
from collections import OrderedDict

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of JSON objects.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Returning empty list.")
        return []

def get_podcast_id(item):
    """
    Returns the podcast id from the item.
    It will check for 'podcastID' and 'podcast_id' keys.
    If neither is found, it returns None.
    """
    if 'podcastID' in item:
        return item['podcastID']
    elif 'podcast_id' in item:
        return item['podcast_id']
    return None

def main():
    # Load data from both files.
    existing_hosts_data = read_jsonl('podcast_hosts.jsonl')
    new_hosts_data = read_jsonl('hosts-extract-one-two.jsonl')
    
    combined_data = OrderedDict()
    overlapping_ids = []  # to track overlapping podcast IDs

    # First, add all existing podcast host data
    print(f"Processing {len(existing_hosts_data)} items from podcast_hosts.jsonl")
    for item in existing_hosts_data:
        podcast_id = get_podcast_id(item)
        if podcast_id is None:
            print("Warning: An item in podcast_hosts.jsonl is missing a podcast ID:", item)
            continue
        
        combined_data[podcast_id] = item

    # Then add new host extraction data
    print(f"Processing {len(new_hosts_data)} items from hosts-extract-one-two.jsonl")
    for item in new_hosts_data:
        podcast_id = get_podcast_id(item)
        if podcast_id is None:
            print("Warning: An item in hosts-extract-one-two.jsonl is missing a podcast ID:", item)
            continue
        
        if podcast_id in combined_data:
            overlapping_ids.append(podcast_id)
            print(f"Warning: Overlap found for podcast ID: {podcast_id}")
            # In this case, we prefer the data from hosts-extract-one-two.jsonl
            combined_data[podcast_id] = item
        else:
            combined_data[podcast_id] = item

    # If overlaps were detected, print them to let you know which IDs had conflicts.
    if overlapping_ids:
        print(f"\n{len(overlapping_ids)} overlapping podcast IDs detected.")
        print("These records have been updated with the latest extraction data.")
    else:
        print("\nNo overlapping podcast IDs found.")

    # Write the combined data into a JSONL file.
    with open('podcast_hosts_combined.jsonl', 'w', encoding='utf-8') as outfile:
        for item in combined_data.values():
            json.dump(item, outfile)
            outfile.write('\n')
    
    # Also create a backup of the original podcast_hosts.jsonl if it exists
    import os
    import shutil
    if os.path.exists('podcast_hosts.jsonl'):
        shutil.copy2('podcast_hosts.jsonl', 'podcast_hosts.jsonl.bak')
    
    # Rename the new file to podcast_hosts.jsonl
    os.rename('podcast_hosts_combined.jsonl', 'podcast_hosts.jsonl')

    print(f"\nCombined JSONL file created as 'podcast_hosts.jsonl' with {len(combined_data)} entries.")
    print("A backup of the original file was created as 'podcast_hosts.jsonl.bak'.")

if __name__ == '__main__':
    main()
