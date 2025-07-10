import pandas as pd
from tqdm.auto import tqdm
import json
from openai import OpenAI
import string
import os
import concurrent.futures
import threading
import re
import traceback
from datetime import datetime

# Set up simple logging
def log_error(message):
    with open("host_error_log.txt", "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")

def remove_html_tags(text):
    if not isinstance(text, str):
        return ""
    clean = re.compile('<.*?>')  # Matches anything between '<' and '>'
    return re.sub(clean, ' ', text)

# Initialize OpenAI client with o3-mini model
client = OpenAI(api_key="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")
printable = set(string.printable)

# Host extraction system prompt (adapted from friend's script)
sys_prompt = """As an AI your task is to use information of a podcast 
to extract information about (co-)host(s) with their names explicitly mentioned with the following json structure.
Just provide a single actual number with digits (not string characters).
Provide the outcome in the following parsable json format. No extra output please.
{
    "total_hosts": "number of hosts with their names explicitly mentioned. If not mentioned return NA",
    "urm_hosts": "number of either hispanic or black hosts explicitly mentioned",
    "female_hosts": "number of female hosts with their names explicitly mentioned",
    "explain": "provide short explanation about numbers provided"
}
Note: the number hispanic/black or female hosts is always smaller than or equal to the total number of hosts 
as the number total hosts includes all hosts including but not limited to hispanic/black or female hosts. If that is not the case explain why.
"""

def process_item(row):
    """Process a single podcast episode row and extract host information"""
    # Only try once instead of twice
    episode_title = row["episode_title"] if not pd.isna(row["episode_title"]) else ""
    episode_description = row['episode_description'] if not pd.isna(row['episode_description']) else ""
    podcast_description = row['podcast_description'] if not pd.isna(row['podcast_description']) else ""
    publisher = row['publisher'] if not pd.isna(row['publisher']) else ""
    
    user_prompt = json.dumps({
         "publisher": publisher,
         "podcast_description": podcast_description,
         "episode_title": episode_title,
         "episode_description": episode_description
    })
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model="gpt-4o-mini",
        )
        
        gpt_response = chat_completion.choices[0].message.content
        # Remove markdown code blocks if present
        clean_response = gpt_response.replace('```json', '').replace('```', '')
        r = json.loads(clean_response)

        r_out = {
            "podcast_id": row['podcast_id'],
            "episode_id": row['episode_id'],
            "unique_id": str(row['unique_id']),
            **r
        }
        
        # Check for inconsistencies
        if 'total_hosts' in r and 'urm_hosts' in r and 'female_hosts' in r:
            if r['total_hosts'] != "NA" and int(r['total_hosts']) < int(r['urm_hosts']):
                print(f"Warning - Inconsistency: {row['unique_id']} has more URM hosts than total hosts")
                
            if r['total_hosts'] != "NA" and int(r['total_hosts']) < int(r['female_hosts']):
                print(f"Warning - Inconsistency: {row['unique_id']} has more female hosts than total hosts")
        
        return r_out

    except Exception as e:
        error_msg = f"Error processing {row['podcast_id']}_{row['episode_id']}: {str(e)}"
        log_error(error_msg)
        log_error(traceback.format_exc())
        
        # Return minimal information when there's an error
        return {
            "podcast_id": row['podcast_id'],
            "episode_id": row['episode_id'],
            "unique_id": str(row['unique_id']),
            "error": str(e)
        }

def main():
    # Load existing podcast IDs from podcast_hosts.jsonl
    print("Loading existing podcast IDs from podcast_hosts.jsonl...")
    existing_podcast_ids = set()
    try:
        with open('podcast_hosts.jsonl', 'r', encoding='utf-8') as fin:
            for l in fin:
                try:
                    d = json.loads(l)
                    existing_podcast_ids.add(d.get('podcast_id', ''))
                except Exception as e:
                    log_error(f"Error parsing podcast_hosts.jsonl line: {str(e)}")
                    continue
    except Exception as e:
        log_error(f"Error opening podcast_hosts.jsonl: {str(e)}")
        print(f"Warning: Could not load podcast_hosts.jsonl: {str(e)}")
    
    print(f"Loaded {len(existing_podcast_ids)} podcast IDs from podcast_hosts.jsonl")

    # Load data from our new JSONL file
    print("Loading data from email_one_two_host.jsonl...")
    data = []
    try:
        with open('email_one_two_host.jsonl', 'r', encoding='utf-8') as fin:
            for l in fin:
                try:
                    d = json.loads(l)
                    podcast_id = d.get('podcast_id', '')
                    
                    # Skip this podcast if it already exists in podcast_hosts.jsonl
                    if podcast_id in existing_podcast_ids:
                        continue
                    
                    podcast_description = d.get('description', '')
                    publisher = d.get('publisher', '')
                    for episode in d.get('episodes', []):
                        data.append({
                            'podcast_id': podcast_id,
                            'episode_id': episode.get('episode_id', ''),
                            'unique_id': f"{podcast_id}-{episode.get('episode_id', '')}",
                            'episode_title': episode.get('title', ''),
                            'episode_description': episode.get('description', ''),
                            'podcast_description': podcast_description,
                            'publisher': publisher
                        })
                except Exception as e:
                    log_error(f"Error parsing JSON line: {str(e)}")
                    continue
    except Exception as e:
        log_error(f"Error opening input file: {str(e)}")
        print(f"Error: {str(e)}")
        return

    print(f"Loaded {len(data)} episodes from JSONL file after filtering out existing podcasts")
    
    # Convert to DataFrame and clean HTML
    df = pd.DataFrame(data)
    if df.empty:
        print("No new episodes to process. Exiting.")
        return
        
    df['episode_title'] = df['episode_title'].apply(remove_html_tags)
    df['episode_description'] = df['episode_description'].apply(remove_html_tags)
    df['podcast_description'] = df['podcast_description'].apply(remove_html_tags)
    df['publisher'] = df['publisher'].apply(remove_html_tags)
    
    # Track already processed IDs
    fname = 'hosts-extract-one-two.jsonl'
    existed_ids = set()
    
    # Check if output file exists and load processed IDs
    if os.path.exists(fname):
        try:
            with open(fname, 'r', encoding='utf-8') as fin:
                for l in fin:
                    try:
                        existed_ids.add(json.loads(l).get('unique_id', ''))
                    except Exception as e:
                        log_error(f"Error reading existing output: {str(e)}")
                        continue
            mode = 'a'
            print(f'Already processed: {len(existed_ids):,} episodes')
        except Exception as e:
            log_error(f"Error opening output file: {str(e)}")
            mode = 'w'
    else:
        mode = 'w'
        
    print('Existing IDs loaded!')
    print(f'Before filtering: {df.shape[0]} episodes')
    df = df[~df['unique_id'].isin(existed_ids)]
    print(f'After filtering: {df.shape[0]} episodes to process')
    
    if df.empty:
        print("No new episodes to process. Exiting.")
        return
    
    # Configure parallel processing
    MAX_WORKERS = min(100, df.shape[0])  # Use fewer workers for small datasets
    print(f"Processing with {MAX_WORKERS} parallel workers")
    
    # Process episodes in parallel
    total = df.shape[0]
    with tqdm(total=total, desc="Processing episodes") as pbar:
        with open(fname, mode, encoding='utf-8') as fout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit tasks in batches to avoid memory issues
                batch_size = 1000
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i+batch_size]
                    futures = {executor.submit(process_item, row): idx for idx, row in batch_df.iterrows()}
                    
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if 'total_hosts' in result:
                            try:
                                json.dump(result, fout)
                                fout.write('\n')
                                fout.flush()  # Ensure data is written immediately
                            except Exception as e:
                                log_error(f"Error writing to output file: {str(e)}")
                        
                        pbar.update(1)
    
    print("Processing complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(f"Unhandled exception in main: {str(e)}")
        log_error(traceback.format_exc())
        print(f"Fatal error: {str(e)}")
