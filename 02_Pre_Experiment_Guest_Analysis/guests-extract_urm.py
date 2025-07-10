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
    with open("error_log.txt", "a", encoding="utf-8") as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")

def remove_html_tags(text):
    if not isinstance(text, str):
        return ""
    clean = re.compile('<.*?>')  # Matches anything between '<' and '>'
    return re.sub(clean, ' ', text)

# Initialize OpenAI client with o3-mini model
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
printable = set(string.printable)

sys_prompt = """As an AI your task is to use the description and title of a podcast episode 
to extract information about guests (not hosts, co-hosts, daughters of guests etc) appeared in an episode with their names expliclity mentioned with the following json structure.
Do not say several or multiple, just provide a single actual number with digits (not string characters).
Provide the outcome in the following parsable json format. No extra output please.
{
    "total_guests": "number of guests (not hosts, co-hosts, daughters of guests etc) appeared in the episode with their names expliclity mentioned. DO NOT INCLUDE HOSTS/CO-HOSTS/MODERATORS, FAMILY/RELAVITES/DAUGHTERS OF THE GUESTS ETC IN YOUR COUNT",
    "urm_guests": "number of either hispanic or black guests (not hosts, co-hosts, daughters of guests etc) appeared in the episode with their names explicitly mentioned. DO NOT INCLUDE HOSTS/CO-HOSTS/MODERATORS, FAMILY/RELAVITES/DAUGHTERS OF THE GUESTS ETC IN YOUR COUNT",
    "female_guests": "number of female guests (not hosts, co-hosts, daughters of guests etc) appeared in the episode with their names explicitly mentioned. DO NOT INCLUDE HOSTS/CO-HOSTS/MODERATORS, FAMILY/RELAVITES/DAUGHTERS OF THE GUESTS ETC IN YOUR COUNT",
    "explain": "provide short explanation about numbers provided"
}
Do not include hosts, co-hosts, daughters of guests etc in the counts. Just the guests that actually appeared in the episode.
Note: the number hispanic/black or female guests is always smaller than or equal to the number of guests as the number guests includes all guests including but not limited to hispanic/black or female guests. If that is not the case explain why.
"""

def process_item(row):
    """Process a single podcast episode row and extract guest information"""
    # Only try once instead of twice
    episode_title = row["episode_title"] if not pd.isna(row["episode_title"]) else ""
    episode_description = row['episode_description'] if not pd.isna(row['episode_description']) else ""
    
    user_prompt = json.dumps({
         "Episode title": episode_title,
         "Episode description": episode_description
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
        if 'total_guests' in r and 'urm_guests' in r and 'female_guests' in r:
            if int(r['total_guests']) < int(r['urm_guests']):
                print(f"Warning - Inconsistency: {row['unique_id']} has more URM guests than total guests")
                
            if int(r['total_guests']) < int(r['female_guests']):
                print(f"Warning - Inconsistency: {row['unique_id']} has more female guests than total guests")
        
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
    # Load previously processed podcast IDs
    print("Loading previously processed podcast IDs...")
    processed_podcast_ids = set()
    
    # Check the first existing file
    if os.path.exists('guests-extract_names_URM.jsonl'):
        try:
            with open('guests-extract_names_URM.jsonl', 'r', encoding='utf-8') as fin:
                for l in fin:
                    try:
                        data = json.loads(l)
                        processed_podcast_ids.add(data.get('podcast_id', ''))
                    except Exception as e:
                        log_error(f"Error reading guests-extract_names_URM.jsonl: {str(e)}")
                        continue
            print(f'Loaded {len(processed_podcast_ids)} podcast IDs from guests-extract_names_URM.jsonl')
        except Exception as e:
            log_error(f"Error opening guests-extract_names_URM.jsonl: {str(e)}")
    
    # Check the second existing file
    if os.path.exists('guests-extract_names_URM-e2.jsonl'):
        existing_count = len(processed_podcast_ids)
        try:
            with open('guests-extract_names_URM-e2.jsonl', 'r', encoding='utf-8') as fin:
                for l in fin:
                    try:
                        data = json.loads(l)
                        processed_podcast_ids.add(data.get('podcast_id', ''))
                    except Exception as e:
                        log_error(f"Error reading guests-extract_names_URM-e2.jsonl: {str(e)}")
                        continue
            print(f'Loaded {len(processed_podcast_ids) - existing_count} additional podcast IDs from guests-extract_names_URM-e2.jsonl')
        except Exception as e:
            log_error(f"Error opening guests-extract_names_URM-e2.jsonl: {str(e)}")
    
    print(f'Total unique podcast IDs already processed: {len(processed_podcast_ids)}')
    
    # Load data from JSONL file
    print("Loading data from email_one_two_prod_db.jsonl...")
    data = []
    try:
        with open('email_one_two_prod_db.jsonl', 'r', encoding='utf-8') as fin:
            for l in fin:
                try:
                    d = json.loads(l)
                    podcast_id = d.get('podcast_id', '')
                    
                    # Skip podcasts that have already been processed
                    if podcast_id in processed_podcast_ids:
                        continue
                    
                    for episode in d.get('episodes', []):
                        data.append({
                            'podcast_id': podcast_id,
                            'episode_id': episode.get('episode_id', ''),
                            'unique_id': f"{podcast_id}-{episode.get('episode_id', '')}",
                            'episode_title': episode.get('title', ''),
                            'episode_description': episode.get('description', '')
                        })
                except Exception as e:
                    log_error(f"Error parsing JSON line: {str(e)}")
                    continue
    except Exception as e:
        log_error(f"Error opening input file: {str(e)}")
        print(f"Error: {str(e)}")
        return

    print(f"Loaded {len(data)} episodes from JSONL file after filtering out previously processed podcasts")
    
    # Convert to DataFrame and clean HTML
    df = pd.DataFrame(data)
    if df.empty:
        print("No new episodes to process. Exiting.")
        return
        
    df['episode_title'] = df['episode_title'].apply(remove_html_tags)
    df['episode_description'] = df['episode_description'].apply(remove_html_tags)
    
    # Track already processed episode IDs (for incremental processing)
    output_file = 'guests-extract_names_URM-one-two.jsonl'
    existed_ids = set()
    
    # Check if output file exists and load processed IDs
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as fin:
                for l in fin:
                    try:
                        existed_ids.add(json.loads(l).get('unique_id', ''))
                    except Exception as e:
                        log_error(f"Error reading existing output: {str(e)}")
                        continue
            mode = 'a'
            print(f'Already processed in this run: {len(existed_ids):,} episodes')
        except Exception as e:
            log_error(f"Error opening output file: {str(e)}")
            mode = 'w'
    else:
        mode = 'w'
        
    print('Existing episode IDs loaded!')
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
        with open(output_file, mode, encoding='utf-8') as fout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit tasks in batches to avoid memory issues
                batch_size = 1000
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i+batch_size]
                    futures = {executor.submit(process_item, row): idx for idx, row in batch_df.iterrows()}
                    
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if 'total_guests' in result:
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
