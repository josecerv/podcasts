#!/usr/bin/env python3
"""
Post-Intervention Podcast Guest Analysis Pipeline (FULL RUN)

Modifications:
- Switched RSS fetching to use direct feedparser.parse(url) and socket.setdefaulttimeout.
- Moved startup prints to if __name__ == "__main__".
- Optionally sets a global feedparser.USER_AGENT.
- Refined logging for CharacterEncodingOverride.
- Retained detailed logging for other feed issues.
"""
import sys
import os
# Startup prints will be moved to the main execution block
import feedparser
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
import hashlib
import time
import re
import multiprocessing
import gc
from functools import partial # Not explicitly used but good for potential future use
from typing import List, Dict, Optional
import concurrent.futures
import signal
import socket # For setdefaulttimeout
import json
from tqdm.auto import tqdm
from openai import OpenAI
import string
from scipy import stats
import numpy as np

# --- Configuration ---
INPUT_CSV = 'podcast_rss_export.csv'
RAW_LLM_OUTPUT_JSONL = 'post_intervention_guest_classification_full.jsonl'
AGGREGATED_OUTPUT_CSV = 'post_intervention_guest_summary_full.csv'
ERROR_LOG_FILE = "post_intervention_error_log_full.txt"

# RSS Feed Processing Config
RSS_TIMEOUT = 25 # Timeout for feedparser.parse (via socket.setdefaulttimeout)
MIN_REQUIRED_EPISODES = 1

# Set a common User-Agent for feedparser globally.
# Some servers might block requests with feedparser's default UA or generic Python UA.
# This can be commented out if not needed or causing issues.
feedparser.USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 PodcastScraper/1.0"

# LLM Config
OPENAI_API_KEY = "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" # Replace or use env var
LLM_MODEL = "gpt-4o-mini"
MAX_LLM_WORKERS = 20
LLM_RETRY_ATTEMPTS = 2
LLM_RETRY_DELAY = 5

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(ERROR_LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

printable = set(string.printable)

# --- Helper Functions ---
def clean_text(text):
    try:
        if not isinstance(text, str): text = str(text)
        text = re.sub(r'<.*?>', ' ', text)
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&quot;', '"').replace('&apos;', "'").replace('&nbsp;', ' ')
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.debug(f"Error cleaning text: '{str(text)[:50]}...'. Error: {e}")
        return ''

def generate_episode_id(podcast_id, entry):
    guid = entry.get('guid', None)
    if guid and isinstance(guid, str) and len(guid) > 10:
        return hashlib.sha256(guid.encode('utf-8', 'ignore')).hexdigest()[:64]
    components = [
        str(podcast_id),
        clean_text(entry.get('title', '')),
        str(entry.get('published_parsed', '')),
        clean_text(entry.get('link', ''))[:100]
    ]
    unique_str = ''.join(str(c) for c in components if c is not None).encode('utf-8', 'ignore')
    return hashlib.sha256(unique_str).hexdigest()

def get_robust_description(entry):
    parts = []
    if 'content' in entry and entry.content:
        for content_item in entry.content:
            if hasattr(content_item, 'value') and content_item.value: parts.append(content_item.value)
    for key in ['itunes_summary', 'description', 'summary', 'subtitle', 'itunes_subtitle']:
        value = entry.get(key)
        if value and isinstance(value, str): parts.append(value)
    summary_detail = entry.get('summary_detail')
    if summary_detail and isinstance(summary_detail, dict):
        value = summary_detail.get('value')
        if value and isinstance(value, str): parts.append(value)
    media_description = entry.get('media_description')
    if media_description and isinstance(media_description, str): parts.append(media_description)
    return clean_text(' '.join(list(dict.fromkeys(parts))).strip())


# --- RSS Feed Processing Logic (FastCounter Style) ---
def fetch_and_filter_episodes(podcast_info):
    rss_url = podcast_info.get('rss')
    podcast_id = podcast_info.get('podcastID')
    
    try:
        completion_timestamp_val = podcast_info.get('completion_timestamp')
        current_timestamp = datetime.now(timezone.utc)

        if not rss_url or not podcast_id or completion_timestamp_val is None or pd.isna(completion_timestamp_val):
            logger.info(f"Skipping {podcast_id} (URL: {rss_url if rss_url else 'N/A'}): Missing essential info (rss, podcastID, or completion_timestamp).")
            return []
            
        episodes = []
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(RSS_TIMEOUT)
        parsed_feed = None

        try:
            logger.debug(f"Fetching and parsing RSS for {podcast_id} from {rss_url} using feedparser.parse(url)")
            # feedparser.parse will use the globally set USER_AGENT if defined
            parsed_feed = feedparser.parse(rss_url)

            if parsed_feed.bozo:
                bozo_type = type(parsed_feed.bozo_exception).__name__
                bozo_msg = str(parsed_feed.bozo_exception)
                
                if isinstance(parsed_feed.bozo_exception, feedparser.CharacterEncodingOverride):
                    # This is often benign, log as DEBUG if entries are found, INFO if no entries
                    log_level = logging.DEBUG if parsed_feed.entries else logging.INFO
                    logger.log(log_level, f"Feed {podcast_id} ({rss_url}): Bozo due to CharacterEncodingOverride. Msg: {bozo_msg[:100]}. Processing entries if any.")
                elif isinstance(parsed_feed.bozo_exception, Exception):
                    logger.warning(f"Feed {podcast_id} ({rss_url}): Bozo. Exception Type: {bozo_type}, Msg: {bozo_msg[:200]}")
                else:
                     logger.warning(f"Feed {podcast_id} ({rss_url}): Bozo (non-exception). Type: {bozo_type}, Details: {bozo_msg[:200]}")

                if not parsed_feed.entries and not isinstance(parsed_feed.bozo_exception, feedparser.CharacterEncodingOverride):
                    logger.warning(f"Bozo feed {podcast_id} ({rss_url}) has no entries and is not CharacterEncodingOverride. Skipping.")
                    return []
            
            if not parsed_feed.entries: 
                logger.info(f"No entries found in feed for {podcast_id} ({rss_url}) after parsing.")
                return []

            logger.debug(f"Processing {len(parsed_feed.entries)} entries for {podcast_id} ({rss_url})")
            for entry_idx, entry in enumerate(parsed_feed.entries):
                published_parsed_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                if not published_parsed_struct: 
                    logger.debug(f"P:{podcast_id} E:{entry_idx+1} - Skipping: no published_parsed or updated_parsed. Title: {entry.get('title', 'N/A')[:50]}")
                    continue
                try:
                    published_date = datetime(*published_parsed_struct[:6], tzinfo=timezone.utc)
                except (ValueError, TypeError) as e:
                    logger.info(f"P:{podcast_id} E:{entry_idx+1} - Skipping: invalid date '{published_parsed_struct}'. Error: {e}. Title: {entry.get('title', 'N/A')[:50]}")
                    continue
                    
                if published_date > completion_timestamp_val and published_date <= current_timestamp:
                    description = get_robust_description(entry)
                    title = clean_text(entry.get('title', ''))
                    if not description and not title: 
                        logger.debug(f"P:{podcast_id} E:{entry_idx+1} (Pub: {published_date.date()}) - Skipping: no description and no title.")
                        continue
                    
                    episode_id_val = generate_episode_id(str(podcast_id), entry)
                    episodes.append({
                        'podcast_id': str(podcast_id), 'episode_id': episode_id_val, 
                        'unique_id': f"{str(podcast_id)}-{episode_id_val}", 
                        'episode_title': title, 'episode_description': description, 
                        'episode_published_utc': published_date
                    })

        except socket.timeout:
            logger.warning(f"Socket timeout ({RSS_TIMEOUT}s) fetching/parsing RSS for {podcast_id} ({rss_url})")
        except Exception as e:
            logger.error(f"Error during feedparser.parse or entry processing for {podcast_id} ({rss_url}): {type(e).__name__} - {e}", exc_info=True)
        finally:
            socket.setdefaulttimeout(original_timeout)
        
        if not episodes and parsed_feed and parsed_feed.entries:
             logger.info(f"No episodes matched date/content criteria for {podcast_id} ({rss_url}) from {len(parsed_feed.entries)} entries.")
        elif episodes:
             logger.debug(f"Found {len(episodes)} relevant episodes for {podcast_id} ({rss_url}).")
        return episodes
    except Exception as outer_e:
        logger.error(f"Outer error in fetch_and_filter_episodes for {podcast_id} (URL: {rss_url if rss_url else 'N/A'}): {type(outer_e).__name__} - {outer_e}", exc_info=True)
        return []

# --- Simpler worker function for multiprocessing ---
def process_podcast(podcast_dict):
    podcast_id_for_log = podcast_dict.get('podcastID', 'Unknown PodcastID')
    try:
        return fetch_and_filter_episodes(podcast_dict)
    except Exception as e:
        logger.error(f"Critical error in process_podcast wrapper for P_ID: {podcast_id_for_log}: {type(e).__name__} - {e}", exc_info=True)
        return []

# --- LLM Classification Logic ---
sys_prompt_llm = """As an AI your task is to use the description and title of a podcast episode
to extract information about guests (not hosts, co-hosts, daughters of guests etc) appeared in an episode with their names explicitly mentioned with the following json structure.
Do not say several or multiple, just provide a single actual number with digits (not string characters).
Provide the outcome in the following parsable json format. No extra output please.
{
    "total_guests": "number of guests (not hosts, co-hosts, daughters of guests etc) appeared in the episode with their names explicitly mentioned. DO NOT INCLUDE HOSTS/CO-HOSTS/MODERATORS, FAMILY/RELATIVES/DAUGHTERS OF THE GUESTS ETC IN YOUR COUNT. If no guests, return 0.",
    "urm_guests": "number of either hispanic or black guests (not hosts, co-hosts, daughters of guests etc) appeared in the episode with their names explicitly mentioned. DO NOT INCLUDE HOSTS/CO-HOSTS/MODERATORS, FAMILY/RELATIVES/DAUGHTERS OF THE GUESTS ETC IN YOUR COUNT. If no URM guests, return 0.",
    "female_guests": "number of female guests (not hosts, co-hosts, daughters of guests etc) appeared in the episode with their names explicitly mentioned. DO NOT INCLUDE HOSTS/CO-HOSTS/MODERATORS, FAMILY/RELATIVES/DAUGHTERS OF THE GUESTS ETC IN YOUR COUNT. If no female guests, return 0.",
    "explain": "provide short explanation about numbers provided, especially if counts are zero or there are any ambiguities."
}
Do not include hosts, co-hosts, daughters of guests etc in the counts. Just the guests that actually appeared in the episode.
Note: the number hispanic/black or female guests is always smaller than or equal to the number of guests as the number guests includes all guests including but not limited to hispanic/black or female guests. If that is not the case explain why. If any count is zero, explicitly state 0.
"""

def classify_episode_guests(episode_info):
    client = OpenAI(api_key=OPENAI_API_KEY)
    episode_title = episode_info.get("episode_title", "")
    episode_description = episode_info.get("episode_description", "")
    clean_title = "".join(filter(lambda x: x in printable, episode_title))
    clean_description = "".join(filter(lambda x: x in printable, episode_description))
    max_desc_len = 8000
    if len(clean_description) > max_desc_len: 
        clean_description = clean_description[:max_desc_len] + "..."
    user_prompt = json.dumps({"Episode title": clean_title, "Episode description": clean_description})
    base_result = {
        "podcast_id": episode_info['podcast_id'], "episode_id": episode_info['episode_id'], 
        "unique_id": episode_info['unique_id'], 
        "episode_published_utc": episode_info['episode_published_utc'].isoformat() if episode_info['episode_published_utc'] else None, 
        "total_guests": None, "urm_guests": None, "female_guests": None, "explain": None, "llm_error": None
    }
    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt_llm},
                    {"role": "user", "content": user_prompt}
                ], model=LLM_MODEL, response_format={"type": "json_object"}
            )
            gpt_response = chat_completion.choices[0].message.content
            clean_response = gpt_response.strip()
            if clean_response.startswith("```json"): clean_response = clean_response[7:]
            if clean_response.endswith("```"): clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            parsed_data = json.loads(clean_response)
            try: total_guests = int(parsed_data.get('total_guests', 0))
            except (ValueError, TypeError): total_guests = 0; logger.warning(f"LLM for {episode_info['unique_id']}: Invalid total_guests '{parsed_data.get('total_guests')}', defaulting to 0.")
            try: urm_guests = int(parsed_data.get('urm_guests', 0))
            except (ValueError, TypeError): urm_guests = 0; logger.warning(f"LLM for {episode_info['unique_id']}: Invalid urm_guests '{parsed_data.get('urm_guests')}', defaulting to 0.")
            try: female_guests = int(parsed_data.get('female_guests', 0))
            except (ValueError, TypeError): female_guests = 0; logger.warning(f"LLM for {episode_info['unique_id']}: Invalid female_guests '{parsed_data.get('female_guests')}', defaulting to 0.")
            if not (0 <= urm_guests <= total_guests and 0 <= female_guests <= total_guests):
                logger.warning(f"LLM for {episode_info['unique_id']}: Guest count inconsistency. T:{total_guests}, U:{urm_guests}, F:{female_guests}. Explain: {parsed_data.get('explain','')}")
            base_result.update({
                "total_guests": total_guests, "urm_guests": urm_guests, "female_guests": female_guests, 
                "explain": parsed_data.get("explain", ""), "llm_error": None
            })
            return base_result
        except json.JSONDecodeError as e: 
            error_msg = f"LLM JSONDecodeError attempt {attempt+1}/{LLM_RETRY_ATTEMPTS} for {episode_info['unique_id']}: {e}. Response: '{clean_response[:100]}...'"
            base_result["llm_error"] = error_msg; logger.warning(error_msg)
        except Exception as e: 
            error_msg = f"LLM API/Other Error attempt {attempt+1}/{LLM_RETRY_ATTEMPTS} for {episode_info['unique_id']}: {type(e).__name__} - {e}"
            base_result["llm_error"] = error_msg; logger.warning(error_msg)
        if attempt < LLM_RETRY_ATTEMPTS - 1: 
            logger.info(f"Retrying LLM for {episode_info['unique_id']} after {LLM_RETRY_DELAY * (attempt + 1)}s delay.")
            time.sleep(LLM_RETRY_DELAY * (attempt + 1))
    logger.error(f"LLM classification failed permanently for {episode_info['unique_id']} after {LLM_RETRY_ATTEMPTS} attempts. Last error: {base_result['llm_error']}")
    return base_result

# --- Aggregation Logic ---
def aggregate_results(llm_output_file):
    logger.info(f"Starting aggregation from {llm_output_file}...")
    results = []
    try:
        with open(llm_output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try: results.append(json.loads(line))
                except json.JSONDecodeError: 
                    logger.warning(f"Skipping invalid JSON on line {line_num+1} in {llm_output_file}: {line[:100]}...")
                    continue
    except FileNotFoundError: 
        logger.error(f"LLM output file not found: {llm_output_file}"); return pd.DataFrame()
    except Exception as e: 
        logger.error(f"Error reading LLM output file {llm_output_file}: {e}", exc_info=True); return pd.DataFrame()
    if not results: 
        logger.warning("No results loaded from LLM output file for aggregation."); return pd.DataFrame()
    logger.info(f"Loaded {len(results)} records from LLM output file. Converting to DataFrame...")
    df = pd.DataFrame(results); del results; gc.collect()
    logger.info("DataFrame created from LLM results.")
    count_cols = ['total_guests', 'urm_guests', 'female_guests']
    for col in count_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    valid_rows_mask = df[count_cols].notna().all(axis=1) & df['llm_error'].isna()
    num_valid_llm_classifications = valid_rows_mask.sum()
    logger.info(f"{num_valid_llm_classifications} episodes have valid LLM classifications out of {len(df)} total from LLM output.")
    if not num_valid_llm_classifications > 0:
        logger.warning("No valid LLM classifications found for aggregation."); return pd.DataFrame()
    df_valid = df[valid_rows_mask].copy(); del df; gc.collect()
    num_podcasts_with_valid_llm_episodes = df_valid['podcast_id'].nunique()
    logger.info(f"Aggregating based on {len(df_valid)} valid episodes from {num_podcasts_with_valid_llm_episodes} podcasts.")
    df_valid.loc[:, 'ep_pct_female'] = np.divide(df_valid['female_guests'], df_valid['total_guests'], 
                                              out=np.full_like(df_valid['female_guests'], np.nan, dtype=float), 
                                              where=df_valid['total_guests'] > 0) * 100
    df_valid.loc[:, 'ep_pct_urm'] = np.divide(df_valid['urm_guests'], df_valid['total_guests'], 
                                           out=np.full_like(df_valid['urm_guests'], np.nan, dtype=float), 
                                           where=df_valid['total_guests'] > 0) * 100
    df_valid.replace([np.inf, -np.inf], np.nan, inplace=True)
    summary = df_valid.groupby('podcast_id').agg(
        post_total_episodes_analyzed=pd.NamedAgg(column='episode_id', aggfunc='nunique'),
        post_total_guests_sum=pd.NamedAgg(column='total_guests', aggfunc='sum'),
        post_urm_guests_sum=pd.NamedAgg(column='urm_guests', aggfunc='sum'),
        post_female_guests_sum=pd.NamedAgg(column='female_guests', aggfunc='sum'),
        post_avg_ep_pct_female=pd.NamedAgg(column='ep_pct_female', aggfunc=lambda x: np.nanmean(x) if not x.isnull().all() else 0.0),
        post_avg_ep_pct_urm=pd.NamedAgg(column='ep_pct_urm', aggfunc=lambda x: np.nanmean(x) if not x.isnull().all() else 0.0)
    ).reset_index()
    del df_valid; gc.collect()
    summary['post_overall_pct_urm'] = (summary['post_urm_guests_sum'] / summary['post_total_guests_sum'] * 100).where(summary['post_total_guests_sum'] > 0, 0.0)
    summary['post_overall_pct_female'] = (summary['post_female_guests_sum'] / summary['post_total_guests_sum'] * 100).where(summary['post_total_guests_sum'] > 0, 0.0)
    summary.rename(columns={'post_total_guests_sum': 'post_total_guests', 'post_urm_guests_sum': 'post_urm_guests', 
                            'post_female_guests_sum': 'post_female_guests', 'podcast_id': 'podcastID'}, inplace=True)
    float_cols = ['post_avg_ep_pct_female', 'post_avg_ep_pct_urm', 'post_overall_pct_urm', 'post_overall_pct_female']
    for col in float_cols: summary[col] = summary[col].astype(float).fillna(0.0)
    logger.info(f"Aggregation complete. Produced aggregated data for {len(summary)} podcasts.")
    return summary

# --- Final Summary Function ---
def summarize_by_condition(aggregated_file, input_file):
    podcasts_in_final_summary = 0
    try:
        agg_df = pd.read_csv(aggregated_file)
        if 'podcastID' not in agg_df.columns: 
            logger.error(f"Required column 'podcastID' not found in aggregated file {aggregated_file}"); return 0
    except FileNotFoundError: 
        logger.error(f"Aggregated results file not found: {aggregated_file}"); return 0
    except Exception as e: 
        logger.error(f"Error reading aggregated file {aggregated_file}: {e}", exc_info=True); return 0
    try:
        input_df = pd.read_csv(input_file, usecols=['podcastID', 'treatment'])
        if 'treatment' not in input_df.columns: 
            logger.error(f"Required column 'treatment' not found in input file {input_file}"); return 0
    except FileNotFoundError: 
        logger.error(f"Input CSV file not found: {input_file}"); return 0
    except ValueError as ve: 
        logger.error(f"Error with columns in {input_file} (is 'treatment' or 'podcastID' missing?): {ve}"); return 0
    except Exception as e: 
        logger.error(f"Error reading input file {input_file}: {e}", exc_info=True); return 0
    summary_df = pd.merge(input_df, agg_df, on='podcastID', how='inner'); del input_df, agg_df; gc.collect()
    podcasts_in_final_summary = summary_df['podcastID'].nunique()
    logger.info(f"Merged data successfully. {podcasts_in_final_summary} podcasts included in the final condition summary.")
    if summary_df.empty: 
        logger.warning("No podcasts remaining after merging. Cannot generate condition summary."); return 0
    condition_summary = summary_df.groupby('treatment').agg(
        n_podcasts=pd.NamedAgg(column='podcastID', aggfunc='nunique'),
        mean_post_overall_pct_female=pd.NamedAgg(column='post_overall_pct_female', aggfunc='mean'),
        sd_post_overall_pct_female=pd.NamedAgg(column='post_overall_pct_female', aggfunc='std'),
        mean_post_overall_pct_urm=pd.NamedAgg(column='post_overall_pct_urm', aggfunc='mean'),
        sd_post_overall_pct_urm=pd.NamedAgg(column='post_overall_pct_urm', aggfunc='std'),
        mean_post_avg_ep_pct_female=pd.NamedAgg(column='post_avg_ep_pct_female', aggfunc='mean'),
        sd_post_avg_ep_pct_female=pd.NamedAgg(column='post_avg_ep_pct_female', aggfunc='std'),
        mean_post_avg_ep_pct_urm=pd.NamedAgg(column='post_avg_ep_pct_urm', aggfunc='mean'),
        sd_post_avg_ep_pct_urm=pd.NamedAgg(column='post_avg_ep_pct_urm', aggfunc='std')
    ).reset_index()
    output_lines = ["\n--- Post-Intervention Guest Diversity Summary by Condition ---", 
                    f"(Based on podcasts with >= {MIN_REQUIRED_EPISODES} post-intervention episode(s) analyzed and valid LLM data)"]
    for _, row in condition_summary.iterrows():
        group = "Treatment" if row['treatment'] == 1 else "Control"
        output_lines.append(f"\nGroup: {group} (N = {row['n_podcasts']:.0f} podcasts)")
        output_lines.append(f"  Overall (M1) Avg. % Female Guests: {row['mean_post_overall_pct_female']:.2f}% (SD={row['sd_post_overall_pct_female']:.2f})")
        output_lines.append(f"  Overall (M1) Avg. % URM Guests:    {row['mean_post_overall_pct_urm']:.2f}% (SD={row['sd_post_overall_pct_urm']:.2f})")
        output_lines.append(f"  Avg. Per-Episode (M2) % Female Guests:  {row['mean_post_avg_ep_pct_female']:.2f}% (SD={row['sd_post_avg_ep_pct_female']:.2f})")
        output_lines.append(f"  Avg. Per-Episode (M2) % URM Guests:     {row['mean_post_avg_ep_pct_urm']:.2f}% (SD={row['sd_post_avg_ep_pct_urm']:.2f})")
    output_lines.append("----------------------------------------------------------\n--- T-Test Results (Treatment vs. Control) ---")
    control_group = summary_df[summary_df['treatment'] == 0]; treatment_group = summary_df[summary_df['treatment'] == 1]
    if len(control_group) < 2 or len(treatment_group) < 2: 
        output_lines.append("  Not enough data for t-tests (need at least 2 podcasts in each group).")
    else:
        metrics_for_ttest = {"Overall % Female (M1)": "post_overall_pct_female", "Overall % URM (M1)": "post_overall_pct_urm", 
                             "Avg. Per-Episode % Female (M2)": "post_avg_ep_pct_female", "Avg. Per-Episode % URM (M2)": "post_avg_ep_pct_urm"}
        for name, col in metrics_for_ttest.items():
            ctrl_vals = control_group[col].dropna(); treat_vals = treatment_group[col].dropna()
            if len(ctrl_vals) < 2 or len(treat_vals) < 2: 
                output_lines.append(f"  {name}: Not enough non-NaN data for t-test (Control N={len(ctrl_vals)}, Treatment N={len(treat_vals)})."); continue
            try:
                t_stat, p_val = stats.ttest_ind(treat_vals, ctrl_vals, equal_var=False, nan_policy='omit')
                output_lines.append(f"  {name}: t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
            except Exception as e: output_lines.append(f"  {name}: T-test Error - {e}")
    output_lines.append("----------------------------------------------------------")
    final_summary_text = '\n'.join(output_lines)
    print(final_summary_text)
    logger.info("Final summary by condition (including t-tests) printed to console.")
    with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as err_f:
        err_f.write("\n\n" + "="*20 + " FINAL SUMMARY OUTPUT " + "="*20 + "\n" + final_summary_text + "\n" + "="*60 + "\n\n")
    return podcasts_in_final_summary

# --- Main Pipeline Execution ---
def main():
    start_time = time.time()
    logger.info(f"--- Pipeline Start --- Date: {datetime.now().isoformat()} ---")
    podcast_counts = {
        "initial_csv_rows": 0, "after_essential_dropna": 0, 
        "with_any_fetched_episodes": 0, "total_relevant_episodes_for_llm": 0,
        "meeting_min_episode_criteria": 0, "final_episodes_for_llm": 0,
        "with_aggregated_llm_data": 0, "in_final_condition_summary": 0
    }
    try:
        podcasts_df_initial = pd.read_csv(INPUT_CSV)
        podcast_counts["initial_csv_rows"] = len(podcasts_df_initial)
        logger.info(f"Loaded {podcast_counts['initial_csv_rows']} rows from {INPUT_CSV}.")
        if 'treatment' not in podcasts_df_initial.columns: 
            logger.critical(f"CRITICAL: 'treatment' column missing in {INPUT_CSV}! Pipeline cannot continue."); return
        podcasts_df = podcasts_df_initial.copy(); del podcasts_df_initial; gc.collect()
        podcasts_df['latest_response_time'] = pd.to_datetime(podcasts_df['latest_response_time'], errors='coerce', utc=True)
        essential_cols = ['podcastID', 'rss', 'latest_response_time', 'treatment']
        initial_rows_before_dropna = len(podcasts_df)
        podcasts_df.dropna(subset=essential_cols, inplace=True)
        podcast_counts["after_essential_dropna"] = len(podcasts_df)
        logger.info(f"{podcast_counts['after_essential_dropna']} rows remaining after dropping rows with missing essential data (was {initial_rows_before_dropna}).")
    except FileNotFoundError: 
        logger.critical(f"CRITICAL: Input CSV '{INPUT_CSV}' not found."); return
    except Exception as e: 
        logger.critical(f"CRITICAL: Error loading/processing {INPUT_CSV}: {e}", exc_info=True); return
    if podcasts_df.empty: logger.info("No podcasts to process after initial load. Exiting."); return
    logger.info(f"Starting RSS feed fetching for {len(podcasts_df)} podcasts...")
    podcasts_df_renamed = podcasts_df.rename(columns={'latest_response_time': 'completion_timestamp'}); del podcasts_df; gc.collect()
    podcast_info_list = podcasts_df_renamed.to_dict('records'); del podcasts_df_renamed; gc.collect()
    BATCH_SIZE = 500 
    total_podcasts_to_fetch = len(podcast_info_list)
    all_relevant_episodes = []
    num_rss_workers = min(8, os.cpu_count() or 4)
    logger.info(f"Using {num_rss_workers} workers for RSS fetching in batches of {BATCH_SIZE}.")
    with tqdm(total=total_podcasts_to_fetch, desc="Fetching RSS feeds") as pbar:
        for i in range(0, total_podcasts_to_fetch, BATCH_SIZE):
            batch_podcast_info = podcast_info_list[i:i + BATCH_SIZE]
            if not batch_podcast_info: continue
            batch_episode_lists = []
            try:
                with multiprocessing.Pool(processes=num_rss_workers) as pool:
                    batch_episode_lists = list(pool.imap_unordered(process_podcast, batch_podcast_info))
                for episode_list_for_podcast in batch_episode_lists:
                    if episode_list_for_podcast: all_relevant_episodes.extend(episode_list_for_podcast)
                pbar.update(len(batch_podcast_info))
            except Exception as e_pool: logger.error(f"Error during multiprocessing batch {i//BATCH_SIZE + 1}: {e_pool}", exc_info=True)
            finally: del batch_podcast_info, batch_episode_lists; gc.collect()
    del podcast_info_list; gc.collect()
    podcast_counts["total_relevant_episodes_for_llm"] = len(all_relevant_episodes)
    if all_relevant_episodes:
        temp_ep_df_for_count = pd.DataFrame(all_relevant_episodes, columns=['podcast_id'])
        podcast_counts["with_any_fetched_episodes"] = temp_ep_df_for_count['podcast_id'].nunique()
        del temp_ep_df_for_count; gc.collect()
        logger.info(f"Fetched {podcast_counts['total_relevant_episodes_for_llm']} relevant episodes from {podcast_counts['with_any_fetched_episodes']} unique podcasts.")
    else: logger.info("No relevant episodes found after RSS fetching and date filtering.")
    if not all_relevant_episodes:
        logger.warning("No episodes fetched. Skipping LLM classification and aggregation.")
    else:
        logger.info("Converting fetched episode list to DataFrame...")
        episodes_df = pd.DataFrame(all_relevant_episodes); del all_relevant_episodes; gc.collect()
        logger.info(f"DataFrame episodes_df created with {len(episodes_df)} rows before min episode filtering.")
        if not episodes_df.empty:
            episode_counts_per_podcast = episodes_df.groupby('podcast_id')['episode_id'].nunique()
            podcasts_meeting_min_criteria_ids = episode_counts_per_podcast[episode_counts_per_podcast >= MIN_REQUIRED_EPISODES].index
            episodes_to_process_df = episodes_df[episodes_df['podcast_id'].isin(podcasts_meeting_min_criteria_ids)].copy()
            podcast_counts["meeting_min_episode_criteria"] = len(podcasts_meeting_min_criteria_ids)
            podcast_counts["final_episodes_for_llm"] = len(episodes_to_process_df)
            logger.info(f"{podcast_counts['meeting_min_episode_criteria']} podcasts meet >= {MIN_REQUIRED_EPISODES} episode(s) criterion.")
            logger.info(f"{podcast_counts['final_episodes_for_llm']} episodes from these podcasts for LLM classification.")
            del episodes_df, episode_counts_per_podcast, podcasts_meeting_min_criteria_ids; gc.collect()
        else:
            logger.info("episodes_df was empty before min episode criteria check."); episodes_to_process_df = pd.DataFrame()
        if not episodes_to_process_df.empty:
            logger.info(f"Starting LLM classification for {len(episodes_to_process_df)} episodes...")
            episodes_to_process_list = episodes_to_process_df.to_dict('records'); del episodes_to_process_df; gc.collect()
            actual_llm_workers = min(MAX_LLM_WORKERS, len(episodes_to_process_list)); actual_llm_workers = max(1, actual_llm_workers) if episodes_to_process_list else 0
            if actual_llm_workers > 0:
                logger.info(f"Using {actual_llm_workers} LLM workers.")
                with open(RAW_LLM_OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_llm_workers) as executor:
                        future_to_uid = {executor.submit(classify_episode_guests, ep_info): ep_info['unique_id'] for ep_info in episodes_to_process_list}
                        for future in tqdm(concurrent.futures.as_completed(future_to_uid), total=len(future_to_uid), desc="Classifying episodes"):
                            uid = future_to_uid[future]
                            try: result = future.result(); fout.write(json.dumps(result) + '\n')
                            except Exception as e_future: 
                                logger.error(f"Unhandled error for LLM future UID {uid}: {e_future}", exc_info=True)
                                fout.write(json.dumps({"unique_id": uid, "llm_error": f"Future error: {type(e_future).__name__}"}) + '\n')
                    fout.flush()
            else: logger.info("No LLM workers or no episodes, skipping LLM ThreadPool.")
            del episodes_to_process_list; gc.collect()
            logger.info("LLM classification stage complete.")
        else: logger.info("No episodes for LLM classification after filtering.")
    aggregated_df = aggregate_results(RAW_LLM_OUTPUT_JSONL)
    if not aggregated_df.empty:
        podcast_counts["with_aggregated_llm_data"] = aggregated_df['podcastID'].nunique()
        try: 
            aggregated_df.to_csv(AGGREGATED_OUTPUT_CSV, index=False, encoding='utf-8')
            logger.info(f"Aggregated results for {podcast_counts['with_aggregated_llm_data']} podcasts saved to {AGGREGATED_OUTPUT_CSV}")
        except Exception as e: logger.error(f"Failed to save aggregated results: {e}", exc_info=True)
    else: logger.warning("No data in aggregated_df. Aggregated output not saved.")
    del aggregated_df; gc.collect()
    if os.path.exists(AGGREGATED_OUTPUT_CSV) and os.path.getsize(AGGREGATED_OUTPUT_CSV) > 0 and \
       os.path.exists(INPUT_CSV) and os.path.getsize(INPUT_CSV) > 0:
        podcasts_in_summary = summarize_by_condition(AGGREGATED_OUTPUT_CSV, INPUT_CSV)
        podcast_counts["in_final_condition_summary"] = podcasts_in_summary
    else:
        logger.warning(f"Could not generate final summary. Check existence/content of '{AGGREGATED_OUTPUT_CSV}' and '{INPUT_CSV}'.")
    summary_log_message = (
        f"\n--- Pipeline Run Summary --- Date: {datetime.now().isoformat()} ---\n"
        f"Config: Input={INPUT_CSV}, MinEps={MIN_REQUIRED_EPISODES}, RSSTimeout={RSS_TIMEOUT}s, LLM={LLM_MODEL}, MaxLLMW={MAX_LLM_WORKERS}\n"
        f"Counts: InitialCSV={podcast_counts['initial_csv_rows']}, AfterDropna={podcast_counts['after_essential_dropna']}, "
        f"PodWithFetchedEps={podcast_counts['with_any_fetched_episodes']}, TotalRelevantEps={podcast_counts['total_relevant_episodes_for_llm']}, "
        f"PodMeetingMinCrit={podcast_counts['meeting_min_episode_criteria']}, FinalEpsForLLM={podcast_counts['final_episodes_for_llm']}, "
        f"PodWithAggLLM={podcast_counts['with_aggregated_llm_data']}, PodInFinalSummary={podcast_counts['in_final_condition_summary']}\n"
        f"--------------------------------------------------------------------"
    )
    logger.info(summary_log_message)
    end_time = time.time()
    logger.info(f"--- Pipeline End --- Total Duration: {end_time - start_time:.2f}s ---")

def signal_handler(signum, frame):
    logger.warning(f"Signal {signal.Signals(signum).name} received. Attempting graceful shutdown...")
    sys.exit(f"Pipeline interrupted by signal {signum}.")

if __name__ == "__main__":
    # These prints will only execute when the script is run directly, not by spawned processes
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    
    multiprocessing.freeze_support() 
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try: 
        main()
    except KeyboardInterrupt: logger.info("Process interrupted by user (KeyboardInterrupt). Exiting.")
    except SystemExit as se: logger.info(f"SystemExit: {se}") # Catches sys.exit from signal_handler
    except ValueError as ve: logger.critical(f"ValueError in main: {ve}", exc_info=True)
    except Exception as e: logger.critical(f"Unhandled critical exception in main: {e}", exc_info=True)
