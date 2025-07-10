# Speaker demographic analysis script
# Uses DeepFace and GPT for predicting gender/race from faces and names

import pandas as pd
import numpy as np
import os
import requests
import time
import json
import re
from PIL import Image
from io import BytesIO
import cv2
import tensorflow as tf
from deepface import DeepFace
from openai import OpenAI
import concurrent.futures
import warnings
import unicodedata
from concurrent.futures import ThreadPoolExecutor
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# Get API keys from environment - script will fail if not set
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # Will raise KeyError if not set
SERPAPI_KEY = os.environ["SERPAPI_KEY"]
GOOGLE_SEARCH_API_KEY = os.environ["GOOGLE_SEARCH_API_KEY"] 
GOOGLE_CSE_ID = os.environ["GOOGLE_CSE_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)

# Basic constants
MAX_SPEAKERS_PER_SEMINAR = 128
MAX_IMAGES_TO_TRY = 3
REQUESTS_TIMEOUT = 15
MAX_WORKERS_IMAGES = 8
MAX_WORKERS_NAMES = 4
IMAGE_BATCH_DELAY = 0.5
NAME_BATCH_DELAY = 3
URM_CATEGORIES = ['Black', 'Hispanic/Latino', 'Native American']
DEFAULT_CONFIDENCE = 0.8

os.makedirs('speaker_photos', exist_ok=True)

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

# Helper functions
def normalize_string(s):
    if pd.isna(s): return None
    s = str(s)
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.lower()
    s = s.replace('æ', 'ae').replace('ø', 'o').replace('ł', 'l').replace('ß', 'ss')
    s = re.sub(r'[^\w\s\'-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else None

def preload_deepface_models():
    sample = np.zeros((100, 100, 3), dtype=np.uint8)
    _ = DeepFace.analyze(sample, actions=['gender', 'race'], detector_backend='opencv', enforce_detection=False, silent=True)
    return True

def load_and_parse_data(file_path, max_speakers_per_seminar=128):
    """Load data and extract appearances"""
    df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
    
    # Basic columns we need
    seminar_cols = ['seminar_id', 'university', 'discipline', 'seminar_name',
                    'condition', 'contact_type', 'Link', 'bin_category']
    
    original_seminar_df = df[seminar_cols].copy()
    original_seminar_df.dropna(subset=['seminar_id'], inplace=True)
    original_seminar_df['department_key'] = original_seminar_df['university'].astype(str) + '-' + original_seminar_df['discipline'].astype(str)
    original_seminar_df = original_seminar_df.drop_duplicates(subset=['seminar_id']).reset_index(drop=True)

    # Extract speaker info
    speakers_list = []
    for index, row in df.iterrows():
        seminar_info = {col: row.get(col) for col in seminar_cols}
        if pd.isna(seminar_info.get('seminar_id')): continue

        for i in range(1, max_speakers_per_seminar + 1):
            fn_col, ln_col = f'First Name_{i}', f'Last Name_{i}'
            rank_col, uni_col, date_col = f'rank_{i}', f'university_{i}', f'date_{i}'
            if fn_col not in df.columns: break
            first_name, last_name = row.get(fn_col), row.get(ln_col)
            has_first = pd.notna(first_name) and str(first_name).strip() != ''
            has_last = pd.notna(last_name) and str(last_name).strip() != ''
            if not (has_first or has_last): continue

            speaker_details = {
                'speaker_num_in_seminar': i,
                'name': f"{first_name or ''} {last_name or ''}".strip(),
                'first_name': str(first_name).strip() if pd.notna(first_name) else '',
                'last_name': str(last_name).strip() if pd.notna(last_name) else '',
                'rank': row.get(rank_col, '') if rank_col in df.columns else '',
                'affiliation_raw': row.get(uni_col, '') if uni_col in df.columns else '',
                'date': row.get(date_col, '') if date_col in df.columns else ''
            }
            speakers_list.append({**seminar_info, **speaker_details})

    all_speaker_appearances_df = pd.DataFrame(speakers_list)
    
    # Process appearance data
    all_speaker_appearances_df['affiliation'] = all_speaker_appearances_df['affiliation_raw'].fillna(all_speaker_appearances_df['university'])
    all_speaker_appearances_df['name_norm'] = all_speaker_appearances_df['name'].apply(normalize_string)
    all_speaker_appearances_df['affiliation_norm'] = all_speaker_appearances_df['affiliation'].apply(normalize_string)
    all_speaker_appearances_df = all_speaker_appearances_df.dropna(subset=['name_norm'])

    # Get unique people
    unique_people_df = all_speaker_appearances_df.groupby(['name_norm', 'affiliation_norm'], as_index=False).agg(
        name=('name', 'first'), first_name=('first_name', 'first'),
        last_name=('last_name', 'first'), affiliation=('affiliation', 'first'))
    unique_people_df = unique_people_df.dropna(subset=['name_norm'])
    unique_people_df['person_id'] = range(1, len(unique_people_df) + 1)

    # Join person_id back to appearances
    all_speaker_appearances_df = pd.merge(all_speaker_appearances_df, unique_people_df[['name_norm', 'affiliation_norm', 'person_id']], 
                                          on=['name_norm', 'affiliation_norm'], how='left')
    all_speaker_appearances_df['appearance_id'] = range(1, len(all_speaker_appearances_df) + 1)

    # Select columns
    appearance_cols = ['appearance_id', 'person_id', 'seminar_id', 'university', 'discipline',
                       'seminar_name', 'condition', 'contact_type', 'Link', 'bin_category',
                       'speaker_num_in_seminar', 'name', 'first_name', 'last_name',
                       'rank', 'affiliation', 'date', 'name_norm', 'affiliation_norm']
    appearance_cols = [col for col in appearance_cols if col in all_speaker_appearances_df.columns]
    all_speaker_appearances_df = all_speaker_appearances_df[appearance_cols]

    people_cols = ['person_id', 'name', 'first_name', 'last_name', 'affiliation', 'name_norm', 'affiliation_norm']
    people_cols = [col for col in people_cols if col in unique_people_df.columns]
    unique_people_df = unique_people_df[people_cols]

    return all_speaker_appearances_df, unique_people_df, original_seminar_df

# Image search and processing functions
def get_image_urls(person_info, num_results=MAX_IMAGES_TO_TRY):
    # Try SerpAPI first
    name = person_info['name']
    affiliation = person_info.get('affiliation', '')
    query = f"{name} {affiliation} professor academic photo"
    
    # SerpAPI request
    params = {"api_key": SERPAPI_KEY, "engine": "google_images", "q": query, "tbm": "isch", "num": num_results}
    response = requests.get("https://serpapi.com/search", params=params, timeout=REQUESTS_TIMEOUT)
    results = response.json()
    urls = []
    if "images_results" in results:
        urls = [img.get("original") for img in results["images_results"] if img.get("original")]
    
    # If SerpAPI fails, try Google CSE
    if not urls:
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {"key": GOOGLE_SEARCH_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "searchType": "image", "num": min(num_results, 10)}
        response = requests.get(url, params=params, timeout=REQUESTS_TIMEOUT)
        results = response.json()
        if "items" in results:
            urls = [item.get("link") for item in results["items"] if item.get("link")]
            
    return urls

def download_image(url, person_id):
    if not url: return None
    save_path = f"speaker_photos/{person_id}.jpg"
    
    response = requests.get(url, timeout=REQUESTS_TIMEOUT, headers={'User-Agent': 'Mozilla/5.0'})
    image = Image.open(BytesIO(response.content))
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    
    max_size = (800, 800)
    image.thumbnail(max_size, Image.LANCZOS)
    image.save(save_path, format='JPEG', quality=85)
    return save_path

def analyze_face(image_path):
    """Analyze a face using DeepFace"""
    analysis = DeepFace.analyze(img_path=image_path, actions=['gender', 'race'], 
                               detector_backend='opencv', enforce_detection=False, align=True, silent=True)
    
    result = analysis[0] if isinstance(analysis, list) else analysis
    
    # Get gender info
    gender_dict = result.get('gender', {})
    gender = max(gender_dict, key=gender_dict.get) if gender_dict else 'unknown'
    gender_confidence = gender_dict.get(gender, 0.0) / 100.0
    gender = {'Man': 'Male', 'Woman': 'Female'}.get(gender, gender)
    
    # Get race info
    race_dict = result.get('race', {})
    race = result.get('dominant_race', 'unknown')
    race_confidence = race_dict.get(race, 0.0) / 100.0
    race_map = {'asian': 'Asian', 'black': 'Black', 'indian': 'South Asian', 
               'latino hispanic': 'Hispanic/Latino', 'middle eastern': 'Middle Eastern', 'white': 'White'}
    race = race_map.get(race.lower(), race if race != 'unknown' else 'unknown')
    
    return {
        "gender": gender,
        "gender_confidence": gender_confidence,
        "race": race,
        "race_confidence": race_confidence,
        "analysis_success": True
    }

def analyze_name_with_gpt4o(person_info):
    """Analyze name using GPT-4o"""
    name = person_info['name']
    first_name = person_info.get('first_name', '')
    last_name = person_info.get('last_name', '')
    
    prompt = f"""Analyze the name: Full Name: {name}, First Name: {first_name}, Last Name: {last_name}. 
    Predict gender ("Male", "Female") and race/ethnicity ("White", "Black", "Asian", "Hispanic/Latino", 
    "Native American", "Middle Eastern", "South Asian"). Provide confidence (0.0-1.0). 
    Return JSON: {{"gender": "classification", "gender_confidence": 0.x, "race": "classification", "race_confidence": 0.x}}"""
    
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "You predict gender and race/ethnicity from names."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    result_str = response.choices[0].message.content
    result = json.loads(result_str)
    result["analysis_success"] = True
    return result

def analyze_names_batch(people_batch):
    results = []
    for person_info in people_batch:
        analysis = analyze_name_with_gpt4o(person_info)
        results.append({
            "person_id": person_info["person_id"],
            "name_gender": analysis.get("gender", "unknown"),
            "name_gender_confidence": analysis.get("gender_confidence", 0.0),
            "name_race": analysis.get("race", "unknown"),
            "name_race_confidence": analysis.get("race_confidence", 0.0),
            "name_analysis_success": analysis.get("analysis_success", False)
        })
        time.sleep(0.5)
    return results

# Process a single person's image
def process_single_person_image(person_info):
    person_id = person_info['person_id']
    image_urls = get_image_urls(person_info)
    
    for url in image_urls[:MAX_IMAGES_TO_TRY]:
        try:
            image_path = download_image(url, person_id)
            if image_path:
                analysis_result = analyze_face(image_path)
                if analysis_result["analysis_success"]:
                    return {
                        "person_id": person_id,
                        "name": person_info['name'],
                        "affiliation": person_info.get('affiliation', ''),
                        "image_found": True,
                        "image_path": image_path,
                        "image_url_tried": url,
                        "face_gender": analysis_result["gender"],
                        "face_gender_confidence": analysis_result["gender_confidence"],
                        "face_race": analysis_result["race"],
                        "face_race_confidence": analysis_result["race_confidence"],
                        "face_analysis_success": True
                    }
        except Exception:
            pass  # Try next image
    
    # If we get here, no valid faces found
    return {
        "person_id": person_id,
        "name": person_info['name'],
        "affiliation": person_info.get('affiliation', ''),
        "image_found": False,
        "image_path": None,
        "image_url_tried": image_urls[0] if image_urls else None,
        "face_gender": "unknown",
        "face_gender_confidence": 0.0,
        "face_race": "unknown",
        "face_race_confidence": 0.0,
        "face_analysis_success": False
    }

# Process all people in parallel
def process_people_images_parallel(unique_people_df):
    output_filename = 'people_face_analysis.csv'
    
    # Check for existing results
    if os.path.exists(output_filename):
        existing_results = pd.read_csv(output_filename)
        if 'person_id' in existing_results.columns:
            existing_person_ids = set(existing_results['person_id'].dropna().astype(int).tolist())
            expected_person_ids = set(unique_people_df['person_id'].astype(int))
            people_to_process_ids = expected_person_ids - existing_person_ids
            
            # If everyone already processed, return existing results
            if not people_to_process_ids:
                return existing_results
            
            # Filter to only process new people
            people_to_process_df = unique_people_df[unique_people_df['person_id'].isin(people_to_process_ids)]
            person_info_list = people_to_process_df.to_dict('records')
        else:
            person_info_list = unique_people_df.to_dict('records')
    else:
        person_info_list = unique_people_df.to_dict('records')
    
    # Process in parallel
    results_list = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_IMAGES) as executor:
        future_to_person = {executor.submit(process_single_person_image, person_info): person_info 
                           for person_info in person_info_list}
        
        for future in concurrent.futures.as_completed(future_to_person):
            try:
                results_list.append(future.result())
            except Exception:
                person_info = future_to_person[future]
                results_list.append({
                    "person_id": person_info["person_id"],
                    "name": person_info["name"],
                    "image_found": False,
                    "image_path": None,
                    "image_url_tried": None,
                    "face_gender": "error",
                    "face_gender_confidence": 0.0,
                    "face_race": "error",
                    "face_race_confidence": 0.0,
                    "face_analysis_success": False
                })
    
    # Combine with existing results if they exist
    new_results_df = pd.DataFrame(results_list)
    if os.path.exists(output_filename):
        existing_results = pd.read_csv(output_filename)
        if not new_results_df.empty:
            results_df = pd.concat([existing_results, new_results_df], ignore_index=True)
        else:
            results_df = existing_results
    else:
        results_df = new_results_df
    
    # Save results
    if not results_df.empty:
        results_df['person_id'] = results_df['person_id'].astype(int)
        results_df.sort_values(by='person_id', inplace=True)
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    return results_df

def process_people_name_analysis_parallel(unique_people_df):
    output_filename = 'people_name_analysis.csv'
    
    # Check for existing results
    if os.path.exists(output_filename):
        existing_results = pd.read_csv(output_filename)
        if 'person_id' in existing_results.columns:
            existing_ids = set(existing_results['person_id'].dropna().astype(int))
            # Only process new people
            to_process = [row.to_dict() for _, row in unique_people_df.iterrows() 
                         if row["person_id"] not in existing_ids]
            
            # If everyone already processed, return existing results
            if not to_process:
                return existing_results
        else:
            to_process = unique_people_df.to_dict('records')
    else:
        to_process = unique_people_df.to_dict('records')
    
    # Process in batches
    all_results = []
    batch_size = 20
    batches = [to_process[i:i + batch_size] for i in range(0, len(to_process), batch_size)]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_NAMES) as executor:
        future_to_batch = {executor.submit(analyze_names_batch, batch): i 
                          for i, batch in enumerate(batches)}
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                all_results.extend(future.result())
            except Exception:
                pass  # Continue with other batches
    
    # Combine with existing results if they exist
    new_results_df = pd.DataFrame(all_results)
    if os.path.exists(output_filename):
        existing_results = pd.read_csv(output_filename)
        if not new_results_df.empty:
            existing_results['person_id'] = existing_results['person_id'].astype(int)
            new_results_df['person_id'] = new_results_df['person_id'].astype(int)
            results_df = pd.concat([existing_results, new_results_df], ignore_index=True)
            results_df = results_df.drop_duplicates(subset=['person_id'], keep='last')
        else:
            results_df = existing_results
    else:
        results_df = new_results_df
    
    # Save results
    if not results_df.empty:
        results_df['person_id'] = results_df['person_id'].astype(int)
        results_df.sort_values(by='person_id', inplace=True)
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    return results_df

def combine_person_analyses(face_analysis_df, name_analysis_df):
    """Combine face and name analysis results"""
    # Handle empty dataframes
    if face_analysis_df.empty and name_analysis_df.empty:
        return pd.DataFrame()
    
    combined_df = pd.DataFrame()
    
    # Face analysis is empty, use only name analysis
    if face_analysis_df.empty:
        combined_df = name_analysis_df.copy()
        combined_df['face_gender'] = 'unknown'
        combined_df['face_gender_confidence'] = 0.0
        combined_df['face_race'] = 'unknown'
        combined_df['face_race_confidence'] = 0.0
        combined_df['face_analysis_success'] = False
        combined_df['image_found'] = False
        combined_df['image_path'] = ''
        combined_df['image_url_tried'] = ''
    
    # Name analysis is empty, use only face analysis
    elif name_analysis_df.empty:
        combined_df = face_analysis_df.copy()
        combined_df['name_gender'] = 'unknown'
        combined_df['name_gender_confidence'] = 0.0
        combined_df['name_race'] = 'unknown'
        combined_df['name_race_confidence'] = 0.0
        combined_df['name_analysis_success'] = False
    
    # Both analyses have data, merge them
    else:
        face_analysis_df['person_id'] = face_analysis_df['person_id'].astype(int)
        name_analysis_df['person_id'] = name_analysis_df['person_id'].astype(int)
        
        name_cols_to_merge = ['person_id', 'name_gender', 'name_gender_confidence', 
                             'name_race', 'name_race_confidence', 'name_analysis_success']
        name_cols_to_merge = [col for col in name_cols_to_merge if col in name_analysis_df.columns]
        
        combined_df = pd.merge(face_analysis_df, name_analysis_df[name_cols_to_merge], on="person_id", how="outer")
        
        # Fill NaNs
        for col in ['face_gender', 'name_gender', 'face_race', 'name_race']:
            combined_df[col].fillna('unknown', inplace=True)
        
        for col in ['face_gender_confidence', 'name_gender_confidence', 'face_race_confidence', 'name_race_confidence']:
            combined_df[col].fillna(0.0, inplace=True)
        
        for col in ['face_analysis_success', 'name_analysis_success']:
            combined_df[col].fillna(False, inplace=True)
        
        if 'image_found' in combined_df.columns:
            combined_df['image_found'].fillna(value=False, inplace=True)
        
        if 'image_path' in combined_df.columns:
            combined_df['image_path'].fillna(value='', inplace=True)
        
        if 'image_url_tried' in combined_df.columns:
            combined_df['image_url_tried'].fillna(value='', inplace=True)
    
    # Final prediction logic
    def determine_agreement(row, attribute):
        face_val = row.get(f'face_{attribute}', 'unknown')
        name_val = row.get(f'name_{attribute}', 'unknown')
        face_success = row.get('face_analysis_success', False)
        name_success = row.get('name_analysis_success', False)
        
        if face_success and name_success and face_val != 'unknown' and name_val != 'unknown':
            return 'Agree' if face_val == name_val else 'Disagree'
        else:
            return 'Undetermined'
    
    def calculate_final(row, attribute):
        face_val = row.get(f'face_{attribute}', 'unknown')
        face_conf = row.get(f'face_{attribute}_confidence', 0.0)
        name_val = row.get(f'name_{attribute}', 'unknown')
        name_conf = row.get(f'name_{attribute}_confidence', 0.0)
        face_success = row.get('face_analysis_success', False)
        name_success = row.get('name_analysis_success', False)
        
        if face_success and face_val != 'unknown' and (not name_success or name_val == 'unknown'):
            return face_val, face_conf
        elif name_success and name_val != 'unknown' and (not face_success or face_val == 'unknown'):
            return name_val, name_conf
        elif face_success and name_success and face_val != 'unknown' and name_val != 'unknown':
            if face_val == name_val:
                return face_val, max(face_conf, name_conf)
            else:
                return (face_val, face_conf) if face_conf >= name_conf else (name_val, name_conf)
        else:
            return 'unknown', 0.0
    
    combined_df['gender_agreement'] = combined_df.apply(lambda r: determine_agreement(r, 'gender'), axis=1)
    combined_df['race_agreement'] = combined_df.apply(lambda r: determine_agreement(r, 'race'), axis=1)
    combined_df[['final_gender', 'final_gender_confidence']] = combined_df.apply(
        lambda r: pd.Series(calculate_final(r, 'gender')), axis=1)
    combined_df[['final_race', 'final_race_confidence']] = combined_df.apply(
        lambda r: pd.Series(calculate_final(r, 'race')), axis=1)
    
    # Add derived columns
    combined_df['is_urm'] = combined_df['final_race'].isin(URM_CATEGORIES)
    combined_df['is_female'] = combined_df['final_gender'] == 'Female'
    combined_df['high_confidence_gender'] = combined_df['final_gender_confidence'] >= DEFAULT_CONFIDENCE
    combined_df['high_confidence_race'] = combined_df['final_race_confidence'] >= DEFAULT_CONFIDENCE
    combined_df['high_confidence_urm'] = combined_df['high_confidence_race']
    
    combined_df.to_csv('people_combined_analysis.csv', index=False, encoding='utf-8-sig')
    return combined_df

def generate_seminar_summary_export(speaker_appearances_with_analysis_df, original_seminar_df, 
                                  output_filename='seminar_demographic_summary.csv'):
    """Aggregate demographics to seminar level"""
    if speaker_appearances_with_analysis_df.empty or original_seminar_df.empty:
        return pd.DataFrame()
    
    # Create department_key if missing
    if 'department_key' not in original_seminar_df.columns:
        original_seminar_df['department_key'] = original_seminar_df['university'].astype(str) + '-' + original_seminar_df['discipline'].astype(str)
    
    # Fill missing values
    speaker_appearances_with_analysis_df['is_urm'] = speaker_appearances_with_analysis_df['is_urm'].fillna(False)
    
    # Group by seminar_id and aggregate
    seminar_groups = speaker_appearances_with_analysis_df.groupby('seminar_id')
    seminar_summary = seminar_groups.agg(
        total_speakers=('appearance_id', 'count'),
        urm_speaker_count=('is_urm', 'sum'),
        female_appearances_high_conf=('is_female', lambda x: x[speaker_appearances_with_analysis_df.loc[x.index, 'high_confidence_gender']].sum()),
        gender_preds_high_conf=('high_confidence_gender', 'sum'),
        urm_appearances_high_conf=('is_urm', lambda x: x[speaker_appearances_with_analysis_df.loc[x.index, 'high_confidence_urm']].sum()),
        urm_preds_high_conf=('high_confidence_urm', 'sum')
    ).reset_index()
    
    # Calculate derived metrics
    seminar_summary['non_urm_speaker_count'] = seminar_summary['total_speakers'] - seminar_summary['urm_speaker_count']
    seminar_summary['pct_female'] = seminar_summary.apply(
        lambda r: (r['female_appearances_high_conf'] / r['gender_preds_high_conf'] * 100) 
        if r['gender_preds_high_conf'] > 0 else 0, axis=1).round(2)
    seminar_summary['pct_urm'] = seminar_summary.apply(
        lambda r: (r['urm_appearances_high_conf'] / r['urm_preds_high_conf'] * 100) 
        if r['urm_preds_high_conf'] > 0 else 0, axis=1).round(2)
    
    # Merge with original seminar data
    orig_cols_to_keep = ['seminar_id', 'university', 'discipline', 'seminar_name',
                        'condition', 'contact_type', 'Link', 'bin_category', 'department_key']
    orig_cols_to_keep = [c for c in orig_cols_to_keep if c in original_seminar_df.columns]
    
    final_seminar_df = pd.merge(
        original_seminar_df[orig_cols_to_keep].drop_duplicates(subset=['seminar_id']),
        seminar_summary, on='seminar_id', how='left')
    
    # Fill NAs
    cols_to_fill = ['total_speakers', 'pct_female', 'pct_urm', 'female_appearances_high_conf',
                   'gender_preds_high_conf', 'urm_appearances_high_conf', 'urm_preds_high_conf',
                   'urm_speaker_count', 'non_urm_speaker_count']
    
    for col in cols_to_fill:
        if col in final_seminar_df.columns:
            final_seminar_df[col].fillna(0, inplace=True)
            if 'count' in col or 'speakers' in col or 'preds' in col:
                final_seminar_df[col] = final_seminar_df[col].astype(int)
    
    final_seminar_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    return final_seminar_df

def perform_clustered_regression(aggregated_seminar_df):
    """Run regression for URM percentage vs treatment with clustered errors"""
    if aggregated_seminar_df.empty:
        return None
    
    # Filter to seminars with speakers
    df_reg = aggregated_seminar_df[aggregated_seminar_df['total_speakers'] > 0].copy()
    if len(df_reg) < 2:
        return None
    
    # Prepare dummy variables
    df_reg['bin_category'] = df_reg['bin_category'].astype(str).fillna('Unknown')
    valid_bins = ['[0,1]', '(1,3]', '(3,5]', '(5,7]', '(7,11]', '(11,17]', '(17,26]', 'Unknown']
    df_reg['bin_category'] = pd.Categorical(df_reg['bin_category'], categories=valid_bins, ordered=False)
    bin_dummies = pd.get_dummies(df_reg['bin_category'], prefix='bin', drop_first=True)
    
    df_reg['discipline'] = df_reg['discipline'].astype(str).fillna('Unknown')
    discipline_dummies = pd.get_dummies(df_reg['discipline'], prefix='disc', drop_first=True)
    
    df_reg = pd.concat([df_reg, bin_dummies, discipline_dummies], axis=1)
    df_reg.columns = df_reg.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)  # Clean names
    
    # Construct formula
    pct_urm_col = 'pcturm' if 'pcturm' in df_reg.columns else 'pct_urm'
    condition_col = 'condition'
    formula = f"{pct_urm_col} ~ C({condition_col}, Treatment('control'))"
    
    bin_terms = [col for col in df_reg.columns if col.startswith('bin_')]
    if bin_terms:
        formula += " + " + " + ".join(bin_terms)
    
    disc_terms = [col for col in df_reg.columns if col.startswith('disc_')]
    if disc_terms:
        formula += " + " + " + ".join(disc_terms)
    
    # Run regression with clustered errors
    cluster_key = 'departmentkey'
    cluster_groups = df_reg[cluster_key]
    
    ols_model = smf.ols(formula, data=df_reg).fit()
    clustered_results = ols_model.get_robustcov_results(cov_type='cluster', groups=cluster_groups)
    
    return clustered_results

# Main execution
def main():
    # Configuration
    INPUT_FILE = 'master-data-fall.csv'
    
    start_time = time.time()
    
    # Load and parse data
    all_speaker_appearances_df, unique_people_df, original_seminar_df = load_and_parse_data(INPUT_FILE)
    
    # Preload DeepFace models
    preload_deepface_models()
    
    # Process faces in parallel
    face_analysis_df = process_people_images_parallel(unique_people_df)
    
    # Process names in parallel
    name_analysis_df = process_people_name_analysis_parallel(unique_people_df)
    
    # Combine face and name analyses
    person_analysis_df = combine_person_analyses(face_analysis_df, name_analysis_df)
    
    # Map analysis results back to appearances
    if not person_analysis_df.empty:
        analysis_cols = list(person_analysis_df.columns)
        analysis_cols = [col for col in analysis_cols if col == 'person_id' or col not in all_speaker_appearances_df.columns]
        
        speaker_appearances_with_analysis_df = pd.merge(
            all_speaker_appearances_df, person_analysis_df[analysis_cols], 
            on='person_id', how='left')
        
        # Fill missing values
        speaker_appearances_with_analysis_df['final_gender'].fillna('unknown', inplace=True)
        speaker_appearances_with_analysis_df['final_race'].fillna('unknown', inplace=True)
        speaker_appearances_with_analysis_df['is_urm'].fillna(False, inplace=True)
        speaker_appearances_with_analysis_df['is_female'].fillna(False, inplace=True)
        
        speaker_appearances_with_analysis_df.to_csv('speaker_appearances_analysis.csv', index=False, encoding='utf-8-sig')
    else:
        speaker_appearances_with_analysis_df = all_speaker_appearances_df
    
    # Generate seminar summary
    if not speaker_appearances_with_analysis_df.empty and not original_seminar_df.empty:
        aggregated_seminar_df = generate_seminar_summary_export(
            speaker_appearances_with_analysis_df, original_seminar_df)
        
        # Run regression
        if not aggregated_seminar_df.empty:
            regression_results = perform_clustered_regression(aggregated_seminar_df)
            if regression_results:
                print("\nRegression Results:")
                print(regression_results.summary())
    
    end_time = time.time()
    print(f"\nAnalysis complete! Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
