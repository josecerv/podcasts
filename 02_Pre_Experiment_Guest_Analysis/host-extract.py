#!/usr/bin/env python3
"""
Podcast Host Demographics Extractor
Extracts host information from podcast metadata and analyzes demographics (gender and race/ethnicity).
"""

import pandas as pd
import logging
import json
import re
import time
import os
from bs4 import BeautifulSoup
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('podcast_host_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
LLM_MODEL = "o3-mini"  # Using o3-mini directly as in your second script

def clean_text(text):
    """Clean text to be UTF-8 compliant and remove problematic characters."""
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.encode('utf-8', 'replace').decode('utf-8', 'replace')
        text = re.sub(r'[\x80-\x9F]', '', text)
        text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = re.sub(r'&#?\w+;', ' ', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ''

def extract_text_from_html(html_content):
    """Extract plain text from HTML content."""
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}")
        return ""

def read_podcast_ids(file_path):
    """Read podcast IDs from the survey file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"File columns: {df.columns.tolist()}")
        
        # Extract IDs from the 'x' column
        if 'x' in df.columns:
            podcast_ids = df['x'].tolist()
            logger.info(f"Extracted {len(podcast_ids)} podcast IDs from 'x' column")
            return podcast_ids
        else:
            # If no 'x' column, try the second column
            logger.warning(f"No 'x' column found. Available columns: {df.columns.tolist()}")
            if len(df.columns) >= 2:
                podcast_ids = df.iloc[:, 1].tolist()
                logger.info(f"Extracted {len(podcast_ids)} podcast IDs from second column")
                return podcast_ids
            else:
                # Last resort: use the first column
                podcast_ids = df.iloc[:, 0].tolist()
                logger.info(f"Extracted {len(podcast_ids)} podcast IDs from first column")
                return podcast_ids
    except Exception as e:
        logger.error(f"Error reading podcast IDs: {str(e)}")
        return []

def get_podcast_metadata(podcast_ids, csv_file):
    """Get publisher and description for each podcast ID from the CSV file."""
    df = pd.read_csv(csv_file, low_memory=False)
    
    # Create dictionary to store results
    podcast_metadata = {}
    
    for podcast_id in podcast_ids:
        # Find matching row
        matching_row = df[df['podcast_id'].astype(str) == str(podcast_id)]
        
        if not matching_row.empty:
            publisher = matching_row['publisher'].iloc[0] if pd.notna(matching_row['publisher'].iloc[0]) else ""
            description = matching_row['description'].iloc[0] if pd.notna(matching_row['description'].iloc[0]) else ""
            
            # Clean up description if it contains HTML
            if description and ('<' in description or '&' in description):
                description = extract_text_from_html(description)
            description = clean_text(description)
            
            podcast_metadata[podcast_id] = {
                'publisher': clean_text(publisher),
                'description': description
            }
        else:
            logger.warning(f"No matching metadata found for podcast ID: {podcast_id}")
    
    return podcast_metadata

def extract_host_names(podcast_data):
    """Extract host names from publisher and description."""
    publisher = podcast_data.get('publisher', '')
    description = podcast_data.get('description', '')
    
    # First try to get name from publisher
    if publisher:
        prompt = f"""
        Analyze if this podcast publisher text contains a person's name that might be the host: "{publisher}"
        Only return actual people's names, not company or podcast names.
        For example, "John Smith" would be identified as a host name, but "ABC Media" or "Nature Podcast" would not.
        If you find names, return them as a list. If no clear host names are found, return "NA".
        """
        
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="high"
            )
            
            result = response.choices[0].message.content.strip()
            
            # Check if result contains names or NA
            if result.lower() != "na" and "na" not in result.lower():
                # If the result looks like a name list, return it
                if '[' in result and ']' in result:
                    try:
                        # Try to parse as JSON list
                        names = json.loads(result)
                        if isinstance(names, list) and names:
                            return names, "publisher"
                    except:
                        # If JSON parsing fails, try regex
                        name_matches = re.findall(r'"([^"]+)"', result)
                        if name_matches:
                            return name_matches, "publisher"
                else:
                    # Handle plain text response
                    lines = [line.strip() for line in result.split('\n') if line.strip()]
                    potential_names = []
                    for line in lines:
                        # Remove common prefixes
                        clean_line = re.sub(r'^[-•*]|\[|\]|"', '', line).strip()
                        if clean_line and clean_line.lower() != "na":
                            potential_names.append(clean_line)
                    if potential_names:
                        return potential_names, "publisher"
        
        except Exception as e:
            logger.error(f"Error extracting names from publisher with LLM: {str(e)}")
    
    # If no names found in publisher or error occurred, try description
    if description:
        prompt = f"""
        Extract podcast host names from this podcast description: "{description[:800]}"
        Only return names if they are explicitly mentioned as hosts, presenters, or the podcast creator.
        Return the names as a list. If no clear host names can be identified, return "NA".
        """
        
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="high"
            )
            
            result = response.choices[0].message.content.strip()
            
            # Check if result contains names or NA
            if result.lower() != "na" and "na" not in result.lower():
                # If the result looks like a name list, process it
                if '[' in result and ']' in result:
                    try:
                        # Try to parse as JSON list
                        names = json.loads(result)
                        if isinstance(names, list) and names:
                            return names, "description"
                    except:
                        # If JSON parsing fails, try regex
                        name_matches = re.findall(r'"([^"]+)"', result)
                        if name_matches:
                            return name_matches, "description"
                else:
                    # Handle plain text response
                    lines = [line.strip() for line in result.split('\n') if line.strip()]
                    potential_names = []
                    for line in lines:
                        # Remove common prefixes
                        clean_line = re.sub(r'^[-•*]|\[|\]|"', '', line).strip()
                        if clean_line and clean_line.lower() != "na":
                            potential_names.append(clean_line)
                    if potential_names:
                        return potential_names, "description"
        
        except Exception as e:
            logger.error(f"Error extracting names from description with LLM: {str(e)}")
    
    # If no names found or errors occurred in both attempts
    return [], "none"

def analyze_host_demographics(names):
    """Analyze if hosts are female and/or URM (Black or Latino)."""
    female_hosts = 0
    urm_hosts = 0
    
    for name in names:
        prompt = f"""
        Analyze the following podcast host name: "{name}"
        
        Determine:
        1. If the person is likely female (based on name conventions)
        2. If the person is likely Black or Latino (based on name conventions)
        
        First, answer whether the person is likely female (Yes/No/Uncertain).
        Then, answer whether the person is likely Black or Latino (Yes/No/Uncertain).
        Provide brief reasoning for each determination.
        """
        
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="high"
            )
            
            result = response.choices[0].message.content.strip()
            
            # Check for female identification
            if re.search(r'female:?\s*yes', result.lower()) or "likely female" in result.lower():
                female_hosts += 1
            
            # Check for URM identification
            if re.search(r'black or latino:?\s*yes', result.lower()) or "likely black" in result.lower() or "likely latino" in result.lower():
                urm_hosts += 1
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error analyzing demographics for {name}: {str(e)}")
    
    return female_hosts, urm_hosts

def process_single_podcast(podcast_id, metadata):
    """Process a single podcast's metadata and extract host demographics."""
    try:
        logger.info(f"Processing podcast ID: {podcast_id}")
        
        # Extract host names from metadata
        names, source = extract_host_names(metadata)
        
        if not names:
            logger.warning(f"No host names found for podcast ID: {podcast_id}")
            return {
                "podcastID": podcast_id,
                "total_host": "NA",
                "female_host": 0,
                "urm_host": 0,
                "publisher": metadata.get('publisher', '')
            }
        
        # Analyze demographics
        female_hosts, urm_hosts = analyze_host_demographics(names)
        
        # Return result
        return {
            "podcastID": podcast_id,
            "total_host": len(names),
            "host_names": names,
            "female_host": female_hosts,
            "urm_host": urm_hosts,
            "publisher": metadata.get('publisher', '')
        }
    
    except Exception as e:
        logger.error(f"Error processing podcast {podcast_id}: {str(e)}")
        return {
            "podcastID": podcast_id,
            "total_host": "NA",
            "female_host": 0,
            "urm_host": 0,
            "publisher": metadata.get('publisher', '')
        }

def main():
    """Main function to run the podcast host extraction."""
    start_time = time.time()
    
    # File paths
    podcast_ids_file = "survey1-podcastID.csv"
    podcast_csv_file = "podcasts_final_sample.csv"
    output_file = "podcast_demographics.jsonl"
    
    # Read podcast IDs
    podcast_ids = read_podcast_ids(podcast_ids_file)
    logger.info(f"Read {len(podcast_ids)} podcast IDs")
    
    # Get metadata for each podcast
    podcast_metadata = get_podcast_metadata(podcast_ids, podcast_csv_file)
    logger.info(f"Retrieved metadata for {len(podcast_metadata)} podcasts")
    
    # Process podcasts sequentially
    with open(output_file, 'w') as f:
        for podcast_id, metadata in podcast_metadata.items():
            result = process_single_podcast(podcast_id, metadata)
            f.write(json.dumps(result) + '\n')
            
            # Add a small delay between podcasts
            time.sleep(1)
    
    processing_time = time.time() - start_time
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    logger.info(f"Results written to {output_file}")

if __name__ == "__main__":
    main()
