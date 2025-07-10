#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import pandas as pd
import time
from datetime import datetime, timezone, timedelta

# Bitly API credentials
API_KEY = 'cf0f12b521bd873f4ac558e366cc946b456ef90d'

# Headers for Bitly API requests
bitly_headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

# Read the CSV file
csv_file = 'podcasts_mail_merge_v2.csv'
df = pd.read_csv(csv_file)

# Define the date range
start_date = datetime(2024, 12, 17, tzinfo=timezone.utc)
end_date = datetime(2024, 12, 19, tzinfo=timezone.utc)

# Generate list of dates
date_range = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') 
              for x in range((end_date - start_date).days + 1)]

def get_bitlink_id(bitlink):
    if pd.isna(bitlink) or bitlink == 'N/A':
        return None
    return bitlink.replace('https://', '').replace('http://', '')

def get_clicks_for_bitlink(bitlink_id, unit='day', units=-1, size=500):
    if bitlink_id is None:
        return []
    url = f'https://api-ssl.bitly.com/v4/bitlinks/{bitlink_id}/clicks'
    params = {
        'unit': unit,
        'units': units,
        'size': size
    }
    response = requests.get(url, headers=bitly_headers, params=params)
    if response.status_code == 200:
        return response.json().get('link_clicks', [])
    else:
        print(f"Error fetching clicks for {bitlink_id}: {response.status_code} {response.text}")
        return []

# Initialize results dictionary
results = []

print("Fetching click data...")
for index, row in df.iterrows():
    podcast_id = row['Podcast ID']
    podcast_title = row['Podcast Title']
    bitlink = row['Branded Bitly Link']
    
    # Initialize result dictionary for this podcast
    result = {
        'Podcast ID': podcast_id,
        'Podcast Title': podcast_title,
        'Bitly Link': bitlink,
        'Total Clicks': 0
    }
    # Initialize clicks for each date as 0
    for date in date_range:
        result[date] = 0

    # Get click data
    bitlink_id = get_bitlink_id(bitlink)
    if bitlink_id:
        link_clicks = get_clicks_for_bitlink(bitlink_id, unit='day')
        
        # Process clicks
        for click in link_clicks:
            date_str = click['date']
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            date_key = dt.strftime('%Y-%m-%d')
            
            # Only count clicks within our date range
            if date_key in date_range:
                clicks = click['clicks']
                result[date_key] = clicks
                result['Total Clicks'] += clicks

    results.append(result)
    time.sleep(0.2)  # Rate limiting

# Create DataFrame and reorder columns
df_results = pd.DataFrame(results)
column_order = ['Podcast ID', 'Podcast Title', 'Bitly Link', 'Total Clicks'] + date_range
df_results = df_results[column_order]

# Save to CSV
df_results.to_csv('bitly_click_summary.csv', index=False)
print("\nClick summary has been saved to 'bitly_click_summary.csv'")


# In[ ]:




