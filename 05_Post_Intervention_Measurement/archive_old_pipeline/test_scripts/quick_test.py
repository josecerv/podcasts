#!/usr/bin/env python3
"""Quick test to examine raw episode data"""

import csv
import feedparser
import requests
from datetime import datetime, timedelta
import random

# Read a few podcasts
podcasts = []
with open('podcast_rss_export.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['rss'] and row['rss'].startswith('http'):
            podcasts.append(row)

# Sample 5 random podcasts
sample = random.sample(podcasts, 5)

print("EXAMINING RAW EPISODE DATA")
print("="*80)

for podcast in sample:
    print(f"\nPodcast ID: {podcast['podcastID']}")
    print(f"RSS URL: {podcast['rss']}")
    
    response_time = datetime.fromisoformat(podcast['latest_response_time'])
    window_end = response_time + timedelta(days=90)
    
    print(f"Response time: {response_time}")
    print(f"Looking for episodes between {response_time.date()} and {window_end.date()}")
    
    try:
        # Fetch RSS
        resp = requests.get(podcast['rss'], timeout=10, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        feed = feedparser.parse(resp.content)
        
        print(f"Feed has {len(feed.entries)} entries")
        
        # Show first 3 entries regardless of date to see what's there
        for i, entry in enumerate(feed.entries[:3]):
            print(f"\n  Episode {i+1}:")
            print(f"  Title: {entry.get('title', 'NO TITLE')[:100]}")
            
            # Get date
            pub_date = None
            for field in ['published_parsed', 'updated_parsed']:
                if hasattr(entry, field) and getattr(entry, field):
                    pub_date = datetime(*getattr(entry, field)[:6])
                    break
            
            print(f"  Date: {pub_date if pub_date else 'NO DATE'}")
            
            # Get description
            desc = ''
            for field in ['description', 'summary']:
                if hasattr(entry, field):
                    desc = getattr(entry, field)[:200] if getattr(entry, field) else ''
                    if desc:
                        break
            
            print(f"  Description: {desc}...")
            
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print("\n" + "-"*80)