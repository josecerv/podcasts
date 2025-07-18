#!/usr/bin/env python3
"""Analyze episode descriptions to understand guest patterns"""

import csv
import feedparser
import requests
from datetime import datetime
import re
import random

# Guest patterns we're using
GUEST_PATTERNS = [
    r"(?:featuring|feature|feat\.?|with guest[s]?|guest[s]?:|interview with|joined by|welcoming|special guest[s]?|our guest[s]?|today'?s guest[s]?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:joins us|is here|stops by|visits|shares|discusses|talks about)",
    r"(?:author|professor|dr\.?|expert|ceo|founder|director|coach|consultant)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}),\s+(?:author of|professor at|ceo of|founder of|director of)",
]

def extract_guests(text):
    """Extract potential guest names"""
    guests = set()
    for pattern in GUEST_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            name = match.strip()
            if len(name.split()) >= 2 and len(name) < 50:
                guests.add(name)
    return list(guests)

# Load podcasts
podcasts = []
with open('podcast_rss_export.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['rss'] and row['rss'].startswith('http'):
            podcasts.append(row)

# Sample 10 podcasts
sample = random.sample(podcasts, 10)

print("ANALYZING EPISODE DESCRIPTIONS FOR GUEST PATTERNS")
print("="*80)

episodes_with_guests = 0
total_episodes = 0

for podcast in sample:
    print(f"\nPodcast: {podcast['podcastID']}")
    
    try:
        resp = requests.get(podcast['rss'], timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        feed = feedparser.parse(resp.content)
        
        # Look at first 5 episodes
        for i, entry in enumerate(feed.entries[:5]):
            total_episodes += 1
            
            title = entry.get('title', '')
            desc = ''
            for field in ['description', 'summary']:
                if hasattr(entry, field) and getattr(entry, field):
                    desc = getattr(entry, field)
                    break
            
            # Strip HTML tags
            desc = re.sub('<[^<]+?>', '', desc)
            
            # Extract guests
            combined = f"{title} {desc}"
            guests = extract_guests(combined)
            
            if guests:
                episodes_with_guests += 1
                print(f"\n  Episode: {title[:80]}...")
                print(f"  GUESTS FOUND: {guests}")
                print(f"  Description preview: {desc[:200]}...")
                
                # Show what matched
                for pattern in GUEST_PATTERNS:
                    matches = re.findall(pattern, combined, re.IGNORECASE)
                    if matches:
                        print(f"  Pattern matched: {pattern[:50]}...")
                        break
            
    except Exception as e:
        print(f"  Error: {e}")

print(f"\n\nSUMMARY:")
print(f"Total episodes analyzed: {total_episodes}")
print(f"Episodes with guests found: {episodes_with_guests}")
print(f"Percentage: {episodes_with_guests/total_episodes*100:.1f}%")