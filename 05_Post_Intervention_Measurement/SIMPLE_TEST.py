#!/usr/bin/env python3
"""Quick test to make sure everything works"""

print("Testing setup...")

# Test 1: Basic imports
try:
    import pandas as pd
    print("✓ pandas installed")
except:
    print("✗ pandas missing - run: pip install -r requirements.txt")
    exit(1)

# Test 2: Check input file
try:
    df = pd.read_csv('podcast_rss_export.csv')
    print(f"✓ Input file found: {len(df)} podcasts")
except:
    print("✗ podcast_rss_export.csv not found")
    exit(1)

# Test 3: Check OpenAI key
import os
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            if 'OPENAI_API_KEY=' in line:
                print("✓ OpenAI key found in ../.env")
                break
else:
    print("✗ No .env file found")
    exit(1)

print("\n✅ All good! Run: python analyze.py")