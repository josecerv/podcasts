#!/usr/bin/env python3
"""Validate improvements in guest extraction"""

import sys
sys.path.append('.')
from guest_analysis import extract_guest_patterns as extract_old
from guest_analysis_optimized import extract_guest_patterns as extract_new, clean_text

# Test cases from real episodes
test_cases = [
    {
        "title": "Jane Forman on Tennis, Parents Gone Wild, and Beating the Odds",
        "description": "Former professional tennis player Jane Forman joins Bo and Joe to talk tennis vs. pickleball",
        "expected_guests": ["Jane Forman"],
        "not_guests": ["Bo", "Joe"]
    },
    {
        "title": "Appraisers & AI with Roy Meyer",
        "description": "Dan Lindeman sits down with Roy Meyer—an experienced appraiser, business coach, AI strategist",
        "expected_guests": ["Roy Meyer"],
        "not_guests": ["Dan Lindeman"]
    },
    {
        "title": "How Your Relationship Changes When Your Partner Has Narcissism and ADHD",
        "description": "This week, ADHD coach and neurodiversity educator Sheila Henson joins Dr. Kerry to explore",
        "expected_guests": ["Sheila Henson"],
        "not_guests": ["Dr. Kerry"]
    },
    {
        "title": "Episode 144: Travis Hunter and how other Non-QB Heisman Winners Fared",
        "description": "In this episode of The 40/40 Vision Podcast, host Khaled Abdallah discusses Travis Hunter",
        "expected_guests": [],  # No guest, just host discussing a topic
        "not_guests": ["Khaled Abdallah"]
    },
    {
        "title": "Breaking Stigmas: Mental Health & Workplace Wellness With Brian Higgins",
        "description": "What happens when traditional mental health therapies fail? Brian Higgins shares his journey",
        "expected_guests": ["Brian Higgins"],
        "not_guests": []
    }
]

print("COMPARING GUEST EXTRACTION METHODS")
print("="*80)

for i, test in enumerate(test_cases):
    print(f"\nTest Case {i+1}:")
    print(f"Title: {test['title']}")
    print(f"Expected guests: {test['expected_guests']}")
    print(f"Should NOT extract: {test['not_guests']}")
    
    # Combine text
    text = f"{test['title']} {test['description']}"
    
    # Old method
    old_results = extract_old(text)
    print(f"\nOLD extraction: {old_results}")
    
    # New method
    new_results = extract_new(text)
    print(f"NEW extraction: {new_results}")
    
    # Check accuracy
    old_correct = sum(1 for g in test['expected_guests'] if any(g in r for r in old_results))
    new_correct = sum(1 for g in test['expected_guests'] if any(g in r for r in new_results))
    
    old_false = sum(1 for g in test['not_guests'] if any(g in r for r in old_results))
    new_false = sum(1 for g in test['not_guests'] if any(g in r for r in new_results))
    
    print(f"\nAccuracy:")
    print(f"  OLD: {old_correct}/{len(test['expected_guests'])} correct, {old_false} false positives")
    print(f"  NEW: {new_correct}/{len(test['expected_guests'])} correct, {new_false} false positives")
    
    print("-"*40)