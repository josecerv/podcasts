#!/usr/bin/env python3
"""Test improved patterns independently"""
import re

# OLD PATTERNS
OLD_PATTERNS = [
    r"(?:featuring|feature|feat\.?|with guest[s]?|guest[s]?:|interview with|joined by|welcoming|special guest[s]?|our guest[s]?|today'?s guest[s]?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:joins us|is here|stops by|visits|shares|discusses|talks about)",
    r"(?:author|professor|dr\.?|expert|ceo|founder|director|coach|consultant)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}),\s+(?:author of|professor at|ceo of|founder of|director of)",
]

# NEW PATTERNS with boundaries
NEW_PATTERNS = [
    r"\b(?:featuring|feature|feat\.?|with guest[s]?|guest[s]?:|interview with|interviewed|joined by|welcoming|special guest[s]?|our guest[s]?|today'?s guest[s]?)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2})\b",
    r"\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2})\s+(?:joins us|is here|stops by|visits us|shares|discusses|talks about|tells us)\b",
    r"\b(?:author|professor|prof\.?|dr\.?|expert|ceo|founder|director|coach|consultant|journalist|reporter)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2})(?:\s*,|\s+(?:joins|shares|discusses|talks))\b",
    r"\b([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2}),\s+(?:author of|professor at|ceo of|founder of|director of|host of)\b",
    r"\b(?:conversation with|chat with|talk with|speaking with|interview with)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?(?:[a-z]+)?(?:\s+[A-Z][a-z]+){1,2})\b",
]

# Test cases from real episodes
test_cases = [
    ("Stu Weinstein: The Dolphins' Secret Keeper on Shula, Security, and Scandals",
     "Bo & Joe sit down with longtime Miami Dolphins Director of Security, Stu Weinstein.",
     ["Stu Weinstein"]),
    
    ("Episode 144: Travis Hunter and how other Non-QB Heisman Winners Fared in the NFL",
     "In this episode of The 40/40 Vision Podcast, host Khaled Abdallah discusses the potential",
     []),  # No guest, just host
    
    ("Mary Crocker Cook joins us to discuss red flags",
     "Why do we pick bad partners? This week, author and clinician, Mary Crocker Cook joins us",
     ["Mary Crocker Cook"]),
    
    ("Dr. Rob Moyer, Brooke Elementary School Principal",
     "chatted with Dr. Rob Moyer, Brooke Elementary School Principal, and Tara Chester",
     ["Rob Moyer", "Tara Chester"]),
]

def extract_names(patterns, text):
    """Extract names using given patterns"""
    found = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            name = match.strip()
            # Basic validation
            if len(name.split()) >= 2 and len(name) < 50:
                found.add(name)
    return list(found)

print("PATTERN COMPARISON TEST")
print("="*80)

for title, desc, expected in test_cases:
    text = f"{title} {desc}"
    print(f"\nText: {text[:100]}...")
    print(f"Expected: {expected}")
    
    old_results = extract_names(OLD_PATTERNS, text)
    new_results = extract_names(NEW_PATTERNS, text)
    
    print(f"OLD found: {old_results}")
    print(f"NEW found: {new_results}")
    
    # Check if we caught the expected names
    old_correct = sum(1 for e in expected if any(e in r or r in e for r in old_results))
    new_correct = sum(1 for e in expected if any(e in r or r in e for r in new_results))
    
    print(f"Accuracy: OLD {old_correct}/{len(expected)}, NEW {new_correct}/{len(expected)}")
    print("-"*40)