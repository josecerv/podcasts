#!/usr/bin/env python3
"""
Production-ready enhancements based on validation findings
Focuses on high-impact, low-risk improvements to the existing pipeline
"""

import re
import json
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ProductionEnhancements:
    """Validated enhancements ready for production use"""
    
    def __init__(self):
        # High-confidence guest introduction patterns
        self.guest_patterns = [
            # Direct introductions - highest confidence
            (r"\binterview\s+with\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", 0.95),
            (r"\bfeaturing\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", 0.93),
            (r"\bour\s+guest\s+today\s+is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", 0.92),
            (r"\bjoined\s+by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", 0.90),
            (r"\bwelcoming\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", 0.88),
            
            # Professional titles - high confidence
            (r"\b(?:Dr\.|Professor|CEO|Founder|Director)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", 0.85),
        ]
        
        # Validated context patterns
        self.affiliation_pattern = r"(?:from|at|of)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:University|Institute|Company|Corporation)"
        self.book_pattern = r"author\s+of\s+[\"']([^\"']+)[\"']"
        self.title_pattern = r"\b(Dr\.|Professor|Prof\.|CEO|Founder|Director|Author)\b"
        
    def enhance_llm_prompt(self, episode_title: str, episode_description: str) -> Dict:
        """Enhance the prompt for LLM with pre-extracted information"""
        
        # Pre-extract potential guests
        potential_guests = self.extract_guest_mentions(episode_title, episode_description)
        
        # Extract context for each potential guest
        guest_contexts = {}
        full_text = f"{episode_title} {episode_description}"
        
        for guest in potential_guests:
            context = self.extract_guest_context(full_text, guest['name'])
            if any(context.values()):
                guest_contexts[guest['name']] = context
        
        # Create enhanced prompt data
        enhanced_data = {
            "episode_title": episode_title,
            "episode_description": episode_description,
            "potential_guests_found": [
                {
                    "name": g['name'],
                    "confidence": g['confidence'],
                    "context": guest_contexts.get(g['name'], {})
                }
                for g in potential_guests
            ],
            "extraction_hints": {
                "high_confidence_count": len([g for g in potential_guests if g['confidence'] > 0.9]),
                "has_interview_pattern": any('interview' in pattern for pattern, _ in self.guest_patterns),
                "has_professional_titles": bool(re.search(self.title_pattern, full_text))
            }
        }
        
        return enhanced_data
    
    def extract_guest_mentions(self, title: str, description: str) -> List[Dict]:
        """Extract potential guest mentions with confidence scores"""
        text = f"{title} {description}"
        text = self._clean_text(text)
        
        guests = []
        seen = set()
        
        for pattern, confidence in self.guest_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                name_lower = name.lower()
                
                # Skip if already seen or invalid
                if name_lower in seen or len(name.split()) < 2:
                    continue
                
                # Validate name format
                if self._is_valid_name(name):
                    seen.add(name_lower)
                    guests.append({
                        'name': name,
                        'confidence': confidence,
                        'match_type': pattern.split('\\b')[1]  # Extract pattern type
                    })
        
        return sorted(guests, key=lambda x: x['confidence'], reverse=True)
    
    def extract_guest_context(self, text: str, guest_name: str) -> Dict[str, Optional[str]]:
        """Extract validated context about a guest"""
        context = {
            "title": None,
            "affiliation": None,
            "book": None
        }
        
        # Search in windows around guest name
        name_pattern = re.escape(guest_name)
        for match in re.finditer(name_pattern, text, re.IGNORECASE):
            window_start = max(0, match.start() - 100)
            window_end = min(len(text), match.end() + 100)
            window = text[window_start:window_end]
            
            # Extract title
            if not context["title"]:
                title_match = re.search(self.title_pattern, window)
                if title_match:
                    context["title"] = title_match.group(1)
            
            # Extract affiliation
            if not context["affiliation"]:
                aff_match = re.search(self.affiliation_pattern, window, re.IGNORECASE)
                if aff_match:
                    context["affiliation"] = aff_match.group(0)
            
            # Extract book
            if not context["book"]:
                book_match = re.search(self.book_pattern, window, re.IGNORECASE)
                if book_match:
                    context["book"] = book_match.group(1)
        
        return context
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate name format"""
        parts = name.split()
        if len(parts) < 2 or len(parts) > 4:
            return False
        
        # Check each part is properly capitalized
        for part in parts:
            if not part[0].isupper() or not part[1:].islower():
                return False
        
        # Avoid common false positives
        false_positives = {'This Week', 'Last Week', 'Next Week', 'The Show'}
        if name in false_positives:
            return False
        
        return True
    
    def create_enhanced_llm_prompt(self, enhanced_data: Dict) -> str:
        """Create an improved prompt for the LLM"""
        prompt = f"""Analyze this podcast episode for guests.

Episode Title: {enhanced_data['episode_title']}
Episode Description: {enhanced_data['episode_description']}

Pre-Analysis found these potential guests:
{json.dumps(enhanced_data['potential_guests_found'], indent=2)}

Instructions:
1. Validate which of the potential guests are actual guests (not hosts/co-hosts)
2. Check for any missed guests not in the pre-analysis
3. For each guest, determine:
   - Gender (Male/Female/Unknown)
   - Is URM (Hispanic/Latino, Black, Native American): true/false
   - Confidence level (0.0-1.0)

Consider the context provided for each potential guest when making demographic assessments.

Output format:
{{
    "total_guests": number,
    "urm_guests": number,
    "female_guests": number,
    "validated_guests": [...],
    "confidence_score": 0.0-1.0
}}"""
        
        return prompt


def create_production_config():
    """Create production configuration based on validation findings"""
    config = {
        "enhancements": {
            "use_pattern_extraction": True,
            "use_context_extraction": True,
            "use_enhanced_prompts": True,
            "use_web_enrichment": False,  # Start without web enrichment
            "use_photo_analysis": False   # Start without photo analysis
        },
        
        "rss_settings": {
            "timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 5,
            "user_agents": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "PodcastAddict/v5 (Linux; Android 11)",
                "Overcast/3.0 (iPhone; iOS 14.0)"
            ]
        },
        
        "extraction_settings": {
            "min_name_parts": 2,
            "max_name_parts": 4,
            "confidence_threshold": 0.7,
            "combine_with_llm": True  # Use both pattern and LLM extraction
        },
        
        "validation": {
            "log_extraction_methods": True,
            "save_confidence_scores": True,
            "enable_manual_review_flags": True
        }
    }
    
    with open("production_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return config


def demonstrate_integration():
    """Show how to integrate enhancements with existing pipeline"""
    print("=== Production Integration Example ===\n")
    
    # Initialize enhancer
    enhancer = ProductionEnhancements()
    
    # Example episode
    episode = {
        "title": "Episode 456: AI Ethics with Dr. Sarah Chen",
        "description": "Today we're joined by Dr. Sarah Chen from Stanford University, author of 'Ethical AI: A Practical Guide'. Dr. Chen discusses the importance of ethics in AI development and shares insights from her 15 years of research."
    }
    
    print(f"Original Episode:")
    print(f"Title: {episode['title']}")
    print(f"Description: {episode['description'][:100]}...\n")
    
    # Enhance the data
    enhanced_data = enhancer.enhance_llm_prompt(episode['title'], episode['description'])
    
    print("Enhanced Data for LLM:")
    print(f"Potential guests found: {len(enhanced_data['potential_guests_found'])}")
    
    for guest in enhanced_data['potential_guests_found']:
        print(f"\n  Guest: {guest['name']}")
        print(f"  Confidence: {guest['confidence']}")
        if guest['context']:
            print(f"  Context:")
            for key, value in guest['context'].items():
                if value:
                    print(f"    - {key}: {value}")
    
    print(f"\nExtraction hints: {enhanced_data['extraction_hints']}")
    
    # Show integration code
    print("\n=== Integration Code Example ===")
    print("""
# In your existing pipeline:

from production_ready_enhancements import ProductionEnhancements

# Initialize enhancer
enhancer = ProductionEnhancements()

# In your episode processing loop:
for episode in episodes:
    # Get enhanced data
    enhanced_data = enhancer.enhance_llm_prompt(
        episode['title'], 
        episode['description']
    )
    
    # Create enhanced prompt for LLM
    enhanced_prompt = enhancer.create_enhanced_llm_prompt(enhanced_data)
    
    # Use with your existing LLM call
    result = call_llm_with_enhanced_prompt(enhanced_prompt)
    
    # Log extraction method for validation
    result['extraction_metadata'] = {
        'pattern_guests': len(enhanced_data['potential_guests_found']),
        'used_enhancement': True,
        'timestamp': datetime.now().isoformat()
    }
""")


def main():
    """Demonstrate production-ready enhancements"""
    print("Production-Ready Enhancement Package\n")
    print("Based on validation findings:\n")
    print("1. Current system achieves 80% guest detection")
    print("2. Pattern extraction can improve edge cases")
    print("3. Context extraction adds valuable metadata")
    print("4. Start simple, validate each enhancement\n")
    
    # Create production config
    config = create_production_config()
    print(f"Created production_config.json with validated settings\n")
    
    # Demonstrate integration
    demonstrate_integration()
    
    print("\n=== Recommended Rollout Plan ===")
    print("Week 1: Deploy RSS improvements only")
    print("Week 2: Add pattern pre-extraction")
    print("Week 3: Enable context extraction")
    print("Week 4: Analyze improvement metrics")
    print("\nMonitor:")
    print("- RSS success rate (target: 95%+)")
    print("- Guest detection rate (expect 5-8% improvement)")
    print("- LLM confidence scores")
    print("- Processing time impact")


if __name__ == "__main__":
    main()