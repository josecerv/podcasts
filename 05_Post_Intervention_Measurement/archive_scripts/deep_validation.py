#!/usr/bin/env python3
"""
Deep validation of enhanced guest extraction pipeline
Tests with real podcast data and provides detailed accuracy metrics
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import feedparser
import socket
import time
import re
from collections import defaultdict, Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import enhanced modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from post_intervention_guest_enhanced import (
        extract_guests_with_patterns,
        extract_guests_with_ner,
        extract_guest_context,
        clean_text,
        get_robust_description,
        fetch_rss_with_retry
    )
    import spacy
    nlp = spacy.load("en_core_web_sm")
    ENHANCED_AVAILABLE = True
except Exception as e:
    logger.error(f"Error loading enhanced modules: {e}")
    ENHANCED_AVAILABLE = False

class DeepValidator:
    def __init__(self):
        self.results = {
            'rss_fetching': {},
            'guest_extraction': {},
            'demographic_analysis': {},
            'comparison': {}
        }
        
    def validate_rss_fetching(self, sample_size=20):
        """Test RSS fetching with actual podcast feeds"""
        logger.info(f"\n=== Testing RSS Fetching (sample size: {sample_size}) ===")
        
        # Load podcast RSS data
        try:
            df = pd.read_csv('podcast_rss_export.csv')
            # Sample diverse podcasts
            sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            success_count = 0
            total_episodes = 0
            failed_feeds = []
            
            for idx, row in sample_df.iterrows():
                podcast_info = {
                    'podcastID': row['podcastID'],
                    'rss': row['rss'],
                    'completion_timestamp': pd.to_datetime(row['latest_response_time'], utc=True)
                }
                
                logger.info(f"Testing podcast {idx+1}/{len(sample_df)}: {podcast_info['podcastID'][:8]}...")
                
                # Test with enhanced fetcher
                result = fetch_rss_with_retry(podcast_info)
                
                if result and hasattr(result, 'entries') and result.entries:
                    success_count += 1
                    episodes_found = len(result.entries)
                    total_episodes += episodes_found
                    logger.info(f"  ✓ Success: {episodes_found} episodes found")
                    
                    # Analyze first episode for content
                    if result.entries:
                        first_entry = result.entries[0]
                        title = clean_text(first_entry.get('title', ''))
                        desc = get_robust_description(first_entry)
                        logger.debug(f"  First episode: {title[:60]}...")
                        logger.debug(f"  Description length: {len(desc)} chars")
                else:
                    failed_feeds.append({
                        'podcast_id': podcast_info['podcastID'],
                        'url': podcast_info['rss']
                    })
                    logger.warning(f"  ✗ Failed to fetch")
            
            # Calculate metrics
            success_rate = (success_count / len(sample_df)) * 100
            avg_episodes = total_episodes / success_count if success_count > 0 else 0
            
            self.results['rss_fetching'] = {
                'sample_size': len(sample_df),
                'successful_feeds': success_count,
                'failed_feeds': len(failed_feeds),
                'success_rate': success_rate,
                'total_episodes': total_episodes,
                'avg_episodes_per_feed': avg_episodes,
                'failed_feed_details': failed_feeds
            }
            
            logger.info(f"\nRSS Fetching Results:")
            logger.info(f"  Success rate: {success_rate:.2f}% (Target: 95%+)")
            logger.info(f"  Average episodes per feed: {avg_episodes:.1f}")
            
            if success_rate < 95:
                logger.warning(f"  ⚠️ Success rate below 95% target!")
            else:
                logger.info(f"  ✓ Success rate meets target!")
                
        except Exception as e:
            logger.error(f"Error in RSS validation: {e}")
            
    def validate_guest_extraction(self):
        """Test guest extraction with known episodes"""
        logger.info("\n=== Testing Guest Extraction Accuracy ===")
        
        # Test cases with known guests
        test_episodes = [
            {
                'title': "EP 101: Interview with Dr. Sarah Johnson from MIT",
                'description': "Today we're joined by Dr. Sarah Johnson, a professor at MIT and author of 'The Future of Technology'. Sarah discusses her groundbreaking research on artificial intelligence and shares insights from her 20 years in the field.",
                'expected_guests': ['Sarah Johnson'],
                'expected_context': {
                    'Sarah Johnson': {
                        'affiliation': 'MIT',
                        'title': 'Dr./Professor',
                        'book': 'The Future of Technology'
                    }
                }
            },
            {
                'title': "Special Guest: Marcus Williams, CEO of TechStart",
                'description': "Featuring Marcus Williams, founder and CEO of TechStart. Marcus shares his journey from college dropout to successful entrepreneur. Also joining us is his co-founder, Jennifer Chen, who leads product development.",
                'expected_guests': ['Marcus Williams', 'Jennifer Chen'],
                'expected_context': {
                    'Marcus Williams': {
                        'title': 'CEO/Founder',
                        'affiliation': 'TechStart'
                    },
                    'Jennifer Chen': {
                        'title': 'co-founder',
                        'affiliation': 'TechStart'
                    }
                }
            },
            {
                'title': "The Marketing Show - Episode 45",
                'description': "This week's episode covers new marketing trends. We discuss social media strategies and content creation tips. No specific guests in this solo episode.",
                'expected_guests': [],
                'expected_context': {}
            },
            {
                'title': "Conversation with Nobel Laureate Dr. Raj Patel",
                'description': "An exclusive interview with Dr. Raj Patel, Nobel Prize winner in Economics. Dr. Patel, currently a professor at Harvard University, discusses his new book 'Global Economics in Crisis' and shares his views on monetary policy.",
                'expected_guests': ['Raj Patel'],
                'expected_context': {
                    'Raj Patel': {
                        'title': 'Dr./Professor',
                        'affiliation': 'Harvard University',
                        'book': 'Global Economics in Crisis'
                    }
                }
            }
        ]
        
        extraction_results = []
        
        for i, episode in enumerate(test_episodes):
            logger.info(f"\nTest Episode {i+1}: {episode['title']}")
            
            # Extract guests using pattern matching
            pattern_guests = extract_guests_with_patterns(episode['title'], episode['description'])
            logger.info(f"  Pattern extraction: {len(pattern_guests)} guests")
            
            # Extract guests using NER
            full_text = f"{episode['title']} {episode['description']}"
            ner_guests = extract_guests_with_ner(full_text)
            logger.info(f"  NER extraction: {len(ner_guests)} guests")
            
            # Combine and deduplicate
            all_guests = {}
            for guest in pattern_guests + ner_guests:
                name_key = guest['name'].lower()
                if name_key not in all_guests or guest['confidence'] > all_guests[name_key]['confidence']:
                    all_guests[name_key] = guest
            
            unique_guests = list(all_guests.values())
            found_names = [g['name'] for g in unique_guests]
            
            # Extract context for each guest
            for guest in unique_guests:
                context = extract_guest_context(full_text, guest['name'])
                guest['context'] = context
            
            # Evaluate accuracy
            expected_set = set(name.lower() for name in episode['expected_guests'])
            found_set = set(name.lower() for name in found_names)
            
            true_positives = len(expected_set & found_set)
            false_positives = len(found_set - expected_set)
            false_negatives = len(expected_set - found_set)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            result = {
                'episode': i + 1,
                'expected_guests': episode['expected_guests'],
                'found_guests': found_names,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'context_accuracy': {}
            }
            
            # Check context accuracy
            for guest in unique_guests:
                if guest['name'] in episode['expected_context']:
                    expected_ctx = episode['expected_context'][guest['name']]
                    found_ctx = guest['context']
                    
                    context_matches = 0
                    context_total = 0
                    
                    for key in ['affiliation', 'title', 'book']:
                        if expected_ctx.get(key):
                            context_total += 1
                            if found_ctx.get(key) and any(part.lower() in found_ctx[key].lower() 
                                                         for part in expected_ctx[key].split('/')):
                                context_matches += 1
                    
                    context_accuracy = context_matches / context_total if context_total > 0 else 0
                    result['context_accuracy'][guest['name']] = context_accuracy
            
            extraction_results.append(result)
            
            logger.info(f"  Expected: {episode['expected_guests']}")
            logger.info(f"  Found: {found_names}")
            logger.info(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}")
            
            # Show context extraction
            for guest in unique_guests:
                if any(guest['context'].values()):
                    logger.info(f"  Context for {guest['name']}:")
                    for key, value in guest['context'].items():
                        if value:
                            logger.info(f"    - {key}: {value}")
        
        # Calculate overall metrics
        avg_precision = np.mean([r['precision'] for r in extraction_results])
        avg_recall = np.mean([r['recall'] for r in extraction_results])
        avg_f1 = np.mean([r['f1_score'] for r in extraction_results])
        
        # Context accuracy
        all_context_scores = []
        for result in extraction_results:
            all_context_scores.extend(result['context_accuracy'].values())
        avg_context_accuracy = np.mean(all_context_scores) if all_context_scores else 0
        
        self.results['guest_extraction'] = {
            'test_episodes': len(test_episodes),
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1,
            'avg_context_accuracy': avg_context_accuracy,
            'detailed_results': extraction_results
        }
        
        logger.info(f"\nGuest Extraction Summary:")
        logger.info(f"  Average Precision: {avg_precision:.2%}")
        logger.info(f"  Average Recall: {avg_recall:.2%}")
        logger.info(f"  Average F1 Score: {avg_f1:.2%}")
        logger.info(f"  Context Extraction Accuracy: {avg_context_accuracy:.2%}")
        
        if avg_f1 >= 0.95:
            logger.info(f"  ✓ Extraction accuracy meets 95%+ target!")
        else:
            logger.warning(f"  ⚠️ Extraction accuracy below target")
    
    def analyze_real_episodes(self, sample_size=50):
        """Analyze real podcast episodes from RSS feeds"""
        logger.info(f"\n=== Analyzing Real Episodes (sample size: {sample_size}) ===")
        
        try:
            # Load podcast data
            df = pd.read_csv('podcast_rss_export.csv')
            sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            episode_analysis = []
            guest_distribution = defaultdict(int)
            extraction_methods = Counter()
            
            episodes_processed = 0
            
            for _, row in sample_df.iterrows():
                if episodes_processed >= sample_size:
                    break
                    
                podcast_info = {
                    'podcastID': row['podcastID'],
                    'rss': row['rss'],
                    'completion_timestamp': pd.to_datetime(row['latest_response_time'], utc=True)
                }
                
                # Fetch feed
                feed = fetch_rss_with_retry(podcast_info)
                if not feed or not hasattr(feed, 'entries') or not feed.entries:
                    continue
                
                # Analyze first few episodes
                for entry in feed.entries[:2]:  # First 2 episodes per podcast
                    if episodes_processed >= sample_size:
                        break
                        
                    title = clean_text(entry.get('title', ''))
                    description = get_robust_description(entry)
                    
                    if not title and not description:
                        continue
                    
                    # Extract guests
                    pattern_guests = extract_guests_with_patterns(title, description)
                    ner_guests = extract_guests_with_ner(f"{title} {description}")
                    
                    # Track extraction methods
                    for g in pattern_guests:
                        extraction_methods['pattern'] += 1
                    for g in ner_guests:
                        extraction_methods['ner'] += 1
                    
                    # Combine guests
                    all_guests = {}
                    for guest in pattern_guests + ner_guests:
                        name_key = guest['name'].lower()
                        if name_key not in all_guests:
                            all_guests[name_key] = guest
                    
                    num_guests = len(all_guests)
                    guest_distribution[num_guests] += 1
                    
                    episode_analysis.append({
                        'podcast_id': podcast_info['podcastID'],
                        'title': title[:100],
                        'num_guests': num_guests,
                        'guests': list(all_guests.keys()),
                        'has_pattern_matches': len(pattern_guests) > 0,
                        'has_ner_matches': len(ner_guests) > 0
                    })
                    
                    episodes_processed += 1
                    
                    if num_guests > 0:
                        logger.info(f"Episode: {title[:60]}...")
                        logger.info(f"  Found {num_guests} guests: {', '.join(g['name'] for g in all_guests.values())}")
            
            # Analyze results
            episodes_with_guests = sum(1 for e in episode_analysis if e['num_guests'] > 0)
            pct_with_guests = (episodes_with_guests / len(episode_analysis) * 100) if episode_analysis else 0
            
            self.results['real_episodes'] = {
                'episodes_analyzed': len(episode_analysis),
                'episodes_with_guests': episodes_with_guests,
                'pct_episodes_with_guests': pct_with_guests,
                'guest_distribution': dict(guest_distribution),
                'extraction_methods': dict(extraction_methods),
                'avg_guests_per_episode': np.mean([e['num_guests'] for e in episode_analysis]) if episode_analysis else 0
            }
            
            logger.info(f"\nReal Episode Analysis:")
            logger.info(f"  Episodes analyzed: {len(episode_analysis)}")
            logger.info(f"  Episodes with guests: {episodes_with_guests} ({pct_with_guests:.1f}%)")
            logger.info(f"  Average guests per episode: {self.results['real_episodes']['avg_guests_per_episode']:.2f}")
            logger.info(f"  Guest count distribution: {dict(guest_distribution)}")
            
        except Exception as e:
            logger.error(f"Error analyzing real episodes: {e}")
    
    def compare_with_baseline(self):
        """Compare enhanced extraction with original results"""
        logger.info("\n=== Comparing with Original Pipeline ===")
        
        try:
            # Load a sample of original results
            original_results = []
            with open('post_intervention_guest_classification_full.jsonl', 'r') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # Sample first 100
                        break
                    original_results.append(json.loads(line))
            
            # Compare guest counts
            original_total_guests = sum(r.get('total_guests', 0) for r in original_results)
            original_with_guests = sum(1 for r in original_results if r.get('total_guests', 0) > 0)
            
            self.results['comparison'] = {
                'sample_size': len(original_results),
                'original_total_guests': original_total_guests,
                'original_episodes_with_guests': original_with_guests,
                'original_pct_with_guests': (original_with_guests / len(original_results) * 100) if original_results else 0
            }
            
            logger.info(f"Original Pipeline (sample of {len(original_results)} episodes):")
            logger.info(f"  Total guests found: {original_total_guests}")
            logger.info(f"  Episodes with guests: {original_with_guests} ({self.results['comparison']['original_pct_with_guests']:.1f}%)")
            
            # If we have real episode results, compare
            if 'real_episodes' in self.results:
                enhanced_pct = self.results['real_episodes']['pct_episodes_with_guests']
                original_pct = self.results['comparison']['original_pct_with_guests']
                improvement = ((enhanced_pct - original_pct) / original_pct * 100) if original_pct > 0 else 0
                
                logger.info(f"\nImprovement Analysis:")
                logger.info(f"  Original: {original_pct:.1f}% episodes with guests")
                logger.info(f"  Enhanced: {enhanced_pct:.1f}% episodes with guests")
                logger.info(f"  Improvement: {improvement:+.1f}%")
                
        except Exception as e:
            logger.error(f"Error comparing with baseline: {e}")
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        logger.info("\n" + "="*60)
        logger.info("DEEP VALIDATION REPORT")
        logger.info("="*60)
        
        # RSS Fetching Performance
        if 'rss_fetching' in self.results and self.results['rss_fetching']:
            rss = self.results['rss_fetching']
            logger.info("\n1. RSS FETCHING PERFORMANCE")
            logger.info(f"   Success Rate: {rss['success_rate']:.2f}% {'✓' if rss['success_rate'] >= 95 else '✗'}")
            logger.info(f"   Feeds Tested: {rss['sample_size']}")
            logger.info(f"   Failed Feeds: {rss['failed_feeds']}")
            
            if rss['success_rate'] < 95:
                logger.info("\n   Failed Feed Analysis:")
                for feed in rss['failed_feed_details'][:5]:  # Show first 5
                    logger.info(f"   - {feed['podcast_id'][:8]}... : {feed['url'][:50]}...")
        
        # Guest Extraction Accuracy
        if 'guest_extraction' in self.results and self.results['guest_extraction']:
            extraction = self.results['guest_extraction']
            logger.info("\n2. GUEST EXTRACTION ACCURACY")
            logger.info(f"   F1 Score: {extraction['avg_f1_score']:.2%} {'✓' if extraction['avg_f1_score'] >= 0.95 else '✗'}")
            logger.info(f"   Precision: {extraction['avg_precision']:.2%}")
            logger.info(f"   Recall: {extraction['avg_recall']:.2%}")
            logger.info(f"   Context Accuracy: {extraction['avg_context_accuracy']:.2%}")
        
        # Real Episode Analysis
        if 'real_episodes' in self.results and self.results['real_episodes']:
            real = self.results['real_episodes']
            logger.info("\n3. REAL EPISODE ANALYSIS")
            logger.info(f"   Episodes Analyzed: {real['episodes_analyzed']}")
            logger.info(f"   With Guests: {real['pct_episodes_with_guests']:.1f}%")
            logger.info(f"   Avg Guests/Episode: {real['avg_guests_per_episode']:.2f}")
            logger.info(f"   Extraction Methods: Pattern={real['extraction_methods'].get('pattern', 0)}, NER={real['extraction_methods'].get('ner', 0)}")
        
        # Comparison with Original
        if 'comparison' in self.results and self.results['comparison']:
            comp = self.results['comparison']
            logger.info("\n4. COMPARISON WITH ORIGINAL PIPELINE")
            logger.info(f"   Original Episodes with Guests: {comp['original_pct_with_guests']:.1f}%")
            
            if 'real_episodes' in self.results:
                enhanced_pct = self.results['real_episodes']['pct_episodes_with_guests']
                improvement = ((enhanced_pct - comp['original_pct_with_guests']) / comp['original_pct_with_guests'] * 100) if comp['original_pct_with_guests'] > 0 else 0
                logger.info(f"   Enhanced Episodes with Guests: {enhanced_pct:.1f}%")
                logger.info(f"   Relative Improvement: {improvement:+.1f}%")
        
        # Overall Assessment
        logger.info("\n5. OVERALL ASSESSMENT")
        
        meets_rss_target = self.results.get('rss_fetching', {}).get('success_rate', 0) >= 95
        meets_extraction_target = self.results.get('guest_extraction', {}).get('avg_f1_score', 0) >= 0.95
        
        if meets_rss_target and meets_extraction_target:
            logger.info("   ✓ BOTH TARGETS MET: System ready for production use")
        else:
            logger.info("   ✗ TARGETS NOT MET:")
            if not meets_rss_target:
                logger.info("     - RSS fetching below 95% success rate")
            if not meets_extraction_target:
                logger.info("     - Guest extraction below 95% F1 score")
        
        # Recommendations
        logger.info("\n6. RECOMMENDATIONS")
        
        if not meets_rss_target:
            logger.info("   For RSS Fetching:")
            logger.info("   - Increase timeout for slow feeds")
            logger.info("   - Add more user agents")
            logger.info("   - Implement proxy rotation for blocked feeds")
        
        if not meets_extraction_target:
            logger.info("   For Guest Extraction:")
            logger.info("   - Add more guest introduction patterns")
            logger.info("   - Fine-tune NER model on podcast data")
            logger.info("   - Implement speaker diarization for audio analysis")
        
        logger.info("\n" + "="*60)
        
        # Save detailed report
        report_file = "deep_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"\nDetailed results saved to: {report_file}")

def main():
    """Run deep validation"""
    if not ENHANCED_AVAILABLE:
        logger.error("Enhanced modules not available. Please run setup first.")
        return
    
    validator = DeepValidator()
    
    # Run all validations
    validator.validate_rss_fetching(sample_size=30)  # Test 30 feeds
    validator.validate_guest_extraction()  # Test known examples
    validator.analyze_real_episodes(sample_size=50)  # Analyze 50 real episodes
    validator.compare_with_baseline()  # Compare with original
    
    # Generate report
    validator.generate_report()

if __name__ == "__main__":
    main()