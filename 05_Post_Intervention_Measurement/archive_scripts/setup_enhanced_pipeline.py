#!/usr/bin/env python3
"""
Setup script for the enhanced podcast guest analysis pipeline
"""

import os
import sys
import subprocess
import json

def check_python_version():
    """Check if Python version is 3.9 or higher"""
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version} detected")

def install_poetry():
    """Install Poetry if not already installed"""
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        print("✓ Poetry is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing Poetry...")
        subprocess.run([sys.executable, "-m", "pip", "install", "poetry"], check=True)
        print("✓ Poetry installed successfully")

def install_dependencies():
    """Install project dependencies using Poetry"""
    print("\nInstalling project dependencies...")
    try:
        subprocess.run(["poetry", "install"], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("\nTrying alternative installation methods...")
        
        # Fallback to pip
        requirements = [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "feedparser>=6.0.10",
            "openai>=1.0.0",
            "scipy>=1.11.0",
            "tqdm>=4.66.0",
            "requests>=2.31.0",
            "beautifulsoup4>=4.12.0",
            "spacy>=3.7.0",
            "transformers>=4.35.0",
            "deepface>=0.0.79",
            "opencv-python>=4.8.0",
            "tensorflow>=2.14.0",
            "Pillow>=10.0.0",
            "nameparser>=1.1.3",
            "nltk>=3.8.0",
            "joblib>=1.3.0",
            "scikit-learn>=1.3.0"
        ]
        
        for req in requirements:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to install {req}")

def download_spacy_model():
    """Download spaCy English model"""
    print("\nDownloading spaCy English model...")
    try:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("✓ spaCy model downloaded successfully")
    except subprocess.CalledProcessError:
        print("Warning: Failed to download spaCy model. You may need to run:")
        print("python -m spacy download en_core_web_sm")

def check_environment_variables():
    """Check required environment variables"""
    print("\nChecking environment variables...")
    
    required_vars = {
        "OPENAI_API_KEY": "Required for LLM guest classification",
        "SERPAPI_KEY": "Optional - for web search enrichment",
        "GOOGLE_SEARCH_API_KEY": "Optional - alternative to SERPAPI",
        "GOOGLE_CSE_ID": "Optional - required with GOOGLE_SEARCH_API_KEY"
    }
    
    missing_required = []
    missing_optional = []
    
    for var, description in required_vars.items():
        if os.getenv(var):
            print(f"✓ {var} is set")
        else:
            if "Required" in description:
                missing_required.append((var, description))
            else:
                missing_optional.append((var, description))
    
    if missing_required:
        print("\n❌ Missing REQUIRED environment variables:")
        for var, desc in missing_required:
            print(f"  - {var}: {desc}")
        print("\nSet these before running the pipeline:")
        print("  export OPENAI_API_KEY='your-key-here'")
    
    if missing_optional:
        print("\n⚠️  Missing optional environment variables:")
        for var, desc in missing_optional:
            print(f"  - {var}: {desc}")
        print("\nThese enable additional features but are not required.")

def create_directories():
    """Create necessary directories"""
    print("\nCreating necessary directories...")
    
    directories = [
        "guest_photos",
        "enrichment_cache",
        "logs",
        "output"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "rss_settings": {
            "timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 5,
            "min_required_episodes": 1
        },
        "llm_settings": {
            "model": "gpt-4o-mini",
            "max_workers": 20,
            "retry_attempts": 3,
            "temperature": 0.1
        },
        "enrichment_settings": {
            "enable_web_search": True,
            "enable_photo_analysis": True,
            "max_images_per_guest": 3,
            "photo_analysis_backends": ["opencv", "retinaface"]
        },
        "output_settings": {
            "save_guest_level_data": True,
            "save_enrichment_cache": True,
            "detailed_logging": True
        }
    }
    
    with open("pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n✓ Created sample configuration file: pipeline_config.json")

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    
    print("\nTo run the enhanced pipeline:")
    print("1. Ensure your environment variables are set:")
    print("   export OPENAI_API_KEY='your-key-here'")
    print("   export SERPAPI_KEY='your-key-here'  # Optional")
    print("\n2. Ensure your podcast_rss_export.csv file is in the current directory")
    print("\n3. Run the pipeline:")
    print("   python post-intervention-guest-enhanced.py")
    print("\nKey improvements in this version:")
    print("- Pattern-based and NER guest extraction")
    print("- Robust RSS fetching with retries and success tracking")
    print("- Multi-source demographic analysis")
    print("- Guest context enrichment")
    print("- Photo-based analysis with DeepFace")
    print("- Detailed guest-level output data")
    print("\nOutput files:")
    print("- post_intervention_guest_classification_enhanced.jsonl")
    print("- post_intervention_guest_summary_enhanced.csv")
    print("- post_intervention_guests_enriched.csv")
    print("- rss_fetch_success_log.csv")
    print("- post_intervention_error_log_enhanced.txt")

def main():
    """Main setup function"""
    print("Enhanced Podcast Guest Analysis Pipeline Setup")
    print("=" * 60)
    
    check_python_version()
    install_poetry()
    install_dependencies()
    download_spacy_model()
    check_environment_variables()
    create_directories()
    create_sample_config()
    print_usage_instructions()

if __name__ == "__main__":
    main()