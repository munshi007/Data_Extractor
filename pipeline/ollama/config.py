"""
Ollama Configuration for Testing
"""

import os

# Ollama Server Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODELS_PATH = os.getenv("OLLAMA_MODELS", "/local/home/munshi/TESTING/.ollama/models")

# Model Configuration
MODELS = {
    "tier1_embedder": {
        "model": "nomic-embed-text:v1.5",
        "enabled": True,
        "purpose": "Semantic embeddings for text grouping",
        "description": "Tier 1: Replaces sentence-transformers with faster GPU embeddings"
    },
    "tier2_cleaner": {
        "model": "mistral",
        "enabled": True,
        "purpose": "Clean and normalize extracted text",
        "description": "Tier 2: Fixes OCR errors and improves text quality"
    },
    "tier3_validator": {
        "model": "deepseek-r1:8b",
        "enabled": True,
        "purpose": "Validate region classifications",
        "description": "Tier 3: Uses reasoning to fix classification errors"
    },
    "tier4_post_processor": {
        "model": "mistral",
        "enabled": True,
        "purpose": "Improve markdown output",
        "description": "Tier 4: Final pass for markdown quality"
    }
}

# Performance Settings
EMBEDDING_CACHE_SIZE = 1000  # Cache up to 1000 embeddings
LLM_TIMEOUT = 300  # 5 minute timeout for LLM calls
LLM_MAX_TOKENS = 512  # Max tokens per LLM generation

# Feature Flags
USE_OLLAMA_EMBEDDINGS = True  # Use Ollama instead of sentence-transformers
AGGRESSIVE_TEXT_CLEANING = False  # Use LLM for all text cleaning (slower but better)
VALIDATE_ALL_REGIONS = True  # Validate every region classification
POST_PROCESS_MARKDOWN = True  # Apply final post-processing pass

def print_config():
    """Print current configuration."""
    print("="*60)
    print("OLLAMA INTEGRATION CONFIGURATION")
    print("="*60)
    print(f"Server: {OLLAMA_HOST}")
    print(f"Models Path: {OLLAMA_MODELS_PATH}")
    print("\nEnabled Tiers:")
    for tier, config in MODELS.items():
        if config["enabled"]:
            print(f"  ✓ {tier}: {config['model']} - {config['purpose']}")
    print("="*60)

if __name__ == "__main__":
    print_config()
