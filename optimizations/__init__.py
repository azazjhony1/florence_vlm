"""
Optimization techniques for Florence-2 model
Each module provides a get_model_and_processor() function
"""

from pathlib import Path

OPTIMIZATIONS_DIR = Path(__file__).parent
MODELS_CACHE_DIR = OPTIMIZATIONS_DIR / "cached_models"
MODELS_CACHE_DIR.mkdir(exist_ok=True)