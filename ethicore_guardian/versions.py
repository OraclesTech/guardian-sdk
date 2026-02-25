"""
Ethicore Engine™ - Guardian SDK - Version Information
"""

__version__ = "1.0.0"
__version_info__ = tuple(map(int, __version__.split('.')))

# Build information
__build__ = "stable.1"
__release_date__ = "2026-02-25"

# Feature flags
FEATURES = {
    "multi_layer_detection": True,
    "ml_learning": True,
    "openai_support": True,
    "anthropic_support": True,
    "async_support": True,
    "framework_integrations": True,
}

# Model versions
MODEL_VERSIONS = {
    "orchestrator": "3.0.0",
    "pattern_analyzer": "1.0.0", 
    "semantic_analyzer": "1.1.0",
    "behavioral_analyzer": "1.0.0",
    "ml_inference_engine": "3.0.0",
}