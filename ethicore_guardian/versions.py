"""
Ethicore Engine™ - Guardian SDK - Version Information
"""

__version__ = "2.5.0"
__version_info__ = tuple(map(int, __version__.split('.')))

# Build information
__build__ = "stable.1"
__release_date__ = "2026-05-16"

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
    "pattern_analyzer": "1.1.0",   # 85 categories (toolChainEscalation, mcpSchemaInjection, posturalManipulation)
    "semantic_analyzer": "1.1.0",
    "behavioral_analyzer": "1.0.0",
    "ml_inference_engine": "3.1.0", # retrained 90k samples, 85 categories, 2026-05-16
}