"""
Ethicore Engine™ - Guardian SDK - Version Information
"""

__version__ = "2.6.2"
__version_info__ = tuple(map(int, __version__.split('.')))

# Build information
__build__ = "stable.1"
__release_date__ = "2026-05-20"

# Feature flags
FEATURES = {
    "multi_layer_detection": True,
    "ml_learning": True,
    "openai_support": True,
    "anthropic_support": True,
    "async_support": True,
    "framework_integrations": True,
    "supply_chain_integrity": True,   # v2.6.0: guardian verify + init self-check
    "local_provider_support": True,   # v2.6.1: LM Studio, llama.cpp, LocalAI, Jan.ai providers
    "child_safety_protection": True,  # v2.6.2: childSafetyViolation absolute-block (Matthew 18:6)
}

# Model versions
MODEL_VERSIONS = {
    "orchestrator": "3.0.0",
    "pattern_analyzer": "1.3.0",   # 7 community categories; +childSafetyViolation (CRITICAL, weight 100)
    "semantic_analyzer": "1.1.0",
    "behavioral_analyzer": "1.0.0",
    "ml_inference_engine": "3.2.0", # retrained 125k samples, 94 categories, 1230 fingerprints, 2026-05-18
}