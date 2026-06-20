"""
Ethicore Engine™ - Guardian SDK - Version Information
"""

__version__ = "2.7.0"
__version_info__ = tuple(map(int, __version__.split('.')))

# Build information
__build__ = "stable.1"
__release_date__ = "2026-06-19"

# Feature flags
FEATURES = {
    "multi_layer_detection": True,
    "ml_learning": True,
    "openai_support": True,
    "anthropic_support": True,
    "async_support": True,
    "framework_integrations": True,
    "supply_chain_integrity": True,        # v2.6.0: guardian verify + init self-check
    "local_provider_support": True,        # v2.6.1: LM Studio, llama.cpp, LocalAI, Jan.ai providers
    "child_safety_protection": True,       # v2.6.2: childSafetyViolation absolute-block (Matthew 18:6)
    "crescendo_trajectory_detection": True,  # v2.6.3: stateful multi-turn crescendo attack detection (Gap 64)
    "agent_identity_spoofing_detection": True,  # v2.6.3: false orchestrator/agent identity claim detection (Gap 65)
    "deepseek_provider": True,             # v2.6.4: DeepSeek V4 provider (deepseek-v4-flash, deepseek-v4-pro)
    "mistral_provider": True,              # v2.6.4: Mistral AI provider (mistral-large, codestral, devstral)
    "perplexity_provider": True,           # v2.6.4: Perplexity Sonar provider (web-grounded models)
    "context_aware_scanning": True,        # v2.6.5: source_type-aware suppression — no encoding FP on external content
}

# Model versions
MODEL_VERSIONS = {
    "orchestrator": "3.0.0",
    "pattern_analyzer": "1.3.0",   # 7 community categories; +childSafetyViolation (CRITICAL, weight 100)
    "semantic_analyzer": "1.1.0",
    "behavioral_analyzer": "1.1.0",  # v2.6.3: cross-turn crescendo trajectory scoring (Gap 64)
    "ml_inference_engine": "3.2.0", # retrained 125k samples, 94 categories, 1230 fingerprints, 2026-05-18
}