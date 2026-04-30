"""
Ethicore Engine™ - Guardian SDK - AI Threat Protection
Multi-layer security for AI applications

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

# Version information
__version__ = "1.9.0"
__author__ = "Oracles Technologies LLC"

# Core exports
from .guardian import (
    Guardian,
    ThreatAnalysis,
    GuardianConfig,
    ThreatChallengeException,
    analyze_text,
    protect_openai,
)

# xAI / Grok provider — always available (depends only on openai package)
try:
    from .providers.xai_provider import (
        XAIProvider,
        ProtectedXAIClient,
        ThreatBlockedException as XAIThreatBlockedException,
        AgentToolBlockedException,
        ToolOutputBlockedException,
        create_protected_xai_client,
    )
    from .guardian import protect_xai  # convenience wrapper registered in guardian.py
except ImportError:
    pass  # openai package not installed

# Azure OpenAI provider — depends only on openai package
try:
    from .providers.azure_provider import (
        AzureOpenAIProvider,
        create_protected_azure_client,
    )
    from .guardian import protect_azure
except ImportError:
    pass

# Google Gemini provider — requires google-genai
try:
    from .providers.gemini_provider import (
        GeminiProvider,
        ProtectedGeminiClient,
        create_protected_gemini_client,
    )
    from .guardian import protect_gemini
except ImportError:
    pass  # google-genai not installed

# AWS Bedrock provider — requires boto3
try:
    from .providers.bedrock_provider import (
        BedrockProvider,
        ProtectedBedrockClient,
        create_protected_bedrock_client,
    )
    from .guardian import protect_bedrock
except ImportError:
    pass  # boto3 not installed

# LiteLLM provider — requires litellm
try:
    from .providers.litellm_provider import (
        LiteLLMProvider,
        ProtectedLiteLLMClient,
        create_protected_litellm,
    )
    from .guardian import protect_litellm
except ImportError:
    pass  # litellm not installed

# Convenience imports for existing analyzers
try:
    from .analyzers.pattern_analyzer import PatternAnalyzer
    from .analyzers.semantic_analyzer import SemanticAnalyzer
    from .analyzers.behavioral_analyzer import BehavioralAnalyzer
    from .analyzers.ml_inference_engine import MLInferenceEngine
except ImportError as e:
    print(f"[WARN]  Some analyzers not available: {e}")

# Phase 3 — output analysis + closed-loop adversarial learning
try:
    from .analyzers.output_analyzer import OutputAnalyzer, OutputAnalysisResult
    from .analyzers.adversarial_learner import AdversarialLearner, LearningOutcome
except ImportError as e:
    print(f"[WARN]  Phase 3 analyzers not available: {e}")

# Multilingual support — Layer 8 (community + API tier)
try:
    from .analyzers.language_detector import LanguageDetector, LanguageDetectionResult
    from .analyzers.multilingual_semantic_analyzer import (
        MultilingualSemanticAnalyzer,
        MultilingualSemanticResult,
        MultilingualMatch,
    )
except ImportError as e:  # pragma: no cover
    print(f"[WARN]  Multilingual analyzers not available: {e}")
    OutputAnalyzer = None        # type: ignore[assignment,misc]
    OutputAnalysisResult = None  # type: ignore[assignment,misc]
    AdversarialLearner = None    # type: ignore[assignment,misc]
    LearningOutcome = None       # type: ignore[assignment,misc]

# Agentic pipeline protection — Phase 4
try:
    from .analyzers.tool_output_scanner import ToolOutputScanner, ToolOutputScanResult
    from .analyzers.tool_call_validator import ToolCallValidator, ToolCallScanResult, ToolCallMatch
except ImportError as e:
    print(f"[WARN]  Agentic analyzers not available: {e}")

# Agentic security extensions — Phase 4b (API tier)
# Tool provenance tracking + response anomaly detection
try:
    from .analyzers.tool_registry import (
        ToolRegistry,
        ToolRegistration,
        ToolProvenanceResult,
    )
    from .analyzers.response_anomaly_detector import (
        ResponseAnomalyDetector,
        ResponseAnomalyResult,
        AnomalySignal,
    )
except ImportError as e:
    print(f"[WARN]  Phase 4b security extensions not available: {e}")

# LangChain callback handlers — optional (requires langchain-core)
try:
    from .providers.langchain_callback import (
        GuardianCallbackHandler,
        GuardianAsyncCallbackHandler,
        GuardianPipelineError,
        GuardianAgentBlockedError,
        GuardianToolCallBlockedError,
        GuardianToolOutputBlockedError,
    )
except ImportError:
    pass  # langchain-core not installed — handlers unavailable

# Main API exports
__all__ = [
    # Core classes
    'Guardian',
    'ThreatAnalysis',
    'GuardianConfig',

    # Exceptions
    'ThreatChallengeException',

    # Convenience functions
    'analyze_text',
    'protect_openai',
    'protect_xai',
    'create_protected_xai_client',
    'protect_azure',
    'create_protected_azure_client',
    'protect_gemini',
    'create_protected_gemini_client',
    'protect_bedrock',
    'create_protected_bedrock_client',
    'protect_litellm',
    'create_protected_litellm',

    # xAI / Grok provider
    'XAIProvider',
    'ProtectedXAIClient',
    'AgentToolBlockedException',
    'ToolOutputBlockedException',

    # Azure OpenAI provider
    'AzureOpenAIProvider',

    # Google Gemini provider
    'GeminiProvider',
    'ProtectedGeminiClient',

    # AWS Bedrock provider
    'BedrockProvider',
    'ProtectedBedrockClient',

    # LiteLLM provider
    'LiteLLMProvider',
    'ProtectedLiteLLMClient',

    # Analyzers (if available)
    'PatternAnalyzer',
    'SemanticAnalyzer',
    'BehavioralAnalyzer',
    'MLInferenceEngine',

    # Phase 3 — output analysis + adversarial learning
    'OutputAnalyzer',
    'OutputAnalysisResult',
    'AdversarialLearner',
    'LearningOutcome',

    # Multilingual — Layer 8
    'LanguageDetector',
    'LanguageDetectionResult',
    'MultilingualSemanticAnalyzer',
    'MultilingualSemanticResult',
    'MultilingualMatch',

    # Agentic pipeline protection — Phase 4
    'ToolOutputScanner',
    'ToolOutputScanResult',
    'ToolCallValidator',
    'ToolCallScanResult',
    'ToolCallMatch',

    # Agentic security extensions — Phase 4b (API tier)
    'ToolRegistry',
    'ToolRegistration',
    'ToolProvenanceResult',
    'ResponseAnomalyDetector',
    'ResponseAnomalyResult',
    'AnomalySignal',

    # LangChain integration (optional)
    'GuardianCallbackHandler',
    'GuardianAsyncCallbackHandler',
    'GuardianPipelineError',
    'GuardianAgentBlockedError',
    'GuardianToolCallBlockedError',
    'GuardianToolOutputBlockedError',

    # Version
    '__version__',
]

# Package metadata
__description__ = "AI Threat Protection SDK - Multi-layer security for AI applications"
__url__ = "https://oraclestechnologies.com/guardian"

def _print_welcome():
    """Print welcome message for interactive use"""
    try:
        import sys
        if hasattr(sys, 'ps1'):  # Interactive Python
            print(f"""
[Guardian]  Ethicore Engine™ - Guardian SDK v{__version__}
   AI Threat Protection Ready

Quick Start:
   from ethicore_guardian import Guardian
   import openai

   guardian = Guardian(api_key='your_key')
   protected_client = guardian.wrap(openai.OpenAI())

   # Your existing multi-layer protection is now active!
""")
    except Exception as _welcome_err:  # noqa: BLE001
        # Non-critical display failure — log at DEBUG so production logs stay
        # clean while the issue remains visible during development.
        # Principle 11 (Sacred Truth): we never silently discard errors.
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "Guardian welcome message could not be displayed: %s", _welcome_err
        )

# Print welcome for interactive use
_print_welcome()