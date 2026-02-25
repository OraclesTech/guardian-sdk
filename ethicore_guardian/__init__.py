"""
Ethicore Engine™ - Guardian SDK - AI Threat Protection
Multi-layer security for AI applications

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

# Version information
__version__ = "1.0.0"
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

# Convenience imports for existing analyzers
try:
    from .analyzers.pattern_analyzer import PatternAnalyzer
    from .analyzers.semantic_analyzer import SemanticAnalyzer
    from .analyzers.behavioral_analyzer import BehavioralAnalyzer
    from .analyzers.ml_inference_engine import MLInferenceEngine
except ImportError as e:
    print(f"⚠️  Some analyzers not available: {e}")

# License validator — stdlib-only, always available
try:
    from .license import LicenseValidator, LicenseInfo, validate_license
except ImportError:
    LicenseValidator = None  # type: ignore[assignment,misc]
    LicenseInfo = None       # type: ignore[assignment,misc]
    validate_license = None  # type: ignore[assignment]

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

    # Analyzers (if available)
    'PatternAnalyzer',
    'SemanticAnalyzer',
    'BehavioralAnalyzer',
    'MLInferenceEngine',

    # License
    'LicenseValidator',
    'LicenseInfo',
    'validate_license',

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
🛡️  Ethicore Engine™ - Guardian SDK v{__version__}
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