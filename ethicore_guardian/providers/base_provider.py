"""
Ethicore Engine™ - Guardian SDK - Base Provider & Configuration
Core abstractions for AI provider integrations
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class GuardianError(Exception):
    """Base Guardian exception"""
    pass


class ConfigurationError(GuardianError):
    """Configuration-related errors"""
    pass


class AuthenticationError(GuardianError):
    """API key authentication errors"""
    pass


class AnalysisError(GuardianError):
    """Threat analysis errors"""
    pass


class RateLimitError(GuardianError):
    """Rate limiting errors"""
    pass


class ModelLoadError(GuardianError):
    """Model loading/initialization errors"""
    pass


class ProviderError(GuardianError):
    """AI provider integration errors"""
    pass


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class GuardianConfig:
    """Guardian configuration object"""
    
    # Core settings
    api_key: Optional[str] = None
    enabled: bool = True
    strict_mode: bool = False
    
    # Sensitivity levels (0.0 to 1.0)
    pattern_sensitivity: float = 0.8
    semantic_sensitivity: float = 0.7
    ml_sensitivity: float = 0.75
    
    # Performance settings
    max_latency_ms: int = 50
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    
    # Logging and metrics
    log_level: str = "INFO"
    enable_metrics: bool = True
    send_telemetry: bool = False
    
    # ML model settings
    ml_model: str = "auto"  # auto, distilbert, roberta, etc.
    ml_learning_enabled: bool = True
    
    # Custom rules
    custom_patterns: Optional[List[Dict]] = None
    allowlist_rules: Optional[List[str]] = None
    
    # Provider-specific settings
    provider_configs: Optional[Dict[str, Dict]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Ensure sensitivity values are in valid range
        for attr in ['pattern_sensitivity', 'semantic_sensitivity', 'ml_sensitivity']:
            value = getattr(self, attr)
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{attr} must be between 0.0 and 1.0, got {value}")
        
        # Ensure max_latency_ms is positive
        if self.max_latency_ms <= 0:
            raise ConfigurationError(f"max_latency_ms must be positive, got {self.max_latency_ms}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of {valid_log_levels}, got {self.log_level}")
        
        # Initialize empty lists if None
        if self.custom_patterns is None:
            self.custom_patterns = []
        if self.allowlist_rules is None:
            self.allowlist_rules = []
        if self.provider_configs is None:
            self.provider_configs = {}


def load_config(config_file: Optional[str] = None, **kwargs) -> GuardianConfig:
    """
    Load Guardian configuration from file or environment
    
    Args:
        config_file: Path to configuration file (JSON/YAML)
        **kwargs: Override configuration values
        
    Returns:
        GuardianConfig instance
    """
    config_data = {}
    
    # Load from file if provided
    if config_file:
        try:
            import json
            from pathlib import Path
            
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.suffix.lower() == '.json':
                        config_data = json.load(f)
                    elif config_path.suffix.lower() in ['.yml', '.yaml']:
                        import yaml
                        config_data = yaml.safe_load(f)
                    else:
                        raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")
            else:
                raise ConfigurationError(f"Configuration file not found: {config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file {config_file}: {e}")
    
    # Apply overrides
    config_data.update(kwargs)
    
    # Create config object
    return GuardianConfig(**config_data)


# ==============================================================================
# BASE PROVIDER INTERFACE
# ==============================================================================

class BaseProvider(ABC):
    """
    Abstract base class for AI provider integrations
    
    Each AI provider (OpenAI, Anthropic, Google, etc.) implements this interface
    to provide consistent threat protection across all providers.
    """
    
    def __init__(self, guardian_instance):
        """
        Initialize provider with Guardian instance
        
        Args:
            guardian_instance: Guardian instance that owns this provider
        """
        self.guardian = guardian_instance
        self.provider_name = "base"
        self.logger = logging.getLogger(f"guardian.providers.{self.provider_name}")
    
    @abstractmethod
    def wrap_client(self, client: Any) -> Any:
        """
        Wrap AI provider client with Guardian protection
        
        Args:
            client: Original AI provider client
            
        Returns:
            Protected client that maintains API compatibility
        """
        pass
    
    @abstractmethod
    def extract_prompt(self, *args, **kwargs) -> str:
        """
        Extract prompt text from API call arguments
        
        Args:
            *args, **kwargs: API call arguments
            
        Returns:
            Extracted prompt text for threat analysis
        """
        pass
    
    def validate_response(self, response: Any) -> bool:
        """
        Validate AI response for policy violations (optional)
        
        Args:
            response: AI provider response object
            
        Returns:
            True if response is acceptable, False if it violates policies
        """
        # Default implementation allows all responses
        return True
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider
        
        Returns:
            Dictionary with provider metadata
        """
        return {
            'name': self.provider_name,
            'version': getattr(self, 'version', '1.0.0'),
            'supported_methods': getattr(self, 'supported_methods', []),
            'configuration': getattr(self, 'configuration', {})
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Exception:
        """
        Handle and potentially transform provider-specific errors
        
        Args:
            error: Original exception
            context: Additional error context
            
        Returns:
            Processed exception (may be transformed)
        """
        # Default implementation returns error as-is
        return error


# ==============================================================================
# THREAT ANALYSIS RESULT TYPES
# ==============================================================================

@dataclass
class LayerResult:
    """Result from a single analysis layer"""
    layer_name: str
    verdict: str  # BLOCK, SUSPICIOUS, ALLOW
    confidence: float  # 0.0 to 1.0
    score: float
    details: Dict[str, Any]
    analysis_time_ms: float


@dataclass
class ThreatDetectionResult:
    """Complete threat detection result from orchestrator"""
    verdict: str  # BLOCK, CHALLENGE, ALLOW
    threat_level: str  # NONE, LOW, MEDIUM, HIGH, CRITICAL
    overall_score: float
    confidence: float
    layer_results: List[LayerResult]
    threats_detected: List[Dict[str, Any]]
    reasoning: List[str]
    analysis_time_ms: float
    metadata: Dict[str, Any]


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_provider_for_client(client: Any) -> str:
    """
    Auto-detect AI provider from client object
    
    Args:
        client: AI provider client instance
        
    Returns:
        Provider name string
    """
    client_type = str(type(client)).lower()
    client_module = getattr(client, '__module__', '').lower()
    
    # Check client type and module for provider indicators
    if 'openai' in client_type or 'openai' in client_module:
        return 'openai'
    elif 'anthropic' in client_type or 'anthropic' in client_module:
        return 'anthropic'
    elif 'azure' in client_type or 'azure' in client_module:
        return 'azure'
    elif 'google' in client_type or 'google' in client_module:
        return 'google'
    elif 'cohere' in client_type or 'cohere' in client_module:
        return 'cohere'
    else:
        raise ConfigurationError(f"Unknown provider for client: {type(client)}")


def normalize_threat_level(score: float, scale: str = "0-10") -> str:
    """
    Normalize threat score to standard threat level
    
    Args:
        score: Threat score in original scale
        scale: Original scale format ("0-1", "0-10", "0-100")
        
    Returns:
        Standard threat level string
    """
    # Normalize to 0-1 scale
    if scale == "0-1":
        normalized = score
    elif scale == "0-10":
        normalized = score / 10.0
    elif scale == "0-100":
        normalized = score / 100.0
    else:
        raise ValueError(f"Unsupported scale: {scale}")
    
    # Convert to threat level
    if normalized >= 0.9:
        return "CRITICAL"
    elif normalized >= 0.7:
        return "HIGH"
    elif normalized >= 0.5:
        return "MEDIUM"
    elif normalized > 0.0:
        return "LOW"
    else:
        return "NONE"


def create_analysis_context(api_call: str, **kwargs) -> Dict[str, Any]:
    """
    Create analysis context from API call information
    
    Args:
        api_call: Name of the API call being made
        **kwargs: Additional context parameters
        
    Returns:
        Context dictionary for threat analysis
    """
    context = {
        'api_call': api_call,
        'timestamp': kwargs.get('timestamp'),
        'user_id': kwargs.get('user_id'),
        'session_id': kwargs.get('session_id'),
        'ip_address': kwargs.get('ip_address'),
        'model': kwargs.get('model'),
        'max_tokens': kwargs.get('max_tokens'),
        'temperature': kwargs.get('temperature'),
    }
    
    # Remove None values
    return {k: v for k, v in context.items() if v is not None}


# ==============================================================================
# VERSION INFORMATION
# ==============================================================================

__version__ = "1.0.0"
__author__ = "Oracles Technologies LLC"
__license__ = "Proprietary"

# Export main classes
__all__ = [
    'BaseProvider',
    'GuardianConfig', 
    'load_config',
    'GuardianError',
    'ConfigurationError',
    'AuthenticationError',
    'AnalysisError',
    'RateLimitError',
    'ModelLoadError',
    'ProviderError',
    'LayerResult',
    'ThreatDetectionResult',
    'get_provider_for_client',
    'normalize_threat_level',
    'create_analysis_context'
]