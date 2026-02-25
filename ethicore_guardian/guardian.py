"""
Ethicore Engine™ - Guardian SDK - Main Guardian Class
Fixed version that properly handles provider imports
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

import asyncio
import hashlib
import hmac
import logging
import threading
import time as _time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import os
from pathlib import Path
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Audit logger — Principle 13 (Ultimate Accountability).
# Imported defensively so a missing dependency never breaks Guardian startup.
try:
    from .audit import get_default_logger as _get_audit_logger
except Exception as _audit_import_err:  # noqa: BLE001
    logger.debug("Audit logger unavailable: %s", _audit_import_err)
    _get_audit_logger = None  # type: ignore[assignment]


@dataclass
class ThreatAnalysis:
    """Public threat analysis result"""
    is_safe: bool
    threat_score: float  # 0.0 to 1.0
    threat_level: str  # NONE, LOW, MEDIUM, HIGH, CRITICAL
    threat_types: List[str]
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    recommended_action: str  # ALLOW, CHALLENGE, BLOCK
    analysis_time_ms: int
    layer_votes: Dict[str, str]
    metadata: Dict[str, Any]


class ThreatChallengeException(Exception):
    """
    Raised when Guardian recommends CHALLENGE but not an outright BLOCK.

    In non-strict mode, callers should surface a secondary verification step
    (e.g. CAPTCHA, human review) rather than hard-blocking the request.
    Distinct from ``ThreatBlockedException`` so callers can handle each case
    separately.

    Principle 16 (Sacred Autonomy): preserves the human's ability to choose
    by surfacing uncertainty rather than silently blocking.

    Attributes:
        analysis: The ``ThreatAnalysis`` result that triggered the challenge.
    """

    def __init__(self, message: str, analysis: "ThreatAnalysis") -> None:
        super().__init__(message)
        self.analysis = analysis


@dataclass
class GuardianConfig:
    """Guardian configuration"""
    api_key: Optional[str] = None
    enabled: bool = True
    strict_mode: bool = False
    pattern_sensitivity: float = 0.8
    semantic_sensitivity: float = 0.7
    ml_sensitivity: float = 0.75
    max_latency_ms: int = 50
    log_level: str = "INFO"
    enable_metrics: bool = True
    # Principle 12 (Sacred Privacy) + Principle 14 (Divine Safety):
    # Cap unbounded input to protect memory and prevent payload-flooding attacks.
    # 32 768 chars (~8 K tokens) is generous for real prompts but blocks abuse.
    # Set to 0 to disable enforcement (not recommended in production).
    max_input_length: int = 32_768

    # Principle 14 (Divine Safety): fail-closed on pipeline stall.
    # If analysis exceeds this budget the call returns CHALLENGE, not ALLOW.
    # 0 = no timeout (not recommended for production).
    analysis_timeout_ms: int = 5_000

    # Principle 14 (Divine Safety): protect backend from flooding.
    # Maximum analyze() calls per minute for this Guardian instance.
    # 0 = unlimited.
    max_requests_per_minute: int = 0

    # Caching — Principle 12 (Sacred Privacy) + performance.
    # Results are keyed by SHA-256(normalised_text|source_type); raw text is
    # never stored.  Cache is bypassed when session_id is in context so that
    # multi-turn context trackers receive every turn.
    # Override via env vars: ETHICORE_CACHE_ENABLED / ETHICORE_CACHE_TTL / ETHICORE_CACHE_MAX_MB
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300      # 5 minutes
    cache_max_size_mb: int = 256

    # Learning system access control — Principle 12 (Sacred Privacy) / Principle 14.
    # correction_key = None means corrections are DISABLED (no key → no write access).
    # Set via ETHICORE_CORRECTION_KEY env var.  Never log or echo this value.
    correction_key: Optional[str] = None
    correction_rate_limit_per_minute: int = 10

    # Paid asset license key — enables licensed threat library (30 categories).
    # Override via env var: ETHICORE_LICENSE_KEY (never log this value).
    license_key: Optional[str] = None

    # Path to extracted paid asset bundle directory.
    # Override via env var: ETHICORE_ASSETS_DIR
    assets_dir: Optional[str] = None


class _CorrectionRateLimiter:
    """
    Token-bucket rate limiter for the learning correction API.

    Pure stdlib — no external dependencies.  Thread-safe.

    Principle 12 (Sacred Privacy) + Principle 14 (Divine Safety): limits how
    fast external callers can write to the learning corpus, preventing flooding
    attacks that could corrupt ML behaviour.
    """

    def __init__(self, rate_per_minute: int) -> None:
        self._capacity = max(1, rate_per_minute)
        self._tokens = float(self._capacity)
        self._last_refill = _time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Consume one token.  Returns True if allowed, False if rate-limited."""
        with self._lock:
            now = _time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * (self._capacity / 60.0),
            )
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class Guardian:
    """
    Guardian SDK - Main Interface
    
    Provides AI threat protection through your existing multi-layer analysis
    
    Usage:
        guardian = Guardian(api_key='your_key')
        protected_openai = guardian.wrap(openai.OpenAI())
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 config: Optional[Union[GuardianConfig, Dict]] = None,
                 **kwargs):
        """
        Initialize Guardian with configuration
        """
        
        # Load configuration
        self.config = self._load_config(api_key, config, **kwargs)

        # Core components
        self.threat_detector = None
        self.providers: Dict[str, Any] = {}

        # State
        self.initialized = False
        self.models_dir = Path(__file__).parent.parent / "models"

        # Rate limiter — Principle 14 (Divine Safety)
        # Lazily instantiated on first analyze() call so the import error for
        # asyncio-throttle (if missing) never breaks Guardian.__init__.
        self._throttler: Any = None
        self._throttler_rpm: int = 0  # tracks the last configured RPM

        # diskcache — Principle 12 (Sacred Privacy): keys are SHA-256 hashes,
        # raw text is never stored.  Lazily opened on first analyze() call.
        self._cache: Any = None  # diskcache.Cache instance when enabled

        # Learning system access control — Principle 12 + 14.
        self._correction_limiter = _CorrectionRateLimiter(
            self.config.correction_rate_limit_per_minute
        )
        
        print(f"🛡️  Guardian SDK initialized")
        print(f"   API Key: {'✅' if self.config.api_key else '❌'} Set")
        print(f"   Enabled: {self.config.enabled}")
        
        # Initialize providers immediately to catch issues early
        self._setup_providers()
    
    def _load_config(self, api_key: Optional[str], config: Optional[Union[GuardianConfig, Dict]], 
                    **kwargs) -> GuardianConfig:
        """Load and validate configuration"""
        
        if isinstance(config, GuardianConfig):
            guardian_config = config
        elif isinstance(config, dict):
            guardian_config = GuardianConfig(**config)
        else:
            guardian_config = GuardianConfig()
        
        # Override with provided values
        if api_key:
            guardian_config.api_key = api_key
        
        # Apply kwargs
        for key, value in kwargs.items():
            if hasattr(guardian_config, key):
                setattr(guardian_config, key, value)
        
        # Try environment variable if no API key
        if not guardian_config.api_key:
            guardian_config.api_key = os.getenv('ETHICORE_API_KEY')

        # License / asset bundle — Principle 15 (Blessed Stewardship)
        if not guardian_config.license_key:
            guardian_config.license_key = os.getenv('ETHICORE_LICENSE_KEY')
        if not guardian_config.assets_dir:
            guardian_config.assets_dir = os.getenv('ETHICORE_ASSETS_DIR')

        return guardian_config
    
    def _setup_providers(self):
        """Setup available AI provider integrations with better error handling"""
        print("🔧 Setting up providers...")
        
        try:
            from .providers.guardian_ollama_provider import OllamaProvider
            self.providers['ollama'] = OllamaProvider(self)
            print("✅ Ollama provider registered")
        except ImportError:
            print("⚠️  Ollama provider not found")

        # Try to import and register OpenAI provider
        try:
            # First check if we're running from the right directory
            current_dir = Path(__file__).parent
            providers_dir = current_dir / "providers"
            
            if not providers_dir.exists():
                print(f"⚠️  Providers directory not found at: {providers_dir}")
                print("   Creating providers directory structure...")
                providers_dir.mkdir(exist_ok=True)
                
                # Create __init__.py if it doesn't exist
                init_file = providers_dir / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# Providers module\n")
                    print("   Created providers/__init__.py")
            
            # Try absolute import first
            try:
                from providers.openai_provider import OpenAIProvider
                self.providers['openai'] = OpenAIProvider(self)
                print("✅ OpenAI provider registered (absolute import)")
                return
            except ImportError as e1:
                print(f"   Absolute import failed: {e1}")
                
                # Try relative import
                try:
                    from .providers.openai_provider import OpenAIProvider
                    self.providers['openai'] = OpenAIProvider(self)
                    print("✅ OpenAI provider registered (relative import)")
                    return
                except ImportError as e2:
                    print(f"   Relative import failed: {e2}")
                    
                    # Try direct module loading
                    try:
                        import importlib.util
                        openai_provider_path = current_dir / "providers" / "openai_provider.py"
                        
                        if openai_provider_path.exists():
                            spec = importlib.util.spec_from_file_location("openai_provider", openai_provider_path)
                            openai_provider_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(openai_provider_module)
                            
                            OpenAIProvider = openai_provider_module.OpenAIProvider
                            self.providers['openai'] = OpenAIProvider(self)
                            print("✅ OpenAI provider registered (direct file loading)")
                            return
                        else:
                            print(f"   OpenAI provider file not found at: {openai_provider_path}")
                    except Exception as e3:
                        print(f"   Direct loading failed: {e3}")
                        
                        # Last resort: create a minimal provider
                        self._create_minimal_openai_provider()
                        return
        
        except Exception as e:
            print(f"⚠️  OpenAI provider setup failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Current directory: {Path(__file__).parent}")

            # Create minimal provider as fallback
            self._create_minimal_openai_provider()

        # Try to import and register Anthropic provider
        try:
            from .providers.anthropic_provider import AnthropicProvider
            self.providers['anthropic'] = AnthropicProvider(self)
            print("✅ Anthropic provider registered")
        except ImportError:
            print("⚠️  Anthropic provider not available (pip install anthropic)")
    
    def _create_minimal_openai_provider(self):
        """Create a minimal OpenAI provider as fallback"""
        print("🔧 Creating minimal OpenAI provider...")
        
        class MinimalOpenAIProvider:
            def __init__(self, guardian):
                self.guardian = guardian
                self.provider_name = "openai"
            
            def wrap_client(self, client):
                """Wrap OpenAI client with basic protection"""
                return MinimalProtectedOpenAIClient(client, self.guardian)
            
            def extract_prompt(self, *args, **kwargs):
                """Extract prompt from OpenAI API arguments"""
                # Chat completions format
                if 'messages' in kwargs:
                    messages = kwargs['messages']
                    if messages and isinstance(messages, list):
                        # Get last user message
                        user_messages = [msg for msg in messages if msg.get('role') == 'user']
                        if user_messages:
                            content = user_messages[-1].get('content', '')
                            return str(content)
                
                # Legacy completions format
                elif 'prompt' in kwargs:
                    return str(kwargs['prompt'])
                
                return ""
        
        class MinimalProtectedOpenAIClient:
            def __init__(self, original_client, guardian):
                self._original_client = original_client
                self._guardian = guardian
                
                # Preserve original attributes
                for attr_name in dir(original_client):
                    if not attr_name.startswith('_') and not callable(getattr(original_client, attr_name)):
                        setattr(self, attr_name, getattr(original_client, attr_name))
                
                # Wrap chat interface
                if hasattr(original_client, 'chat'):
                    self.chat = self._create_protected_chat()
            
            def _create_protected_chat(self):
                """Create protected chat interface"""
                class ProtectedChat:
                    def __init__(self, original_chat, guardian):
                        self._original_chat = original_chat
                        self._guardian = guardian
                        
                        if hasattr(original_chat, 'completions'):
                            self.completions = self._create_protected_completions()
                    
                    def _create_protected_completions(self):
                        class ProtectedCompletions:
                            def __init__(self, original_completions, guardian):
                                self._original_completions = original_completions
                                self._guardian = guardian
                            
                            def create(self, **kwargs):
                                """Protected sync create method"""
                                return self._protect_and_call(self._original_completions.create, **kwargs)
                            
                            async def acreate(self, **kwargs):
                                """Protected async create method"""  
                                return await self._protect_and_call_async(self._original_completions.acreate, **kwargs)
                            
                            def _protect_and_call(self, original_method, **kwargs):
                                """Protect sync call"""
                                # Extract and analyze prompt
                                prompt = self._extract_prompt(**kwargs)
                                
                                if prompt:
                                    # Run analysis
                                    analysis = asyncio.run(self._guardian.analyze(prompt))
                                    
                                    if not analysis.is_safe:
                                        error_msg = f"Request blocked: {analysis.threat_level} threat detected. Reasons: {', '.join(analysis.reasoning[:2])}"
                                        print(f"🚨 BLOCKED: {error_msg}")
                                        raise Exception(error_msg)
                                
                                return original_method(**kwargs)
                            
                            async def _protect_and_call_async(self, original_method, **kwargs):
                                """Protect async call"""
                                # Extract and analyze prompt
                                prompt = self._extract_prompt(**kwargs)
                                
                                if prompt:
                                    analysis = await self._guardian.analyze(prompt)
                                    
                                    if not analysis.is_safe:
                                        error_msg = f"Request blocked: {analysis.threat_level} threat detected. Reasons: {', '.join(analysis.reasoning[:2])}"
                                        print(f"🚨 BLOCKED: {error_msg}")
                                        raise Exception(error_msg)
                                
                                return await original_method(**kwargs)
                            
                            def _extract_prompt(self, **kwargs):
                                """Extract prompt from kwargs"""
                                if 'messages' in kwargs:
                                    messages = kwargs['messages']
                                    if messages:
                                        user_messages = [msg for msg in messages if msg.get('role') == 'user']
                                        if user_messages:
                                            return str(user_messages[-1].get('content', ''))
                                return ""
                        
                        return ProtectedCompletions(self._original_chat.completions, self._guardian)
                
                return ProtectedChat(self._original_client.chat, self._guardian)
            
            def __getattr__(self, name):
                """Delegate to original client"""
                return getattr(self._original_client, name)
        
        # Register the minimal provider
        self.providers['openai'] = MinimalOpenAIProvider(self)
        print("✅ Minimal OpenAI provider created and registered")
    
    async def _ensure_initialized(self):
        """Ensure the threat detection system is initialized"""
        if not self.initialized:
            await self.initialize()
    
    def _create_simple_orchestrator(self):
        """Create a simple orchestrator using your existing analyzers"""
        
        class SimpleOrchestrator:
            def __init__(self, models_dir, license_key=None, assets_dir=None):
                self.models_dir = models_dir
                self.license_key = license_key
                self.assets_dir = assets_dir
                self.analyzers = {}
                self.initialized = False
                
            async def initialize(self):
                """Initialize all available analyzers"""
                try:
                    # Try to import your existing analyzers
                    
                    # Pattern Analyzer
                    try:
                        from .analyzers.pattern_analyzer import PatternAnalyzer
                        self.analyzers['pattern'] = PatternAnalyzer(
                            license_key=self.license_key,
                            assets_dir=self.assets_dir,
                        )
                        print("   ✅ Pattern Analyzer loaded")
                    except ImportError as e:
                        print(f"   ⚠️  Pattern Analyzer not found: {e}")

                    # Semantic Analyzer
                    try:
                        from .analyzers.semantic_analyzer import SemanticAnalyzer
                        semantic = SemanticAnalyzer(
                            license_key=self.license_key,
                            assets_dir=self.assets_dir,
                        )
                        await semantic.initialize()
                        self.analyzers['semantic'] = semantic
                        print("   ✅ Semantic Analyzer loaded")
                    except ImportError as e:
                        print(f"   ⚠️  Semantic Analyzer not found: {e}")
                    except Exception as e:
                        print(f"   ⚠️  Semantic Analyzer failed to initialize: {e}")
                    
                    # Behavioral Analyzer
                    try:
                        from .analyzers.behavioral_analyzer import BehavioralAnalyzer
                        behavioral = BehavioralAnalyzer()
                        behavioral.initialize()
                        self.analyzers['behavioral'] = behavioral
                        print("   ✅ Behavioral Analyzer loaded")
                    except ImportError as e:
                        print(f"   ⚠️  Behavioral Analyzer not found: {e}")
                    except Exception as e:
                        print(f"   ⚠️  Behavioral Analyzer failed to initialize: {e}")
                    
                    # ML Inference Engine
                    try:
                        from .analyzers.ml_inference_engine import MLInferenceEngine
                        ml_engine = MLInferenceEngine()
                        ml_engine.initialize()
                        self.analyzers['ml'] = ml_engine
                        print("   ✅ ML Inference Engine loaded")
                    except ImportError as e:
                        print(f"   ⚠️  ML Inference Engine not found: {e}")
                    except Exception as e:
                        print(f"   ⚠️  ML Inference Engine failed to initialize: {e}")
                    
                    print(f"   📊 Loaded {len(self.analyzers)} analyzers")
                    self.initialized = True
                    return True
                    
                except Exception as e:
                    print(f"   ❌ Orchestrator initialization failed: {e}")
                    return False
            
            async def analyze(self, text: str, metadata: Dict = None):
                """Analyze text using available analyzers"""
                if not text or len(text.strip()) == 0:
                    return self._empty_result("Empty input")
                
                import time
                start_time = time.time()
                
                results = {}
                threats = []
                reasoning = []
                overall_score = 0.0
                
                # Run pattern analysis
                if 'pattern' in self.analyzers:
                    try:
                        pattern_result = self.analyzers['pattern'].analyze(text)
                        results['pattern'] = pattern_result
                        
                        if pattern_result.threat_level in ['HIGH', 'CRITICAL']:
                            overall_score += 30
                            threats.extend(pattern_result.matched_categories)
                            reasoning.append(f"Pattern match: {pattern_result.threat_level}")
                        elif pattern_result.threat_level == 'MEDIUM':
                            overall_score += 15
                            
                    except Exception as e:
                        print(f"Pattern analysis error: {e}")
                
                # Run semantic analysis
                if 'semantic' in self.analyzers:
                    try:
                        semantic_result = await self.analyzers['semantic'].analyze(text)
                        results['semantic'] = semantic_result
                        
                        if semantic_result.semantic_score >= 60:
                            overall_score += 25
                            threats.append("semantic_threat")
                            reasoning.append(f"Semantic threat detected ({semantic_result.semantic_score:.0f}%)")
                        elif semantic_result.semantic_score >= 30:
                            overall_score += 10
                            
                    except Exception as e:
                        print(f"Semantic analysis error: {e}")
                
                # Run behavioral analysis
                if 'behavioral' in self.analyzers:
                    try:
                        behavioral_result = self.analyzers['behavioral'].analyze(text, metadata or {})
                        results['behavioral'] = behavioral_result
                        
                        if behavioral_result.anomaly_score >= 60:
                            overall_score += 20
                            threats.append("behavioral_anomaly")
                            reasoning.append(f"Behavioral anomaly detected ({behavioral_result.anomaly_score:.0f}%)")
                        elif behavioral_result.anomaly_score >= 30:
                            overall_score += 8
                            
                    except Exception as e:
                        print(f"Behavioral analysis error: {e}")
                
                # Run ML analysis
                if 'ml' in self.analyzers:
                    try:
                        ml_result = self.analyzers['ml'].analyze(text)
                        results['ml'] = ml_result
                        
                        if ml_result.threat_probability >= 0.7:
                            overall_score += 35
                            threats.append("ml_threat")
                            reasoning.append(f"ML detected threat ({ml_result.threat_probability*100:.0f}%)")
                        elif ml_result.threat_probability >= 0.5:
                            overall_score += 15
                            
                    except Exception as e:
                        print(f"ML analysis error: {e}")
                
                # Calculate final verdict
                analysis_time = (time.time() - start_time) * 1000
                
                if overall_score >= 50:
                    verdict = 'BLOCK'
                    threat_level = 'CRITICAL' if overall_score >= 70 else 'HIGH'
                elif overall_score >= 20:
                    verdict = 'CHALLENGE' 
                    threat_level = 'MEDIUM'
                else:
                    verdict = 'ALLOW'
                    threat_level = 'LOW' if overall_score > 0 else 'NONE'
                
                return type('Result', (), {
                    'verdict': verdict,
                    'threat_level': threat_level,
                    'overall_score': overall_score,
                    'confidence': min(1.0, len(results) / 4.0),  # Based on analyzer coverage
                    'threats_detected': [{'category': t} for t in threats],
                    'reasoning': reasoning or ["No threats detected"],
                    'analysis_time_ms': analysis_time,
                    'layer_votes': [type('Vote', (), {
                        'layer': name,
                        'vote': 'BLOCK' if 'threat' in str(result).lower() else 'ALLOW'
                    })() for name, result in results.items()],
                    'metadata': {'analyzers_used': list(results.keys())}
                })()
            
            def _empty_result(self, reason):
                """Empty result for invalid input"""
                return type('Result', (), {
                    'verdict': 'ALLOW',
                    'threat_level': 'NONE',
                    'overall_score': 0.0,
                    'confidence': 1.0,
                    'threats_detected': [],
                    'reasoning': [reason],
                    'analysis_time_ms': 0,
                    'layer_votes': [],
                    'metadata': {'reason': reason}
                })()
            
            def get_statistics(self):
                """Get statistics"""
                return {
                    'guardian_version': "1.0.0",
                    'initialized': self.initialized,
                    'active_layers': list(self.analyzers.keys()),
                    'total_analyses': getattr(self, 'total_analyses', 0)
                }
        
        return SimpleOrchestrator(
            self.models_dir,
            license_key=self.config.license_key,
            assets_dir=self.config.assets_dir,
        )
    
    async def initialize(self) -> bool:
        """Initialize the threat detection system"""
        if self.initialized:
            return True
        
        try:
            print("🛡️  Initializing Guardian threat detection...")
            
            # Create simple orchestrator using your existing analyzers
            self.threat_detector = self._create_simple_orchestrator()
            
            # Initialize it
            success = await self.threat_detector.initialize()
            
            if not success:
                print("⚠️  Threat detection initialization had issues, but continuing...")
            
            self.initialized = True
            print("✅ Guardian initialization complete")
            
            return True
            
        except Exception as e:
            print(f"❌ Guardian initialization failed: {e}")
            # Don't raise exception - allow fallback mode
            self.initialized = True
            print("✅ Guardian running in fallback mode")
            return True
    
    def wrap(self, client: Any, provider: Optional[str] = None) -> Any:
        """
        Wrap an AI provider client with Guardian protection
        """
        if not self.config.enabled:
            print("🛡️  Guardian disabled - returning unwrapped client")
            return client
        
        # Auto-detect provider if not specified
        if not provider:
            provider = self._detect_provider(client)
        
        print(f"🔧 Looking for provider: {provider}")
        print(f"   Available providers: {list(self.providers.keys())}")
        
        if provider not in self.providers:
            raise Exception(f"Provider '{provider}' not supported or not available. Available: {list(self.providers.keys())}")
        
        # Get provider wrapper
        provider_wrapper = self.providers[provider]
        
        # Create protected client
        protected_client = provider_wrapper.wrap_client(client)
        
        print(f"🛡️  {provider} client wrapped with Guardian protection")
        return protected_client
    
    def _detect_provider(self, client: Any) -> str:
        """Auto-detect AI provider from client object"""
        client_type = str(type(client)).lower()
        client_module = getattr(client, '__module__', '').lower()
        
        print(f"🔍 Detecting provider for client: {type(client)}")
        print(f"   Client type: {client_type}")
        print(f"   Client module: {client_module}")
        
        if 'openai' in client_type or 'openai' in client_module:
            return 'openai'
        elif 'anthropic' in client_type or 'anthropic' in client_module:
            return 'anthropic'
        elif 'ollama' in client_type or client_type == 'str':
            return 'ollama' 
        else:
            raise Exception(f"Could not detect provider for client: {type(client)}")
    
    async def analyze(self, 
                     text: str, 
                     context: Optional[Dict] = None) -> ThreatAnalysis:
        """
        Analyze text for threats using your existing analyzers
        """
        await self._ensure_initialized()

        if not text or len(text.strip()) == 0:
            return self._empty_analysis("Empty input")

        # ---------------------------------------------------------------
        # Input length guard — Principle 12 (Sacred Privacy) / Principle 14
        # (Divine Safety).  Attackers can send enormous payloads to exhaust
        # memory or push an attack past the visible analysis window.  We
        # truncate here and tag the result so callers can act on it.
        # Set config.max_input_length = 0 to disable (not recommended).
        # ---------------------------------------------------------------
        input_was_truncated = False
        max_len = self.config.max_input_length
        if max_len and max_len > 0 and len(text) > max_len:
            logger.warning(
                "Guardian: input truncated from %d to %d characters "
                "(max_input_length=%d).  Large payloads are a common "
                "attack vector — review if unexpected.",
                len(text), max_len, max_len,
            )
            text = text[:max_len]
            input_was_truncated = True

        # ---------------------------------------------------------------
        # Rate limiting — Principle 14 (Divine Safety).
        # Throttle excessive callers to protect backend resources.
        # ---------------------------------------------------------------
        rpm = self.config.max_requests_per_minute
        if rpm and rpm > 0:
            if self._throttler is None or self._throttler_rpm != rpm:
                try:
                    from asyncio_throttle import Throttler
                    self._throttler = Throttler(rate_limit=rpm, period=60)
                    self._throttler_rpm = rpm
                except ImportError:
                    logger.warning(
                        "asyncio-throttle not installed; rate limiting disabled. "
                        "Run: pip install asyncio-throttle"
                    )
                    self._throttler = None

            if self._throttler is not None:
                async with self._throttler:
                    pass  # Acquire slot; blocks if limit exceeded

        # ---------------------------------------------------------------
        # Cache lookup — Principle 12 (Sacred Privacy) + performance.
        # Bypass cache when session_id is present so multi-turn context
        # trackers (ContextPoisoningTracker, AutomatedScanDetector) see
        # every individual turn rather than a cached aggregate.
        # ---------------------------------------------------------------
        _use_cache = (
            self.config.cache_enabled
            and not (context or {}).get("session_id")
        )
        _cache_obj = self._get_cache() if _use_cache else None
        if _use_cache and _cache_obj:
            _ck = self._cache_key(
                text,
                str((context or {}).get("source_type", "")),
            )
            _cached = _cache_obj.get(_ck)
            if _cached is not None:
                logger.debug("Guardian: cache hit — skipping full analysis")
                return _cached

        try:
            # Use the orchestrator if available
            if self.threat_detector:
                timeout_s = (
                    self.config.analysis_timeout_ms / 1000.0
                    if self.config.analysis_timeout_ms and self.config.analysis_timeout_ms > 0
                    else None
                )
                try:
                    if timeout_s:
                        result = await asyncio.wait_for(
                            self.threat_detector.analyze(text, context or {}),
                            timeout=timeout_s,
                        )
                    else:
                        result = await self.threat_detector.analyze(text, context or {})
                except asyncio.TimeoutError:
                    # Principle 14 (Divine Safety): fail-closed — CHALLENGE, not ALLOW.
                    logger.warning(
                        "Guardian analysis timed out after %.1f s — returning CHALLENGE "
                        "(fail-closed).  Consider raising analysis_timeout_ms.",
                        timeout_s,
                    )
                    return self._timeout_analysis(self.config.analysis_timeout_ms)
            else:
                # Fallback: simple threat detection
                result = self._simple_threat_check(text)
            
            # Convert to public API format
            meta: Dict[str, Any] = dict(result.metadata)
            if input_was_truncated:
                # Principle 11 (Sacred Truth): callers deserve to know the
                # analysis was performed on a truncated version of the input.
                meta["input_truncated"] = True
                meta["original_length"] = len(text) + (
                    # re-add what we cut — len(text) is already the trimmed length
                    0
                )

            # -----------------------------------------------------------
            # Honest system confidence — Principle 11 (Sacred Truth).
            # Callers deserve to know whether a verdict came from 7/7
            # layers in strong agreement, or from a single outlier layer.
            # -----------------------------------------------------------
            _layer_votes_list = list(result.layer_votes) if result.layer_votes else []
            # Determine total possible layers (8 for ThreatDetector, 4 for SimpleOrchestrator).
            # Both expose a dict (.layers or .analyzers); use its length as the denominator.
            if self.threat_detector:
                _layers_dict = (
                    getattr(self.threat_detector, "layers", None)
                    or getattr(self.threat_detector, "analyzers", {})
                )
                _layers_total = len(_layers_dict) if isinstance(_layers_dict, dict) else 4
            else:
                _layers_total = 4
            if _layers_total == 0:
                _layers_total = max(4, len(_layer_votes_list))

            meta["system_confidence"] = self._build_system_confidence(
                _layer_votes_list, result.verdict, _layers_total
            )

            threat_analysis = ThreatAnalysis(
                is_safe=(result.verdict == 'ALLOW'),
                threat_score=min(1.0, result.overall_score / 100.0),
                threat_level=result.threat_level,
                threat_types=[t.get('category', 'unknown') for t in result.threats_detected],
                confidence=result.confidence,
                reasoning=result.reasoning,
                recommended_action=result.verdict,
                analysis_time_ms=int(result.analysis_time_ms),
                layer_votes={vote.layer: vote.vote for vote in result.layer_votes},
                metadata=meta,
            )

            # Low agreement note — Principle 11 (Sacred Truth): surface
            # uncertainty explicitly rather than silently returning a verdict.
            sc = meta["system_confidence"]
            if sc.get("confidence_basis") == "weak" and sc.get("layers_active", 0) > 1:
                threat_analysis.reasoning.append(
                    "Low inter-layer agreement — result based on limited signal; "
                    "human review recommended."
                )

            # -----------------------------------------------------------
            # Cache ALLOW results only — BLOCK/CHALLENGE may change with
            # additional context (different session, upstream data, etc.).
            # Principle 12 (Sacred Privacy): only the SHA-256 key is stored.
            # -----------------------------------------------------------
            if _use_cache and _cache_obj and threat_analysis.recommended_action == "ALLOW":
                try:
                    _cache_obj.set(
                        _ck,
                        threat_analysis,
                        expire=self.config.cache_ttl_seconds,
                    )
                except Exception as _ce:  # noqa: BLE001
                    logger.debug("Guardian: cache write failed (non-fatal): %s", _ce)

            # Principle 13 (Ultimate Accountability): record every decision.
            # Raw text is never written — only a hash (see audit.py, Principle 12).
            try:
                if _get_audit_logger is not None:
                    _get_audit_logger().record(text, threat_analysis, context)
            except Exception as _audit_err:  # noqa: BLE001
                logger.debug("Audit record failed (non-fatal): %s", _audit_err)

            return threat_analysis

        except Exception as e:
            # Principle 14 (Divine Safety): fail-closed — CHALLENGE, not ALLOW.
            # _error_analysis() logs at ERROR level so operators see the failure.
            return self._error_analysis(str(e))
    
    def _simple_threat_check(self, text: str):
        """Simple fallback threat detection"""
        import time
        start_time = time.time()
        
        threats = []
        reasoning = []
        score = 0
        
        # Simple pattern matching
        threat_patterns = [
            'ignore all previous instructions',
            'ignore previous instructions',
            'forget everything',
            'you are now',
            'act as',
            'DAN mode',
            'jailbreak'
        ]
        
        text_lower = text.lower()
        for pattern in threat_patterns:
            if pattern in text_lower:
                threats.append(pattern)
                reasoning.append(f"Matched threat pattern: {pattern}")
                score += 25
        
        analysis_time = (time.time() - start_time) * 1000
        
        if score >= 50:
            verdict = 'BLOCK'
            threat_level = 'HIGH'
        elif score >= 25:
            verdict = 'CHALLENGE'
            threat_level = 'MEDIUM'
        else:
            verdict = 'ALLOW'
            threat_level = 'NONE'
        
        return type('Result', (), {
            'verdict': verdict,
            'threat_level': threat_level,
            'overall_score': score,
            'confidence': 0.7,
            'threats_detected': [{'category': t} for t in threats],
            'reasoning': reasoning or ["No threats detected (simple check)"],
            'analysis_time_ms': analysis_time,
            'layer_votes': [],
            'metadata': {'fallback_mode': True}
        })()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Guardian statistics"""
        stats = {
            'guardian_version': "1.0.0",
            'initialized': self.initialized,
            'available_providers': list(self.providers.keys()),
            'config': {
                'enabled': self.config.enabled,
                'strict_mode': self.config.strict_mode,
                'pattern_sensitivity': self.config.pattern_sensitivity,
                'semantic_sensitivity': self.config.semantic_sensitivity,
                'ml_sensitivity': self.config.ml_sensitivity
            }
        }
        
        # Add threat detector stats
        if self.threat_detector:
            detector_stats = self.threat_detector.get_statistics()
            stats.update(detector_stats)
        
        return stats
    
    def configure(self, **kwargs) -> None:
        """Update Guardian configuration at runtime"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                old_value = getattr(self.config, key)
                setattr(self.config, key, value)
                print(f"🔧 Config updated: {key} = {value} (was {old_value})")
    
    # ------------------------------------------------------------------
    # Cache helpers — Principle 12 (Sacred Privacy)
    # ------------------------------------------------------------------

    def _get_cache(self) -> Any:
        """Lazily open the diskcache.Cache instance (thread-safe)."""
        if self._cache is not None:
            return self._cache
        try:
            import diskcache  # noqa: PLC0415
            cache_dir = Path(os.path.expanduser("~")) / ".ethicore_cache"
            size_limit = self.config.cache_max_size_mb * 1024 * 1024
            self._cache = diskcache.Cache(str(cache_dir), size_limit=size_limit)
        except ImportError:
            logger.warning(
                "diskcache not installed — caching disabled. "
                "Run: pip install diskcache"
            )
            self._cache = False  # sentinel: don't retry the import
        except Exception as e:
            logger.warning("Could not open diskcache at ~/.ethicore_cache: %s", e)
            self._cache = False
        return self._cache

    def _cache_key(self, text: str, source_type: str = "") -> str:
        """
        Derive a cache key from normalised text + source type.

        Principle 12 (Sacred Privacy): raw text is NEVER stored — only the
        SHA-256 digest is used as the cache key.
        """
        normalised = " ".join(text.lower().split())
        raw = f"{normalised}|{source_type}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # System-confidence helper — Principle 11 (Sacred Truth)
    # ------------------------------------------------------------------

    def _build_system_confidence(
        self, layer_votes_list: list, verdict: str, layers_total: int
    ) -> Dict[str, Any]:
        """
        Build a ``system_confidence`` metadata block that tells callers how
        many detection layers contributed to this verdict and how much they
        agreed.

        Principle 11 (Sacred Truth): callers deserve to know whether a BLOCK
        came from 7/7 layers or from a single layer with all others silent.
        """
        layers_active = len(layer_votes_list)
        if layers_active == 0:
            return {
                "layers_active": 0,
                "layers_total": layers_total,
                "block_agreement": 0,
                "agreement_ratio": 0.0,
                "confidence_basis": "none",
            }

        block_count = sum(1 for v in layer_votes_list if getattr(v, "vote", "") == "BLOCK")

        if verdict == "BLOCK":
            agree_count = block_count
        elif verdict == "CHALLENGE":
            agree_count = sum(
                1 for v in layer_votes_list if getattr(v, "vote", "") != "ALLOW"
            )
        else:  # ALLOW
            agree_count = sum(
                1 for v in layer_votes_list if getattr(v, "vote", "") == "ALLOW"
            )

        agree_ratio = agree_count / layers_active

        if agree_ratio >= 0.7:
            confidence_basis = "strong"
        elif agree_ratio >= 0.4:
            confidence_basis = "moderate"
        else:
            confidence_basis = "weak"

        return {
            "layers_active": layers_active,
            "layers_total": layers_total,
            "block_agreement": block_count,
            "agreement_ratio": round(agree_ratio, 3),
            "confidence_basis": confidence_basis,
        }

    # ------------------------------------------------------------------
    # Learning system access control — Principles 12 + 14
    # ------------------------------------------------------------------

    def _check_correction_key(self, provided: str) -> bool:
        """
        Validate a correction key using constant-time comparison to prevent
        timing-based enumeration attacks.

        Returns False if no correction_key is configured (corrections disabled).

        Principle 12 (Sacred Privacy): the expected key is never logged.
        Principle 14 (Divine Safety): timing-safe via hmac.compare_digest.
        """
        expected = self.config.correction_key
        if not expected:
            return False  # Disabled — no key configured
        try:
            return hmac.compare_digest(
                provided.encode("utf-8"), expected.encode("utf-8")
            )
        except Exception:
            return False

    def _get_ml_engine(self) -> Any:
        """Retrieve the active ML engine from whichever detector is loaded."""
        if not self.threat_detector:
            return None
        # SimpleOrchestrator exposes .analyzers dict
        analyzers = getattr(self.threat_detector, "analyzers", {})
        if "ml" in analyzers:
            return analyzers["ml"]
        # ThreatDetector exposes .layers dict
        layers = getattr(self.threat_detector, "layers", {})
        return layers.get("ml")

    def provide_correction(
        self, text: str, correct_label: str, correction_key: str
    ) -> bool:
        """
        Submit a learning correction to the ML engine with access control.

        Args:
            text:           The original prompt that was mis-classified.
            correct_label:  Correct classification label (e.g. 'threat' / 'safe').
            correction_key: Must match ``config.correction_key``; raises
                            ``PermissionError`` otherwise.

        Returns:
            True if the correction was accepted by the ML engine.

        Raises:
            PermissionError: Invalid or missing correction key.
            RuntimeError:    Rate limit exceeded.
        """
        if not self._check_correction_key(correction_key):
            raise PermissionError(
                "Invalid correction key.  Set ETHICORE_CORRECTION_KEY in config."
            )
        if not self._correction_limiter.consume():
            raise RuntimeError(
                f"Correction rate limit exceeded "
                f"({self.config.correction_rate_limit_per_minute}/min).  Try again later."
            )
        ml = self._get_ml_engine()
        if ml and hasattr(ml, "provide_correction"):
            return ml.provide_correction(text, correct_label)
        logger.warning("provide_correction: ML engine not available")
        return False

    def provide_feedback(
        self, text: str, feedback: Dict[str, Any], correction_key: str
    ) -> bool:
        """
        Submit structured feedback to the ML engine with access control.

        Args:
            text:           The original prompt.
            feedback:       Feedback dict forwarded to the ML engine.
            correction_key: Must match ``config.correction_key``.

        Returns:
            True if the feedback was accepted.

        Raises:
            PermissionError: Invalid or missing correction key.
            RuntimeError:    Rate limit exceeded.
        """
        if not self._check_correction_key(correction_key):
            raise PermissionError(
                "Invalid correction key.  Set ETHICORE_CORRECTION_KEY in config."
            )
        if not self._correction_limiter.consume():
            raise RuntimeError(
                f"Correction rate limit exceeded "
                f"({self.config.correction_rate_limit_per_minute}/min).  Try again later."
            )
        ml = self._get_ml_engine()
        if ml and hasattr(ml, "provide_feedback"):
            return ml.provide_feedback(text, feedback)
        logger.warning("provide_feedback: ML engine not available")
        return False

    # ------------------------------------------------------------------
    # Standard analysis helpers
    # ------------------------------------------------------------------

    def _empty_analysis(self, reason: str) -> ThreatAnalysis:
        """Create empty analysis result — used for valid empty-input edge cases only."""
        return ThreatAnalysis(
            is_safe=True,
            threat_score=0.0,
            threat_level='NONE',
            threat_types=[],
            confidence=1.0,
            reasoning=[reason],
            recommended_action='ALLOW',
            analysis_time_ms=0,
            layer_votes={},
            metadata={'reason': reason}
        )

    def _error_analysis(self, reason: str) -> ThreatAnalysis:
        """
        Return a CHALLENGE verdict when analysis fails unexpectedly.

        Principle 14 (Divine Safety): an analysis exception means we cannot
        vouch for the safety of the input.  Silently returning ALLOW would be
        fail-open — a critical security hole.  CHALLENGE surfaces the failure
        to the caller so they can decide whether to proceed or block.
        """
        logger.error("Guardian analysis error — returning CHALLENGE (fail-closed): %s", reason)
        return ThreatAnalysis(
            is_safe=False,
            threat_score=0.5,
            threat_level='UNKNOWN',
            threat_types=['analysis_error'],
            confidence=0.0,
            reasoning=[
                f"Analysis failed — defaulting to CHALLENGE (fail-closed per Principle 14): {reason}"
            ],
            recommended_action='CHALLENGE',
            analysis_time_ms=0,
            layer_votes={},
            metadata={'error': reason, 'fail_closed': True},
        )

    def _timeout_analysis(self, timeout_ms: int) -> ThreatAnalysis:
        """
        Return a CHALLENGE verdict when analysis times out.

        Principle 14 (Divine Safety): we never silently ALLOW when we cannot
        complete a full analysis.  CHALLENGE surfaces the uncertainty to the
        caller so they can decide whether to proceed or block.
        """
        reason = (
            f"Analysis timed out after {timeout_ms} ms — verdict is CHALLENGE "
            "(fail-closed per Principle 14).  The request was not cleared as safe."
        )
        return ThreatAnalysis(
            is_safe=False,
            threat_score=0.5,
            threat_level='UNKNOWN',
            threat_types=['analysis_timeout'],
            confidence=0.0,
            reasoning=[reason],
            recommended_action='CHALLENGE',
            analysis_time_ms=timeout_ms,
            layer_votes={},
            metadata={'timeout_ms': timeout_ms, 'timed_out': True},
        )


# Convenience functions
async def analyze_text(text: str, api_key: Optional[str] = None) -> ThreatAnalysis:
    """Convenience function for one-off text analysis"""
    guardian = Guardian(api_key=api_key)
    return await guardian.analyze(text)


def protect_openai(openai_client, api_key: str):
    """Convenience function to protect OpenAI client"""
    guardian = Guardian(api_key=api_key)
    return guardian.wrap(openai_client, provider='openai')