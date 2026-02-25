"""
Ethicore Engine™ - Guardian SDK - OpenAI Provider (Fixed)
Self-contained version that doesn't rely on external base classes
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import json
from functools import wraps

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Provider-specific exception"""
    pass


class ThreatBlockedException(Exception):
    """Exception raised when Guardian issues a BLOCK verdict."""
    def __init__(self, analysis_result, message="Threat detected and blocked"):
        self.analysis_result = analysis_result
        super().__init__(message)


class ThreatChallengeException(Exception):
    """
    Exception raised when Guardian issues a CHALLENGE verdict in non-strict mode.

    Callers should surface a secondary verification step (e.g. CAPTCHA, human
    review) rather than hard-blocking the request.  In ``strict_mode``,
    CHALLENGE is escalated to ``ThreatBlockedException`` instead.

    Principle 16 (Sacred Autonomy): preserves human agency by surfacing
    uncertainty rather than silently blocking.
    """
    def __init__(self, analysis_result: Any, message: str = "Request requires verification") -> None:
        self.analysis_result = analysis_result
        super().__init__(message)


class OpenAIProvider:
    """
    OpenAI Provider Wrapper (Self-contained)
    
    Intercepts OpenAI API calls and applies Guardian threat detection
    before allowing requests to proceed.
    
    Maintains complete API compatibility while adding security.
    """
    
    def __init__(self, guardian_instance):
        self.guardian = guardian_instance
        self.provider_name = "openai"
    
    def wrap_client(self, client) -> 'ProtectedOpenAIClient':
        """
        Wrap OpenAI client with Guardian protection
        
        Args:
            client: OpenAI client instance
            
        Returns:
            ProtectedOpenAIClient that maintains API compatibility
        """
        try:
            import openai
        except ImportError:
            raise ProviderError("OpenAI package not installed. Run: pip install openai")
        
        # Validate client type
        if not self._is_openai_client(client):
            raise ProviderError(f"Expected OpenAI client, got {type(client)}")
        
        return ProtectedOpenAIClient(client, self.guardian)
    
    def _is_openai_client(self, client) -> bool:
        """Check if client is a valid OpenAI client"""
        client_type = str(type(client)).lower()
        return 'openai' in client_type
    
    def extract_prompt(self, *args, **kwargs) -> str:
        """
        Extract prompt text from OpenAI API call arguments
        
        Handles various OpenAI API formats:
        - chat.completions.create(messages=[...])
        - completions.create(prompt="...")
        """
        # Chat completions format
        if 'messages' in kwargs:
            return self._extract_from_messages(kwargs['messages'])
        
        # Legacy completions format
        elif 'prompt' in kwargs:
            prompt = kwargs['prompt']
            return prompt if isinstance(prompt, str) else str(prompt)
        
        # Check args for messages
        elif len(args) > 0:
            for arg in args:
                if isinstance(arg, dict) and 'messages' in arg:
                    return self._extract_from_messages(arg['messages'])
        
        return ""
    
    def _extract_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Extract text from OpenAI messages format"""
        if not messages:
            return ""
        
        # Get the last user message (most relevant for threat detection)
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if user_messages:
            last_message = user_messages[-1]
            content = last_message.get('content', '')
            
            # Handle both string and list content formats
            if isinstance(content, list):
                # Extract text from content array
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                return ' '.join(text_parts)
            else:
                return str(content)
        
        return ""


class ProtectedOpenAIClient:
    """
    Protected OpenAI client that maintains full API compatibility
    while adding Guardian threat detection
    """
    
    def __init__(self, original_client, guardian_instance):
        self._original_client = original_client
        self._guardian = guardian_instance
        self._provider = OpenAIProvider(guardian_instance)
        
        # Preserve all original client attributes and methods
        for attr_name in dir(original_client):
            if not attr_name.startswith('_'):
                attr = getattr(original_client, attr_name)
                if not callable(attr):
                    # Copy non-callable attributes directly
                    setattr(self, attr_name, attr)
        
        # Wrap the chat completions interface
        if hasattr(original_client, 'chat'):
            self.chat = self._create_protected_chat()
        
        # Wrap legacy completions interface
        if hasattr(original_client, 'completions'):
            self.completions = self._create_protected_completions()
        
        logger.debug("🛡️  OpenAI client protection enabled")
    
    def _create_protected_chat(self):
        """Create protected chat interface"""
        class ProtectedChat:
            def __init__(self, original_chat, guardian, provider):
                self._original_chat = original_chat
                self._guardian = guardian
                self._provider = provider
                
                # Preserve other chat attributes
                for attr_name in dir(original_chat):
                    if not attr_name.startswith('_') and attr_name != 'completions':
                        attr = getattr(original_chat, attr_name)
                        if not callable(attr):
                            setattr(self, attr_name, attr)
                
                # Create protected completions
                if hasattr(original_chat, 'completions'):
                    self.completions = self._create_protected_completions()
            
            def _create_protected_completions(self):
                """Create protected completions interface"""
                class ProtectedCompletions:
                    def __init__(self, original_completions, guardian, provider):
                        self._original_completions = original_completions
                        self._guardian = guardian
                        self._provider = provider
                        
                        # Preserve other completions attributes
                        for attr_name in dir(original_completions):
                            if not attr_name.startswith('_') and attr_name not in ['create', 'acreate']:
                                attr = getattr(original_completions, attr_name)
                                if not callable(attr):
                                    setattr(self, attr_name, attr)
                    
                    def create(self, **kwargs):
                        """Protected chat completions create method"""
                        return self._guardian_protect_request(
                            self._original_completions.create,
                            **kwargs
                        )
                    
                    async def acreate(self, **kwargs):
                        """Protected async chat completions create method"""
                        return await self._guardian_protect_request_async(
                            self._original_completions.acreate,
                            **kwargs
                        )
                    
                    def _guardian_protect_request(self, original_method, **kwargs):
                        """Apply Guardian protection to sync request"""
                        # Extract prompt for analysis
                        prompt_text = self._provider.extract_prompt(**kwargs)
                        
                        if prompt_text and len(prompt_text.strip()) > 0:
                            # Run threat analysis (async)
                            loop = None
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                pass
                            
                            if loop:
                                # We're in an async context, need to run in thread
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(asyncio.run, self._analyze_threat(prompt_text, kwargs))
                                    analysis = future.result()
                            else:
                                # Not in async context, can run directly
                                analysis = asyncio.run(self._analyze_threat(prompt_text, kwargs))
                            
                            # Check result
                            if not analysis.is_safe:
                                self._handle_threat_detected(analysis, prompt_text)
                        
                        # Request is safe, proceed with original call
                        return original_method(**kwargs)
                    
                    async def _guardian_protect_request_async(self, original_method, **kwargs):
                        """Apply Guardian protection to async request"""
                        # Extract prompt for analysis
                        prompt_text = self._provider.extract_prompt(**kwargs)
                        
                        if prompt_text and len(prompt_text.strip()) > 0:
                            # Run threat analysis
                            analysis = await self._analyze_threat(prompt_text, kwargs)
                            
                            # Check result
                            if not analysis.is_safe:
                                self._handle_threat_detected(analysis, prompt_text)
                        
                        # Request is safe, proceed with original call
                        return await original_method(**kwargs)
                    
                    async def _analyze_threat(self, prompt_text: str, request_kwargs: Dict) -> Any:
                        """Analyze prompt for threats"""
                        # Prepare analysis context
                        context = {
                            'api_call': 'openai.chat.completions.create',
                            'model': request_kwargs.get('model', 'unknown'),
                            'max_tokens': request_kwargs.get('max_tokens'),
                            'temperature': request_kwargs.get('temperature'),
                            'request_size': len(prompt_text),
                        }
                        
                        # Run Guardian analysis
                        return await self._guardian.analyze(prompt_text, context)
                    
                    def _handle_threat_detected(self, analysis, prompt_text: str):
                        """
                        Apply Guardian policy to the analysis result.

                        BLOCK                    → always raise ThreatBlockedException
                        CHALLENGE + strict_mode  → escalate to ThreatBlockedException
                        CHALLENGE + non-strict   → raise ThreatChallengeException so
                                                   callers can surface a verification step
                        """
                        if analysis.recommended_action == 'BLOCK':
                            logger.warning(
                                "🚨 BLOCKED OpenAI request — %s: %.100s…",
                                analysis.threat_level, prompt_text,
                            )
                            logger.warning("   Reasons: %s", ', '.join(analysis.reasoning[:2]))
                            raise ThreatBlockedException(
                                analysis_result=analysis,
                                message=(
                                    f"Request blocked: {analysis.threat_level} threat detected. "
                                    f"Reasons: {', '.join(analysis.reasoning[:2])}"
                                ),
                            )

                        elif analysis.recommended_action == 'CHALLENGE':
                            logger.warning(
                                "⚠️  CHALLENGE OpenAI request — %s: %.100s…",
                                analysis.threat_level, prompt_text,
                            )
                            logger.warning("   Reasons: %s", ', '.join(analysis.reasoning[:2]))
                            if self._guardian.config.strict_mode:
                                # Principle 14 (Divine Safety): in strict mode,
                                # treat CHALLENGE the same as BLOCK.
                                raise ThreatBlockedException(
                                    analysis_result=analysis,
                                    message=(
                                        f"Request blocked (strict mode — CHALLENGE): "
                                        f"{analysis.threat_level} threat detected."
                                    ),
                                )
                            else:
                                raise ThreatChallengeException(
                                    analysis_result=analysis,
                                    message=(
                                        f"Request requires verification: "
                                        f"{analysis.threat_level} threat level."
                                    ),
                                )
                
                return ProtectedCompletions(self._original_chat.completions, self._guardian, self._provider)
        
        return ProtectedChat(self._original_client.chat, self._guardian, self._provider)
    
    def _create_protected_completions(self):
        """Create protected legacy completions interface"""
        class ProtectedLegacyCompletions:
            def __init__(self, original_completions, guardian, provider):
                self._original_completions = original_completions
                self._guardian = guardian
                self._provider = provider
            
            def create(self, **kwargs):
                """Protected legacy completions create method"""
                prompt_text = self._provider.extract_prompt(**kwargs)
                
                if prompt_text:
                    analysis = asyncio.run(self._guardian.analyze(prompt_text))
                    
                    if not analysis.is_safe and (self._guardian.config.strict_mode or analysis.recommended_action == 'BLOCK'):
                        raise ThreatBlockedException(
                            analysis_result=analysis,
                            message=f"Request blocked: {analysis.threat_level} threat detected"
                        )
                
                return self._original_completions.create(**kwargs)
        
        return ProtectedLegacyCompletions(self._original_client.completions, self._guardian, self._provider)
    
    def __getattr__(self, name):
        """Delegate unknown attributes to original client"""
        return getattr(self._original_client, name)
    
    def __repr__(self):
        """String representation"""
        return f"ProtectedOpenAIClient(original={repr(self._original_client)})"


# Helper functions for easier integration
def create_protected_openai_client(api_key: str, guardian_api_key: str, **openai_kwargs):
    """
    Create a protected OpenAI client in one step
    
    Args:
        api_key: OpenAI API key
        guardian_api_key: Guardian API key
        **openai_kwargs: Arguments passed to OpenAI client
        
    Returns:
        Protected OpenAI client
    """
    try:
        import openai
    except ImportError:
        raise ProviderError("OpenAI package not installed. Run: pip install openai")
    
    # Create OpenAI client
    openai_client = openai.OpenAI(api_key=api_key, **openai_kwargs)
    
    # Create Guardian and wrap client
    from ..guardian import Guardian
    guardian = Guardian(api_key=guardian_api_key)
    
    return guardian.wrap(openai_client)