"""
Guardian SDK - Ollama Provider
Protects local LLM interactions through Ollama
Version: 1.0.0

Supports all Ollama models: Mistral, Llama, CodeLlama, Vicuna, etc.
"""

import asyncio
import logging
import httpx
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ThreatBlockedException(Exception):
    """Exception raised when a threat is blocked"""
    def __init__(self, analysis_result, message="Threat detected and blocked"):
        self.analysis_result = analysis_result
        super().__init__(message)


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection"""
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    verify_ssl: bool = True


class OllamaProvider:
    """
    Ollama Provider Wrapper for Guardian SDK
    
    Protects local LLM interactions (Mistral, Llama, CodeLlama, etc.)
    through Ollama API with Guardian threat detection.
    """
    
    def __init__(self, guardian_instance, config: Optional[OllamaConfig] = None):
        self.guardian = guardian_instance
        self.config = config or OllamaConfig()
        self.provider_name = "ollama"
        
        # HTTP client for Ollama API
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl
        )
        
        logger.info(f"🦙 Ollama provider initialized: {self.config.base_url}")
    
    def wrap_client(self, ollama_client=None):
        """Create protected Ollama client"""
        return ProtectedOllamaClient(self.guardian, self.config)
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = await self.client.get("/api/tags")
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
            return []


class ProtectedOllamaClient:
    """Protected Ollama client with Guardian threat detection"""
    
    def __init__(self, guardian_instance, config: OllamaConfig):
        self.guardian = guardian_instance
        self.config = config
        
        # HTTP client for API calls
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            verify=config.verify_ssl
        )
        
        logger.debug("🛡️ Protected Ollama client created")
    
    async def chat(self, 
                   model: str,
                   messages: List[Dict[str, str]],
                   stream: bool = False,
                   options: Optional[Dict] = None) -> Dict[str, Any]:
        """Protected chat completion with local LLM"""
        
        # Extract user message for analysis
        user_message = self._extract_user_message(messages)
        
        if user_message:
            # Analyze with Guardian
            context = {
                'provider': 'ollama',
                'model': model,
                'local_llm': True,
                'base_url': self.config.base_url
            }
            
            analysis = await self.guardian.analyze(user_message, context)
            
            # Handle threat detection
            if not analysis.is_safe:
                self._handle_threat_detected(analysis, user_message, model)
        
        # Safe to proceed with Ollama API call
        return await self._make_ollama_request(
            model=model,
            messages=messages,
            stream=stream,
            options=options
        )
    
    def _extract_user_message(self, messages: List[Dict[str, str]]) -> str:
        """Extract user message from chat messages"""
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if user_messages:
            return user_messages[-1].get('content', '')
        return ''
    
    def _handle_threat_detected(self, analysis, prompt: str, model: str):
        """Handle detected threats based on configuration"""
        
        if self.guardian.config.strict_mode or analysis.recommended_action == 'BLOCK':
            logger.warning(
                f"🚨 BLOCKED Ollama/{model} request - {analysis.threat_level}: "
                f"{prompt[:100]}..."
            )
            logger.warning(f"   Reasons: {', '.join(analysis.reasoning)}")
            
            raise ThreatBlockedException(
                analysis_result=analysis,
                message=f"Local LLM request blocked: {analysis.threat_level} threat detected"
            )
    
    async def _make_ollama_request(self, model: str, messages: List[Dict], 
                                  stream: bool = False, options: Optional[Dict] = None) -> Dict[str, Any]:
        """Make actual Ollama chat API request"""
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if options:
            payload["options"] = options
        
        try:
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json()
                
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Ollama request failed: {e}")
    
    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = await self.client.get("/api/tags")
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Convenience function for easy setup
def create_protected_ollama_client(guardian_api_key: str, 
                                 ollama_base_url: str = "http://localhost:11434",
                                 strict_mode: bool = True):
    """Create a protected Ollama client in one step"""
    from ethicore_guardian import Guardian
    
    guardian = Guardian(
        api_key=guardian_api_key,
        strict_mode=strict_mode
    )
    
    config = OllamaConfig(base_url=ollama_base_url)
    provider = OllamaProvider(guardian, config)
    
    return provider.wrap_client()