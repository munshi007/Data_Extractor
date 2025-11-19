"""
Core Ollama Client
Manages connection and communication with Ollama server
"""

import logging
import requests
import json
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class OllamaClient:
    """Base client for communicating with Ollama server."""
    
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: int = 300):
        """
        Initialize Ollama client.
        
        Args:
            host: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.host = host
        self.timeout = timeout
        self.session = requests.Session()
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to Ollama server."""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Ollama server at {self.host}")
            else:
                logger.warning(f"Ollama server returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server at {self.host}: {e}")
            raise
    
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        num_predict: int = 256,
        stream: bool = False
    ) -> str:
        """
        Generate text using Ollama model.
        
        Args:
            model: Model name (e.g., "mistral", "deepsee-r1")
            prompt: Input prompt
            system: System prompt/instructions
            temperature: Model temperature (0-1)
            num_predict: Max tokens to generate
            stream: Whether to stream response
            
        Returns:
            Generated text
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": num_predict,
            "stream": stream
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        full_response += data.get("response", "")
                        if data.get("done"):
                            break
                return full_response.strip()
            else:
                data = response.json()
                return data.get("response", "").strip()
        
        except Exception as e:
            logger.error(f"Error generating text with {model}: {e}")
            return ""
    
    def embed(self, model: str, text: str) -> List[float]:
        """
        Generate embeddings using Ollama model.
        
        Args:
            model: Embedding model name
            text: Input text
            
        Returns:
            Embedding vector
        """
        payload = {
            "model": model,
            "input": text
        }
        
        try:
            response = self.session.post(
                f"{self.host}/api/embed",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            return embeddings[0] if embeddings else []
        
        except Exception as e:
            logger.error(f"Error embedding text with {model}: {e}")
            return []
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def is_model_available(self, model: str) -> bool:
        """Check if a model is available."""
        models = self.list_models()
        model_names = [m.get("name", "") for m in models]
        
        # Check exact match
        if model in model_names:
            return True
        
        # Check base model name (without tag)
        model_base = model.split(":")[0]
        for m in model_names:
            if m.split(":")[0] == model_base:
                return True
        
        return False
