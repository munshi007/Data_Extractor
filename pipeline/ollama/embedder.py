"""
Tier 1: Ollama Embedder for Semantic Grouping
Replaces sentence-transformers with Ollama embeddings for better performance
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from .client import OllamaClient

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Generate embeddings using Ollama for semantic similarity."""
    
    def __init__(self, model: str = "nomic-embed-text", host: str = "http://127.0.0.1:11434"):
        """
        Initialize embedder with Ollama.
        
        Args:
            model: Embedding model name
            host: Ollama server URL
        """
        self.model = model
        self.client = OllamaClient(host=host)
        self.cache = {}
        
        if not self.client.is_model_available(model):
            logger.warning(f"Model {model} not available in Ollama")
        else:
            logger.info(f"Initialized OllamaEmbedder with model: {model}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if text in self.cache:
            return self.cache[text]
        
        # Generate embedding
        embedding = self.client.embed(self.model, text)
        
        if embedding:
            embedding_array = np.array(embedding, dtype=np.float32)
            self.cache[text] = embedding_array
            return embedding_array
        else:
            logger.warning(f"Failed to embed text: {text[:50]}...")
            return np.zeros(768, dtype=np.float32)  # Default embedding size
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Matrix of embeddings
        """
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def encode(self, texts, show_progress_bar=False):
        """
        Compatibility method for sentence-transformers API.
        
        Args:
            texts: Single text or list of texts
            show_progress_bar: Ignored, included for compatibility
            
        Returns:
            Single embedding or matrix of embeddings
        """
        if isinstance(texts, str):
            return self.embed(texts)
        else:
            return self.embed_batch(texts)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1, text2: Input texts
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        if len(emb1) == 0 or len(emb2) == 0:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()
