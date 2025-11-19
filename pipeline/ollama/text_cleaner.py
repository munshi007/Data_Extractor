"""
Tier 2: Ollama Text Cleaner
Uses Mistral to clean and normalize extracted text
"""

import logging
import re
from typing import Optional
from .client import OllamaClient

logger = logging.getLogger(__name__)


class OllamaTextCleaner:
    """Clean and normalize extracted PDF text using LLM."""
    
    def __init__(self, model: str = "mistral", host: str = "http://127.0.0.1:11434"):
        """
        Initialize text cleaner.
        
        Args:
            model: LLM model for cleaning
            host: Ollama server URL
        """
        self.model = model
        self.client = OllamaClient(host=host)
        
        self.system_prompt = """You are a text normalization expert. 
Your task is to clean and fix extracted text from PDFs. 
- Fix OCR errors and typos
- Normalize spacing and formatting
- Preserve original meaning
- Keep it concise
Return ONLY the cleaned text, nothing else."""
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Clean extracted text using LLM.
        
        Args:
            text: Raw extracted text
            aggressive: Apply more aggressive cleaning
            
        Returns:
            Cleaned text
        """
        if not text or len(text.strip()) == 0:
            return text
        
        # First pass: regex-based cleaning (fast)
        cleaned = self._regex_clean(text)
        
        # Second pass: LLM-based cleaning (if aggressive or complex text)
        if aggressive or self._should_clean_with_llm(cleaned):
            cleaned = self._llm_clean(cleaned)
        
        return cleaned.strip()
    
    def _regex_clean(self, text: str) -> str:
        """Apply regex-based cleaning."""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove trailing/leading spaces
        text = text.strip()
        
        # Fix common OCR errors
        replacements = {
            'Mulitifunctional': 'Multifunctional',
            'occured': 'occurred',
            'recieve': 'receive',
            'seperator': 'separator',
            'Seperator': 'Separator',
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _should_clean_with_llm(self, text: str) -> bool:
        """Determine if text needs LLM-based cleaning."""
        # Only clean if text is long enough and contains potential issues
        issues = [
            r'[0-9]\s+[a-z]',  # number followed by lowercase (potential OCR error)
            r'[a-z]{2,}\s{2,}[a-z]',  # strange spacing
            r'[^a-zA-Z0-9\s\-\(\)\[\],.:;]',  # unusual characters
        ]
        
        return len(text) > 50 and any(re.search(issue, text) for issue in issues)
    
    def _llm_clean(self, text: str) -> str:
        """Use LLM for intelligent text cleaning."""
        try:
            prompt = f"Clean this extracted text:\n\n{text}"
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.3,
                num_predict=512
            )
            return response if response else text
        except Exception as e:
            logger.warning(f"LLM cleaning failed, returning original: {e}")
            return text
    
    def clean_heading(self, text: str) -> str:
        """Clean heading text specifically."""
        text = self.clean_text(text, aggressive=False)
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        return text
    
    def clean_paragraph(self, text: str) -> str:
        """Clean paragraph text specifically."""
        return self.clean_text(text, aggressive=False)
    
    def clean_table_text(self, text: str) -> str:
        """Clean table cell text."""
        text = self.clean_text(text, aggressive=False)
        # Keep cell text concise
        if len(text) > 200:
            text = text[:197] + "..."
        return text
