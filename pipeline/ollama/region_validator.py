"""
Tier 3: Ollama Region Validator
Uses DeepSeek-R1 to validate and fix region classification
"""

import logging
from typing import Dict, Any, Optional, List
from .client import OllamaClient

logger = logging.getLogger(__name__)


class OllamaRegionValidator:
    """Validate and fix region classifications using reasoning model."""
    
    def __init__(self, model: str = "deepsee-r1", host: str = "http://127.0.0.1:11434"):
        """
        Initialize region validator.
        
        Args:
            model: Reasoning model for validation
            host: Ollama server URL
        """
        self.model = model
        self.client = OllamaClient(host=host)
        
        self.system_prompt = """You are an expert document analyzer. 
Validate if the given region classification is correct based on its content and characteristics.
Respond with ONLY a JSON object like: {"valid": true/false, "corrected_type": "heading/text/title/caption", "confidence": 0.0-1.0, "reason": "brief reason"}"""
    
    def validate_region(self, region: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate region classification.
        
        Args:
            region: Region dict with 'text', 'type', 'bbox', etc.
            
        Returns:
            Validation result with correction suggestions
        """
        text = region.get('text', '')
        region_type = region.get('type', 'unknown')
        
        if not text or len(text.strip()) < 5:
            return {
                'valid': True,
                'corrected_type': region_type,
                'confidence': 0.9,
                'reason': 'Text too short for validation'
            }
        
        try:
            # Prepare validation prompt
            prompt = f"""Validate this region classification:
Text: {text[:200]}
Current Type: {region_type}
Font Size: {region.get('font_size', 'unknown')}
Is Bold: {region.get('font_info', {}).get('flags', 0) & 16 != 0}

Is the type '{region_type}' correct? Consider:
- Length and structure
- Font characteristics
- Content pattern"""
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.3,
                num_predict=256
            )
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response)
                if result.get('valid'):
                    logger.debug(f"Region validated: {region_type}")
                else:
                    logger.info(f"Region corrected: {region_type} → {result.get('corrected_type')}")
                return result
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON response from validator")
                return {
                    'valid': True,
                    'corrected_type': region_type,
                    'confidence': 0.5,
                    'reason': 'Validation model response parsing failed'
                }
        
        except Exception as e:
            logger.warning(f"Region validation failed: {e}")
            return {
                'valid': True,
                'corrected_type': region_type,
                'confidence': 0.0,
                'reason': f'Validation error: {str(e)}'
            }
    
    def validate_regions_batch(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate multiple regions and apply corrections.
        
        Args:
            regions: List of regions
            
        Returns:
            Corrected regions
        """
        corrected_regions = []
        
        for region in regions:
            validation = self.validate_region(region)
            
            # Apply correction if needed
            if not validation.get('valid') and validation.get('confidence', 0) > 0.7:
                region['type'] = validation.get('corrected_type', region.get('type'))
                region['validation'] = validation
                logger.info(f"Applied correction: {validation.get('reason')}")
            
            corrected_regions.append(region)
        
        return corrected_regions
    
    def is_likely_heading(self, text: str, font_size: float = 12, is_bold: bool = False) -> bool:
        """Quick heuristic check if text is likely a heading."""
        # Check length
        if len(text) > 100:
            return False
        
        # Check font properties
        if font_size > 14 or is_bold:
            return True
        
        # Check text patterns
        if text.isupper() or text.startswith("Chapter") or text.startswith("Section"):
            return True
        
        return False
    
    def is_likely_caption(self, text: str) -> bool:
        """Quick heuristic check if text is likely a caption."""
        if len(text) < 10 or len(text) > 200:
            return False
        
        # Caption patterns
        patterns = [
            r'^(Figure|Fig\.?\s*|Table|Diagram)\s+\d+',
            r'^\d+[\.\)]\s+',  # Numbered caption
            r'^(Source|Source:|Note|Note:)',
        ]
        
        import re
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
