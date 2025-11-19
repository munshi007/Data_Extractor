"""
Tier 4: Ollama Post-Processor
Final pass using Mistral to improve markdown quality and readability
"""

import logging
import re
from typing import Optional, Dict, Any
from .client import OllamaClient

logger = logging.getLogger(__name__)


class OllamaPostProcessor:
    """Post-process final markdown output for quality improvement."""
    
    def __init__(self, model: str = "mistral", host: str = "http://127.0.0.1:11434"):
        """
        Initialize post-processor.
        
        Args:
            model: LLM model for post-processing
            host: Ollama server URL
        """
        self.model = model
        self.client = OllamaClient(host=host)
        
        self.system_prompt = """You are a markdown editor specializing in technical documents.
Your task is to improve the markdown quality while preserving all content.
- Fix formatting issues
- Improve readability
- Ensure consistent structure
- Remove redundancy
Return the improved markdown directly."""
    
    def improve_markdown(self, markdown: str, aggressive: bool = False) -> str:
        """
        Improve markdown output quality.
        
        Args:
            markdown: Raw markdown content
            aggressive: Apply more aggressive improvements
            
        Returns:
            Improved markdown
        """
        if not markdown or len(markdown.strip()) < 100:
            return markdown
        
        # First pass: regex-based improvements
        improved = self._regex_improve(markdown)
        
        # Second pass: LLM-based improvements (optional, slower)
        if aggressive:
            improved = self._llm_improve(improved)
        
        return improved
    
    def _regex_improve(self, markdown: str) -> str:
        """Apply regex-based markdown improvements."""
        
        # Fix multiple blank lines
        markdown = re.sub(r'\n\n\n+', '\n\n', markdown)
        
        # Ensure proper heading spacing
        markdown = re.sub(r'\n(#+\s)', r'\n\n\1', markdown)
        markdown = re.sub(r'(#+\s.*)\n(?!#)', r'\1\n', markdown)
        
        # Fix table formatting
        markdown = re.sub(r'\|\s*\n', '|\n', markdown)
        markdown = re.sub(r'\n\s*\|', '\n|', markdown)
        
        # Fix list formatting
        markdown = re.sub(r'\n(-\s+)(\s+)', r'\n\1', markdown)
        markdown = re.sub(r'\n(\*\s+)(\s+)', r'\n\1', markdown)
        
        # Remove trailing spaces
        markdown = '\n'.join(line.rstrip() for line in markdown.split('\n'))
        
        # Ensure document ends with newline
        if not markdown.endswith('\n'):
            markdown += '\n'
        
        return markdown
    
    def _llm_improve(self, markdown: str) -> str:
        """Use LLM for intelligent markdown improvement."""
        try:
            # Only process first 4000 chars to avoid token limits
            chunk = markdown[:4000]
            
            prompt = f"Improve this markdown document:\n\n{chunk}"
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.3,
                num_predict=1024
            )
            
            return response if response else markdown
        except Exception as e:
            logger.warning(f"LLM improvement failed: {e}")
            return markdown
    
    def validate_markdown_structure(self, markdown: str) -> Dict[str, Any]:
        """
        Validate markdown structure and identify issues.
        
        Args:
            markdown: Markdown content
            
        Returns:
            Validation report
        """
        issues = []
        stats = {
            'headings': len(re.findall(r'^\#+\s', markdown, re.MULTILINE)),
            'tables': len(re.findall(r'^\|', markdown, re.MULTILINE)) // 3,  # Approximate
            'code_blocks': len(re.findall(r'```', markdown)),
            'images': len(re.findall(r'!\[', markdown)),
            'links': len(re.findall(r'\[.*?\]\(.*?\)', markdown)),
            'lines': len(markdown.split('\n')),
        }
        
        # Check for common issues
        if re.search(r'\n\n\n\n', markdown):
            issues.append("Multiple blank lines detected")
        
        if re.search(r'^\#+\s*$', markdown, re.MULTILINE):
            issues.append("Empty headings detected")
        
        # Check table structure
        tables = re.findall(r'^\|.*\|$', markdown, re.MULTILINE)
        if tables:
            for i, table in enumerate(tables):
                if table.count('|') < 3:
                    issues.append(f"Table {i+1} may be malformed")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }
    
    def add_table_of_contents(self, markdown: str) -> str:
        """Add table of contents to markdown."""
        headings = re.findall(r'^(#+)\s+(.+)$', markdown, re.MULTILINE)
        
        if not headings or len(headings) < 3:
            return markdown  # Too few headings for TOC
        
        toc = "## Table of Contents\n\n"
        for level, heading in headings:
            indent = "  " * (len(level) - 1)
            anchor = heading.lower().replace(" ", "-")
            toc += f"{indent}- [{heading}](#{anchor})\n"
        
        # Insert TOC after first heading
        first_heading_pos = markdown.find('\n#')
        if first_heading_pos > 0:
            return markdown[:first_heading_pos+1] + toc + "\n" + markdown[first_heading_pos+1:]
        
        return markdown


# Type hints (moved to top of file)
