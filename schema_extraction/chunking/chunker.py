import re
from typing import List, Dict, Any

class MarkdownChunker:
    """
    Smart chunker for Markdown content that preserves structural context.
    """
    
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        # Rough estimate: 1 token ~= 4 chars
        self.max_chars = max_tokens * 4

    def chunk(self, text: str) -> List[str]:
        """
        Split markdown text into chunks based on headings and size limits.
        """
        # Split by headers (H1, H2, H3)
        # This regex looks for lines starting with #, ##, ###
        sections = re.split(r'(^#{1,3} .*$)', text, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if not section.strip():
                continue
                
            # If adding this section exceeds max size, save current chunk and start new
            if len(current_chunk) + len(section) > self.max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += section
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Fallback: If a single section is still too huge, split by paragraphs
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chars:
                final_chunks.extend(self._split_large_chunk(chunk))
            else:
                final_chunks.append(chunk)
                
        return final_chunks

    def _split_large_chunk(self, text: str) -> List[str]:
        """Split a very large chunk by paragraphs."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
