
import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check for Ollama text cleaner availability
OLLAMA_CLEANER_AVAILABLE = False
try:
    from .ollama.text_cleaner import OllamaTextCleaner
    OLLAMA_CLEANER_AVAILABLE = True
except ImportError:
    logger.info("Ollama text cleaner not available, text will not be LLM-cleaned")


class MarkdownRenderer:
    """
    Renders ordered regions to markdown with proper formatting for headings, paragraphs, tables, and figures.
    With Ollama text cleaning support (Tier 2 enhancement)
    """

    def __init__(self, debug: bool = False, use_ollama_cleaning: bool = True):
        self.debug = debug
        self.use_ollama_cleaning = use_ollama_cleaning
        self.text_cleaner = None
        
        # Initialize Ollama text cleaner (Tier 2)
        if use_ollama_cleaning and OLLAMA_CLEANER_AVAILABLE:
            try:
                logger.info("Initializing Ollama text cleaner for markdown (Tier 2)")
                self.text_cleaner = OllamaTextCleaner()
                logger.info("Ollama text cleaner initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama text cleaner: {e}")

    def extract_markdown_from_regions(self, regions: List[Dict[str, Any]]) -> str:
        """
        Convert ordered regions to markdown format.
        
        Args:
            regions: List of regions in reading order
            
        Returns:
            Markdown formatted string
        """
        parts: List[str] = []

        for region in regions:
            region_id = region.get("region_id", "unknown")
            region_type = region.get("type", region.get("region_type", "unknown"))
            bbox = region.get("bbox", [])
            method = region.get("method", region.get("source", "unknown"))

            # Debug comments removed for cleaner output
            # if self.debug:
            #     parts.append(f"<!-- Region {region_id}: {region_type} ({method}) with bbox {bbox} -->")

            # Render based on region type
            rendered = self._render_region(region)
            if rendered:
                parts.append(rendered)

        return "\n\n".join([p for p in parts if p is not None])
    
    def _render_region(self, region: Dict[str, Any]) -> str:
        """
        Render a single region based on its type.
        
        Args:
            region: Region dictionary
            
        Returns:
            Markdown formatted string
        """
        region_type = region.get("type", region.get("region_type", "unknown")).lower()
        
        # Headings
        if region_type in ["title", "heading"]:
            return self._render_heading(region)
        
        # Tables (case-insensitive)
        elif region_type.lower() == "table":
            return self._render_table(region)
        
        # Figures (case-insensitive)
        elif region_type.lower() == "figure":
            return self._render_figure(region)
        
        # Lists
        elif region.get("is_list_item"):
            return self._render_list_item(region)
        
        # Captions (if not already associated with figure/table)
        elif region_type in ["figurecaption", "tablecaption"] and not region.get("associated_with"):
            return self._render_caption(region)
        
        # Regular text/paragraphs
        else:
            text = region.get("text", "").strip()
            return text if text else None
    
    def _render_heading(self, region: Dict[str, Any]) -> str:
        """Render heading with appropriate level."""
        text = region.get("text", "").strip()
        if not text:
            return None
        
        # Clean text with Ollama if available (Tier 2)
        if self.text_cleaner:
            text = self.text_cleaner.clean_heading(text)
        
        # Determine heading level based on font size or type
        font_size = region.get("font_size", 12)
        region_type = region.get("type", "").lower()
        
        if region_type == "title" or font_size > 20:
            level = 1
        elif font_size > 16:
            level = 2
        else:
            level = 3
        
        return f"{'#' * level} {text}"
    
    def _render_list_item(self, region: Dict[str, Any]) -> str:
        """Render list item (already has bullet/number in text)."""
        text = region.get("text", "").strip()
        return text if text else None

    def _render_table(self, region: Dict[str, Any]) -> str:
        """Render table region to markdown."""
        # Get rows from table_data if available
        table_data = region.get("table_data", {})
        rows = table_data.get("rows") or region.get("rows")
        
        if not rows or len(rows) == 0:
            return None
        
        table_md = self._format_table_as_markdown(rows)
        
        # Add caption if available
        caption = region.get("caption")
        if caption:
            return f"{caption}\n\n{table_md}"
        
        return table_md
    
    def _render_figure(self, region: Dict[str, Any]) -> str:
        """Render figure with caption."""
        parts = []
        
        # Add figure image
        img_path = region.get("snapshot_image")
        if img_path:
            parts.append(f"![Figure]({img_path})")
        
        # Add caption
        caption = region.get("caption") or region.get("text")
        if caption:
            parts.append(caption)
        
        return "\n\n".join(parts) if parts else None
    
    def _render_caption(self, region: Dict[str, Any]) -> str:
        """Render standalone caption (not associated with figure/table)."""
        text = region.get("text", "").strip()
        if text:
            return f"*{text}*"  # Italicize captions
        return None
    
    def _format_table_as_markdown(self, rows: List[List[str]]) -> str:
        if not rows:
            return ""

        lengths = [len(r) for r in rows if r]
        if not lengths:
            return ""
        median_len = int(np.median(lengths))
        if median_len <= 0:
            return ""

        norm: List[List[str]] = []
        for r in rows:
            r = r[:median_len] + [""] * (median_len - len(r))
            norm.append([self._clean_cell_for_markdown(c) for c in r])

        header_keywords = ["Pin", "Channel", "IN", "OUT", "Type"]

        def row_has_header_words(r: List[str]) -> bool:
            return any(any(k in (c or "") for k in header_keywords) for c in r)

        if row_has_header_words(norm[0]):
            header = norm[0]
            data_rows = norm[1:]
        else:
            header = [f"Col{i+1}" for i in range(median_len)]
            data_rows = norm

        lines: List[str] = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * median_len) + " |")
        for r in data_rows:
            lines.append("| " + " | ".join(r) + " |")

        return "\n".join(lines)

    @staticmethod
    def _clean_cell_for_markdown(s: Any) -> str:
        if s is None:
            return ""
        text = str(s)
        text = text.replace("|", r"\|")
        return " ".join(text.split())
