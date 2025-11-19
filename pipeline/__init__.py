"""
Enhanced PDF Processing Pipeline - Modular Architecture

This package provides a modular PDF processing pipeline with advanced layout detection,
OCR capabilities, and table extraction using spatial clustering.
"""

from .enhanced_pipeline import EnhancedPipeline
from .layout_detector import LayoutDetector
from .ocr_engine import OCREngine
from .table_extractor import TableExtractor
from .region_merger import merge_regions
from .markdown_renderer import MarkdownRenderer
from .page_processor import PageProcessor
from .reading_order import ReadingOrderResolver
from .semantic_grouper import SemanticTextGrouper
from .document_analyzer import DocumentAnalyzer
from .figure_caption_processor import FigureCaptionProcessor
from .region_processor import RegionProcessor
from .layoutlm_classifier import LayoutLMClassifier

__version__ = "2.0.0"
__all__ = [
    "EnhancedPipeline",
    "LayoutDetector", 
    "OCREngine",
    "TableExtractor",
    "merge_regions", 
    "MarkdownRenderer",
    "PageProcessor",
    "ReadingOrderResolver",
    "SemanticTextGrouper",
    "DocumentAnalyzer",
    "FigureCaptionProcessor",
    "RegionProcessor",
    "LayoutLMClassifier"
]