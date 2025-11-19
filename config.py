"""
Configuration settings for the Enhanced PDF Processing Pipeline
"""

import os
from pathlib import Path

# Model Configuration
MODEL_CONFIG = {
    'confidence_threshold': 0.25,
    'layout_detection_dpi': 300,
    'thumbnail_dpi': 150,
    'table_detection_threshold': 0.6,
}

# Font Classification Thresholds
FONT_CONFIG = {
    'heading_font_threshold': 14,
    'paragraph_font_threshold': 10,
}

# OCR Configuration
OCR_CONFIG = {
    'use_angle_cls': True,
    'lang': 'en',
    'confidence_threshold': 0.5,
}

# Table Processing Configuration
TABLE_CONFIG = {
    'max_text_boxes': 1000,
    'max_rows': 100,
    'max_cols': 20,
    'y_threshold': 8,  # pixels for row clustering
    'cell_padding': 2,  # padding to avoid line artifacts
}

# Output Configuration
OUTPUT_CONFIG = {
    'base_output_dir': 'Output',  # Base directory for all outputs
    'thumbnails_subdir': 'layout_thumbnails',
    'tables_subdir': 'extracted_tables',
    'figures_subdir': 'extracted_figures',  # Separate folder for figures
    'debug_subdir': 'debug_visualizations',
    'json_filename': 'enhanced_layout_blocks.json',
    'markdown_filename': 'extracted_content.md',
}

# Environment Configuration
ENV_CONFIG = {
    'opencv_io_enable_openexr': '0',
    'display': '',  # Disable display for headless operation
}

# DocLayout-YOLO Model Configuration
DOCLAYOUT_CONFIG = {
    'repo_id': 'juliozhao/DocLayout-YOLO-DocStructBench',
    'filename': 'doclayout_yolo_docstructbench_imgsz1024.pt',
    'target_size': 1024,
    'id2label': {
        0: "Title", 1: "Text", 2: "Abandon",
        3: "Figure", 4: "FigureCaption", 5: "Table",
        6: "TableCaption", 7: "TableFootnote",
        8: "IsolatedFormula", 9: "FormulaCaption"
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

def setup_environment():
    """Setup environment variables for headless operation."""
    for key, value in ENV_CONFIG.items():
        os.environ[key.upper()] = value

def get_output_paths(output_dir: str = None):
    """Get standardized output directory paths.
    
    Args:
        output_dir: Output directory path. If None, uses default from config
    
    Returns:
        Dictionary with all output paths
    """
    base_dir = Path(output_dir or OUTPUT_CONFIG['base_output_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'base': base_dir,
        'thumbnails': base_dir / OUTPUT_CONFIG['thumbnails_subdir'],
        'tables': base_dir / OUTPUT_CONFIG['tables_subdir'],
        'figures': base_dir / OUTPUT_CONFIG['figures_subdir'],
        'debug': base_dir / OUTPUT_CONFIG['debug_subdir'],
        'json_file': base_dir / OUTPUT_CONFIG['json_filename'],
        'markdown_file': base_dir / OUTPUT_CONFIG['markdown_filename'],
    }