"""
Enhanced Pipeline Module - Main Pipeline Class with Model Reuse
Exact architectural match to original enhanced_pdf_pipeline.py but with modular components
"""

import json
import logging
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import fitz  # PyMuPDF
from PIL import Image
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import MODEL_CONFIG, FONT_CONFIG, OUTPUT_CONFIG, setup_environment
from .layout_detector import LayoutDetector
from .ocr_engine import OCREngine
from .table_extractor import TableExtractor
from .region_merger import merge_regions
from .markdown_renderer import MarkdownRenderer
from .page_processor import PageProcessor
from .utils import (
    extract_text_blocks_with_fonts, 
    save_image, 
    aggregate_region_stats,
    create_bounding_box_visualization,
    create_debug_comparison
)

# Setup environment
setup_environment()

logger = logging.getLogger(__name__)

class EnhancedPipeline:
    """Enhanced PDF processing pipeline with proper model integration and reuse."""
    
    def __init__(self, output_dir: str = None, debug_mode: bool = False):
        """Initialize the enhanced PDF pipeline with model reuse."""
        logger.info("Initializing Enhanced PDF Pipeline with modular architecture")
        
        # Get output paths (will auto-number if output_dir is None)
        from config import get_output_paths
        output_paths = get_output_paths(output_dir)
        
        self.output_dir = output_paths['base']
        self.thumbnails_dir = output_paths['thumbnails']
        self.tables_dir = output_paths['tables']
        self.figures_dir = output_paths['figures']
        self.debug_dir = output_paths['debug']
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        
        self.debug_mode = debug_mode
        if debug_mode:
            self.debug_dir.mkdir(exist_ok=True)
        
        # Store configuration
        self.config = MODEL_CONFIG.copy()
        self.config.update(FONT_CONFIG)
        
        # Initialize output paths dictionary
        self.output_paths = {
            'main': self.output_dir,
            'thumbnails': self.thumbnails_dir,
            'tables': self.tables_dir,
            'figures': self.figures_dir,
            'debug': self.debug_dir,
            'json_file': self.output_dir / OUTPUT_CONFIG['json_filename'],
            'markdown_file': self.output_dir / OUTPUT_CONFIG['markdown_filename']
        }
        
        # Initialize components ONCE (key difference from functional approach)
        self._initialize_components()
        
        logger.info(f"Enhanced PDF Pipeline initialized with output dir: {output_dir}")
        logger.info(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")
    
    def _initialize_components(self):
        """Initialize all pipeline components once for reuse across pages."""
        logger.info("Initializing pipeline components...")
        
        # Initialize document analyzer (computes adaptive thresholds)
        from .document_analyzer import DocumentAnalyzer
        self.document_analyzer = DocumentAnalyzer()
        logger.info("Document analyzer initialized")
        
        # Initialize layout detector (loads model once)
        self.layout_detector = LayoutDetector(debug_mode=self.debug_mode)
        logger.info("Layout detector initialized")
        
        # Initialize OCR engine (loads model once)
        self.ocr_engine = OCREngine()
        logger.info("OCR engine initialized")
        
        # Initialize reading order resolver
        from .reading_order import ReadingOrderResolver
        self.reading_order_resolver = ReadingOrderResolver()
        logger.info("Reading order resolver initialized")
        
        # Initialize semantic text grouper
        from .semantic_grouper import SemanticTextGrouper
        self.semantic_grouper = SemanticTextGrouper()
        logger.info("Semantic text grouper initialized")
        
        # Initialize figure-caption processor
        from .figure_caption_processor import FigureCaptionProcessor
        self.figure_caption_processor = FigureCaptionProcessor()
        logger.info("Figure-caption processor initialized")
        
        # Initialize region processor (hierarchical processing with LayoutLMv3)
        from .region_processor import RegionProcessor
        self.region_processor = RegionProcessor(use_layoutlm=True)
        logger.info("Region processor initialized")
        
        # Initialize table extractor (reuses OCR engine)
        self.table_extractor = TableExtractor(self.output_paths)
        logger.info("Table extractor initialized")
        
        # Region merger is now a function, no initialization needed
        logger.info("Region merger function available")
        
        # Initialize markdown renderer
        self.markdown_renderer = MarkdownRenderer(debug=self.debug_mode)
        logger.info("Markdown renderer initialized")
        
        # Initialize page processor for region snapshots
        self.page_processor = PageProcessor(self.output_paths)
        logger.info("Page processor initialized")
        
        # Initialize Ollama post-processor (Tier 4)
        self.post_processor = None
        try:
            from .ollama.post_processor import OllamaPostProcessor
            logger.info("Initializing Ollama post-processor (Tier 4)")
            self.post_processor = OllamaPostProcessor()
            logger.info("Ollama post-processor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama post-processor (Tier 4): {e}")
        
        logger.info("All pipeline components initialized successfully")
    
    def process_page(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Process a single PDF page using initialized components."""
        logger.info(f"Processing page {page_num}")
        
        try:
            # Extract page information
            page_info = {
                "page_num": page_num,
                "page_size": {
                    "width": page.rect.width,
                    "height": page.rect.height
                }
            }
            
            # Convert page to image for processing
            page_image, dpi_scale = self._convert_page_to_image(page)
            
            # Save page thumbnail
            thumbnail_path = self.thumbnails_dir / f"page_{page_num:02d}.png"
            save_image(page_image, thumbnail_path, f"page thumbnail: page_{page_num:02d}.png")
            page_info["thumbnail_path"] = f"{OUTPUT_CONFIG['thumbnails_subdir']}/page_{page_num:02d}.png"
            
            # Step 0: Analyze document to compute adaptive profile
            logger.info("Analyzing document for adaptive thresholds")
            doc_profile = self.document_analyzer.analyze(page_image)
            logger.info(f"Document type detected: {doc_profile.document_type.value}")
            
            # Step 1: Extract text blocks with font information
            logger.info("Extracting text blocks with font information")
            text_blocks = extract_text_blocks_with_fonts(page)
            logger.info(f"Extracted {len(text_blocks)} text blocks from page {page_num}")
            
            # Step 2: Detect layout regions using ensemble (reuses initialized models)
            logger.info("Detecting layout regions with ensemble")
            layout_regions = self.layout_detector.detect_layout_regions(
                page_image, 
                debug=self.debug_mode,
                use_ensemble=True  # Use ensemble detection
            )
            logger.info(f"Detected {len(layout_regions)} layout regions on page {page_num}")
            
            # Step 3: Process table regions (smart hybrid: pdfplumber + PaddleOCR)
            table_regions = [r for r in layout_regions if r.get('type') in ['Table', 'table']]
            for i, table_region in enumerate(table_regions):
                logger.info(f"Processing table region {i+1} with confidence {table_region.get('confidence', 0):.3f}")
                
                # Convert table bbox from image space to PDF space for pdfplumber
                image_bbox = table_region['bbox']
                pdf_bbox = [coord / dpi_scale for coord in image_bbox]
                
                # Pass PDF path for pdfplumber extraction
                table_data = self.table_extractor.extract_table_structure(
                    page_image, 
                    image_bbox,  # Use image bbox for cropping
                    page_num, 
                    i+1,
                    doc_profile=doc_profile,
                    pdf_path=self.current_pdf_path,  # Pass PDF path
                    pdf_page_num=page_num - 1,  # pdfplumber uses 0-indexed pages
                    pdf_bbox=pdf_bbox  # Pass PDF-space bbox for pdfplumber
                )
                table_region['table_data'] = table_data
                
                # Save table image
                table_image_name = f"table_page_{page_num:02d}_{i+1:02d}.png"
                table_region['table_image_path'] = table_image_name
                logger.info(f"Saved table image: {table_image_name}")
            
            # Step 3.5: Create visualizations BEFORE coordinate conversion (so coordinates match image space)
            self._create_bounding_box_visualization(page_image, layout_regions, page_num)
            
            # Step 3.6: Attach region snapshots for Figure/Table regions (before coordinate conversion)
            layout_regions = self.page_processor.attach_region_snapshots(page_image, layout_regions)
            
            # Step 4: Associate figures with captions (before coordinate conversion)
            logger.info("Associating figures with captions")
            layout_regions = self.figure_caption_processor.associate_captions(
                layout_regions, 
                doc_profile
            )
            
            # Step 5: Convert layout regions from image coordinates to PDF coordinates
            logger.info("Converting layout regions to PDF coordinate space")
            layout_regions = self._convert_regions_to_pdf_coords(layout_regions, dpi_scale)
            
            # Step 6: Apply reading order
            logger.info("Applying reading order")
            layout_regions = self.reading_order_resolver.order_regions(
                layout_regions,
                page_image,
                doc_profile
            )
            logger.info(f"Regions ordered by reading flow")
            
            # Step 7: Process regions hierarchically (trust model labels, filter text inside figures)
            logger.info("Processing regions hierarchically with LayoutLMv3")
            processed_regions = self.region_processor.process_regions_hierarchically(
                layout_regions, 
                text_blocks,
                page_image=page_image  # Pass image for LayoutLMv3
            )
            logger.info(f"Hierarchical processing completed: {len(processed_regions)} regions")
            
            # Step 7.5: Merge remaining regions intelligently
            logger.info("Merging processed regions")
            merged_regions = merge_regions(processed_regions, [])
            logger.info(f"Region merging completed: {len(merged_regions)} final regions")
            
            # Step 8: Group text into paragraphs using semantic similarity
            text_regions = [r for r in merged_regions if r.get('type') in ['text', 'paragraph', 'Text']]
            if text_regions:
                logger.info("Grouping text into paragraphs")
                paragraph_groups = self.semantic_grouper.group_paragraphs(
                    text_regions,
                    doc_profile
                )
                logger.info(f"Grouped {len(text_regions)} text regions into {len(paragraph_groups)} paragraphs")
                
                # Create new merged paragraph regions
                new_paragraph_regions = []
                for i, group in enumerate(paragraph_groups):
                    if not group:
                        continue
                    
                    # Sort group by reading order (top-down, left-right)
                    group.sort(key=lambda r: (r.get('bbox', [0,0,0,0])[1], r.get('bbox', [0,0,0,0])[0]))
                    
                    # Merge text with smart separator (newline for vertical gaps)
                    merged_text_parts = []
                    valid_group = [r for r in group if r.get('text', '').strip()]
                    
                    if valid_group:
                        merged_text_parts.append(valid_group[0].get('text', '').strip())
                        last_bbox = valid_group[0].get('bbox', [0,0,0,0])
                        
                        for r in valid_group[1:]:
                            text = r.get('text', '').strip()
                            bbox = r.get('bbox', [0,0,0,0])
                            
                            # Check vertical distance (if y1 of current is > y2 of last + threshold)
                            # Use 2 pixels as threshold for line break (tight threshold for lists)
                            if bbox[1] > last_bbox[3] + 2:
                                merged_text_parts.append("\n" + text)
                            else:
                                merged_text_parts.append(" " + text)
                            
                            last_bbox = bbox
                    
                    full_text = "".join(merged_text_parts)
                    
                    # Calculate union bbox
                    bboxes = [r.get('bbox') for r in group if r.get('bbox')]
                    if bboxes:
                        x1 = min(b[0] for b in bboxes)
                        y1 = min(b[1] for b in bboxes)
                        x2 = max(b[2] for b in bboxes)
                        y2 = max(b[3] for b in bboxes)
                        new_bbox = [x1, y1, x2, y2]
                    else:
                        new_bbox = group[0].get('bbox', [0,0,0,0])
                    
                    new_region = {
                        'id': f"paragraph_{page_num}_{i}",
                        'type': 'paragraph',
                        'text': full_text,
                        'bbox': new_bbox,
                        'page': page_num,
                        'source': 'semantic_grouper'
                    }
                    new_paragraph_regions.append(new_region)
                
                # Replace text regions with new paragraphs
                non_text_regions = [r for r in merged_regions if r not in text_regions]
                merged_regions = non_text_regions + new_paragraph_regions
            
            # Step 8.5: Final cleanup - sort by Y-coordinate and remove duplicates
            logger.info("Final cleanup: sorting and deduplication")
            merged_regions = self._final_cleanup(merged_regions)
            logger.info(f"After cleanup: {len(merged_regions)} regions")
            
            # Step 9: Generate clean markdown content
            logger.info("Generating clean markdown content")
            self.markdown_renderer.clean_output = not self.debug_mode  # Clean output unless debug
            markdown_content = self.markdown_renderer.extract_markdown_from_regions(merged_regions)
            
            # Step 7: Create debug visualizations if enabled (using image-space coordinates)
            if self.debug_mode:
                # Convert merged regions back to image space for visualization
                image_space_regions = self._convert_regions_to_image_coords(merged_regions, dpi_scale)
                self._create_debug_visualizations(page_image, image_space_regions, page_num)
            
            # Aggregate statistics (using original methods)
            stats = {
                "total_regions": len(merged_regions),
                "text_blocks": len(text_blocks),
                "layout_regions": len(layout_regions),
                "tables_found": len([r for r in layout_regions if r['type'] in ['Table', 'table']]),
                "region_types": self._count_region_types(merged_regions),
                "processing_methods": self._count_processing_methods(merged_regions)
            }
            
            # Build page result
            page_result = {
                **page_info,
                "regions": merged_regions,
                "markdown": markdown_content,
                "stats": stats
            }
            
            logger.info(f"Page {page_num} processed successfully: {len(merged_regions)} total regions")
            return page_result
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "page_num": page_num,
                "error": str(e),
                "regions": [],
                "markdown": "",
                "stats": {}
            }
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process entire PDF document with enhanced functionality."""
        logger.info(f"Starting enhanced PDF processing: {pdf_path}")
        
        # Store PDF path for table extraction
        self.current_pdf_path = pdf_path
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Process each page using initialized components
            pages_data = []
            all_markdown = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_result = self.process_page(page, page_num + 1)
                pages_data.append(page_result)
                
                # Collect markdown
                if page_result.get('markdown'):
                    all_markdown.append(f"## Page {page_num + 1}\\n\\n{page_result['markdown']}")
            
            # Build final result (same structure as original)
            result = self._build_final_result(pdf_path, doc, pages_data, all_markdown)
            
            # Save results
            self._save_results(result)
            
            logger.info(f"Enhanced PDF processing completed. Results saved to: {self.output_paths['json_file']}")
            logger.info(f"Markdown content saved to: {self.output_paths['markdown_file']}")
            logger.info(f"Summary: {result['summary']}")
            
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _final_cleanup(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Smart cleanup: Trust model labels, only filter text that's truly inside tables/figures
        
        Strategy:
        1. Keep ALL model-detected regions (they're semantic, not spatial)
        2. For PyMuPDF text: only keep if it's in "empty space" (not covered by model regions)
        3. Use LayoutLMv3 to classify ambiguous text
        """
        # Separate by source
        model_regions = [r for r in regions if r.get('source') == 'layout_model']
        text_regions = [r for r in regions if r.get('source') != 'layout_model']
        
        logger.info(f"Cleanup: {len(model_regions)} model regions, {len(text_regions)} text regions")
        
        # For each text region, check if it's in "empty space"
        filtered_text = []
        for text_region in text_regions:
            text_bbox = text_region.get('bbox')
            if not text_bbox:
                continue
            
            # Check if this text is already covered by a model region
            # Only remove if text is COMPLETELY inside a Table/Figure (not Title/Text regions)
            is_duplicate = False
            for model_region in model_regions:
                model_type = model_region.get('type', '').lower()
                model_bbox = model_region.get('bbox')
                
                if not model_bbox:
                    continue
                
                # Only filter text inside Table/Figure (not Title/Text/Caption)
                # Tables are handled by pdfplumber, so remove OCR text from tables
                # Figures should keep their captions, so only remove text DEEP inside
                if model_type == 'table':
                    # Remove text completely inside tables (pdfplumber handles it)
                    if self._is_contained(text_bbox, model_bbox, threshold=0.95):
                        is_duplicate = True
                        logger.debug(f"Removing duplicate text inside table")
                        break
                elif model_type == 'figure':
                    # Only remove text VERY deep inside figures (keep captions/labels)
                    if self._is_contained(text_bbox, model_bbox, threshold=0.99):
                        is_duplicate = True
                        logger.debug(f"Removing text deep inside figure")
                        break
            
            if not is_duplicate:
                filtered_text.append(text_region)
        
        logger.info(f"After filtering: {len(filtered_text)} text regions kept")
        
        # Combine and sort by Y-coordinate
        all_regions = model_regions + filtered_text
        all_regions.sort(key=lambda r: r.get('bbox', [0, 0, 0, 0])[1])
        
        return all_regions
    
    def _is_contained(self, small_bbox: List[float], large_bbox: List[float], threshold: float = 0.8) -> bool:
        """Check if small_bbox is contained within large_bbox"""
        x1_min, y1_min, x1_max, y1_max = small_bbox
        x2_min, y2_min, x2_max, y2_max = large_bbox
        
        # Check if small bbox is mostly inside large bbox
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return False
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        small_area = (x1_max - x1_min) * (y1_max - y1_min)
        
        # If most of the small bbox is inside the large bbox
        return (inter_area / small_area) > threshold if small_area > 0 else False
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _convert_page_to_image(self, page: fitz.Page) -> tuple:
        """Convert PDF page to numpy image array and return scaling info."""
        dpi_scale = self.config['layout_detection_dpi'] / 72
        mat = fitz.Matrix(dpi_scale, dpi_scale)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        page_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        page_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2RGB)
        
        return page_image, dpi_scale
    
    def _convert_regions_to_pdf_coords(self, regions: List[Dict], dpi_scale: float) -> List[Dict]:
        """Convert region coordinates from image space to PDF coordinate space."""
        converted_regions = []
        
        for region in regions:
            # Create a copy of the region
            converted_region = region.copy()
            
            # Convert bounding box coordinates
            if 'bbox' in region:
                bbox = region['bbox']
                converted_bbox = [
                    bbox[0] / dpi_scale,  # x1
                    bbox[1] / dpi_scale,  # y1
                    bbox[2] / dpi_scale,  # x2
                    bbox[3] / dpi_scale   # y2
                ]
                converted_region['bbox'] = converted_bbox
            
            converted_regions.append(converted_region)
        
        return converted_regions
    
    def _convert_regions_to_image_coords(self, regions: List[Dict], dpi_scale: float) -> List[Dict]:
        """Convert region coordinates from PDF space back to image coordinate space for visualization."""
        converted_regions = []
        
        for region in regions:
            # Create a copy of the region
            converted_region = region.copy()
            
            # Convert bounding box coordinates
            if 'bbox' in region:
                bbox = region['bbox']
                converted_bbox = [
                    bbox[0] * dpi_scale,  # x1
                    bbox[1] * dpi_scale,  # y1
                    bbox[2] * dpi_scale,  # x2
                    bbox[3] * dpi_scale   # y2
                ]
                converted_region['bbox'] = converted_bbox
            
            converted_regions.append(converted_region)
        
        return converted_regions
    
    def _create_debug_visualizations(self, page_image: np.ndarray, regions: List[Dict], page_num: int):
        """Create debug visualizations for the page - matching original behavior."""
        try:
            # Create bounding box visualization and save to thumbnails_dir (like original)
            bbox_image = create_bounding_box_visualization(page_image.copy(), regions, debug_mode=False)
            bbox_path = self.thumbnails_dir / f"page_{page_num:02d}_bboxes.png"
            save_image(bbox_image, bbox_path, f"enhanced bounding box visualization: page_{page_num:02d}_bboxes.png")
            
            # Create debug comparison and save to debug_dir (only in debug mode)
            if self.debug_mode:
                debug_image = create_debug_comparison(page_image, bbox_image, regions)
                debug_path = self.debug_dir / f"page_{page_num:02d}_debug.png"
                save_image(debug_image, debug_path, f"debug visualization: page_{page_num:02d}_debug.png")
            
        except Exception as e:
            logger.warning(f"Failed to create debug visualizations for page {page_num}: {e}")
    
    def _build_final_result(self, pdf_path: str, doc, pages_data: List[Dict], all_markdown: List[str]) -> Dict[str, Any]:
        """Build the final result dictionary matching original structure."""
        result = {
            "document_info": {
                "path": pdf_path,
                "filename": os.path.basename(pdf_path),
                "total_pages": len(doc),
                "processing_timestamp": datetime.now().isoformat()
            },
            "pipeline_config": self.config,
            "model_info": {
                "layout_model_available": self.layout_detector.is_available(),
                "ocr_reader_available": self.ocr_engine.is_available(),
                "ocr_type": self.ocr_engine.get_ocr_info().get('type', 'unknown'),
                "dependencies": self.layout_detector.get_dependencies()
            },
            "pages": pages_data,
            "full_markdown": "\n\n".join(all_markdown),
            "summary": {
                "total_regions": sum(len(p.get("regions", [])) for p in pages_data),
                "pages_processed": len(pages_data),
                "pages_with_errors": len([p for p in pages_data if "error" in p]),
                "tables_found": sum(p.get("stats", {}).get("tables_found", 0) for p in pages_data),
                "region_types": self._aggregate_stats(pages_data, "region_types"),
                "processing_methods": self._aggregate_stats(pages_data, "processing_methods")
            }
        }
        
        return result
    
    def _aggregate_stats(self, pages_data: List[Dict], stat_key: str) -> Dict[str, int]:
        """Aggregate statistics across all pages."""
        aggregated = {}
        for page_data in pages_data:
            stats = page_data.get("stats", {}).get(stat_key, {})
            for key, value in stats.items():
                aggregated[key] = aggregated.get(key, 0) + value
        return aggregated
    
    def _save_results(self, result: Dict[str, Any]):
        """Save processing results to files."""
        # Save JSON results
        with open(self.output_paths['json_file'], 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save markdown content
        with open(self.output_paths['markdown_file'], 'w', encoding='utf-8') as f:
            f.write(result['full_markdown'])
    
    def _create_bounding_box_visualization(self, page_image: np.ndarray, regions: List[Dict], page_num: int):
        """Create and save enhanced bounding box visualization using utility function."""
        try:
            # Use utility function to create visualization
            vis_image = create_bounding_box_visualization(page_image.copy(), regions, debug_mode=self.debug_mode)
            
            # Save visualization
            bbox_filename = f"page_{page_num:02d}_bboxes.png"
            bbox_path = self.thumbnails_dir / bbox_filename
            save_image(vis_image, bbox_path, f"enhanced bounding box visualization: {bbox_filename}")
            
        except Exception as e:
            logger.warning(f"Failed to create bounding box visualization for page {page_num}: {e}")
    

    
    def _create_debug_visualizations(self, page_image: np.ndarray, regions: List[Dict], page_num: int):
        """Create debug visualizations using utility functions."""
        try:
            # Always create bounding box visualization
            self._create_bounding_box_visualization(page_image, regions, page_num)
            
            # Create debug comparison only in debug mode
            if self.debug_mode:
                # Create annotated version
                annotated = create_bounding_box_visualization(page_image.copy(), regions, debug_mode=True)
                
                # Create comparison using utility function
                debug_comparison = create_debug_comparison(page_image, annotated, regions)
                
                # Save debug comparison
                debug_filename = f"page_{page_num:02d}_debug.png"
                debug_path = self.debug_dir / debug_filename
                save_image(debug_comparison, debug_path, f"debug visualization: {debug_filename}")
                
        except Exception as e:
            logger.warning(f"Failed to create debug visualizations for page {page_num}: {e}")
    
    def _count_region_types(self, regions: List[Dict]) -> Dict[str, int]:
        """Count regions by type (exact copy from original)."""
        counts = {}
        for region in regions:
            region_type = region.get('type', 'unknown')
            counts[region_type] = counts.get(region_type, 0) + 1
        return counts
    
    def _count_processing_methods(self, regions: List[Dict]) -> Dict[str, int]:
        """Count regions by processing method (exact copy from original)."""
        counts = {}
        for region in regions:
            source = region.get('source', 'unknown')
            counts[source] = counts.get(source, 0) + 1
        return counts