import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum
import json

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# Import Ollama cleaner
try:
    from .ollama.text_cleaner import OllamaTextCleaner
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class TableStyle(Enum):
    """Table style classification"""
    GRID = "grid"
    SEMI_GRID = "semi_grid"
    BORDERLESS = "borderless"
    UNKNOWN = "unknown"


def save_image(img, out_path: Path, msg: str = "") -> None:
    """Utility to save an image if OpenCV is available."""
    if img is None or img.size == 0:
        logger.warning("save_image: empty image for %s", out_path)
        return
    if cv2 is None:
        logger.warning("OpenCV not available; cannot save %s", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    if msg:
        logger.debug(msg)


class TableStyleDetector:
    """Detect table style using Hough transform for grid detection"""
    
    def detect_style(self, table_image: np.ndarray) -> TableStyle:
        """
        Detect table style using adaptive Hough transform
        
        Args:
            table_image: Table image as numpy array
            
        Returns:
            TableStyle classification
        """
        if cv2 is None or table_image is None or table_image.size == 0:
            return TableStyle.UNKNOWN
        
        try:
            # Convert to grayscale
            if len(table_image.shape) == 3:
                gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = table_image
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180,
                threshold=50,
                minLineLength=30,
                maxLineGap=10
            )
            
            if lines is None:
                return TableStyle.BORDERLESS
            
            # Classify lines as horizontal or vertical
            h_lines = []
            v_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 10 or angle > 170:  # Horizontal
                    h_lines.append(line)
                elif 80 < angle < 100:  # Vertical
                    v_lines.append(line)
            
            # Grid table if enough lines in both directions
            if len(h_lines) >= 3 and len(v_lines) >= 3:
                return TableStyle.GRID
            elif len(h_lines) >= 2 or len(v_lines) >= 2:
                return TableStyle.SEMI_GRID
            else:
                return TableStyle.BORDERLESS
                
        except Exception as e:
            logger.warning(f"Failed to detect table style: {e}")
            return TableStyle.UNKNOWN


class TableExtractor:
    """
    Smart hybrid table extraction:
    1. Use pdfplumber on detected table bbox (fast, accurate for structured PDFs)
    2. Use PPStructureV3 for image-based tables (ML-based structure recognition)
    3. Fallback to PaddleOCR if PPStructureV3 fails
    4. Use grid detection as last resort
    """

    def __init__(self, output_paths: Dict[str, Path]):
        # expects keys like 'tables'
        self.output_paths = output_paths
        self.style_detector = TableStyleDetector()
        
        # Try to import pdfplumber
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
            self.pdfplumber_available = True
            logger.info("pdfplumber available for table extraction")
        except ImportError:
            self.pdfplumber = None
            self.pdfplumber_available = False
            logger.warning("pdfplumber not available - will use OCR fallback")
        
        # Try to import PPStructureV3
        try:
            from paddleocr import PPStructureV3
            self.ppstructure = PPStructureV3(lang='en')
            self.ppstructure_available = True
            logger.info("PPStructureV3 available for table structure recognition")
        except Exception as e:
            self.ppstructure = None
            self.ppstructure_available = False
            logger.warning(f"PPStructureV3 not available: {e}")
            
        # Initialize Ollama cleaner
        # DISABLED: Table refinement is too slow/unstable
        self.ollama_cleaner = None
        if OLLAMA_AVAILABLE:
             logger.info("Ollama table refiner disabled by configuration")

    # ----------------------------- PUBLIC API ----------------------------- #
    def extract_table_structure(
        self,
        page_image: np.ndarray,
        table_bbox: List[float],
        page_num: int,
        table_num: int,
        doc_profile: Optional[Any] = None,
        pdf_path: Optional[str] = None,
        pdf_page_num: Optional[int] = None,
        pdf_bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Smart hybrid table extraction:
        1. Try pdfplumber on detected bbox (best for structured PDFs)
        2. Fallback to PaddleOCR (for scanned/image PDFs)
        3. Last resort: grid detection
        
        Args:
            page_image: Page image as numpy array
            table_bbox: Table bounding box [x1, y1, x2, y2]
            page_num: Page number for output naming
            table_num: Table number on page
            doc_profile: Optional document profile
            pdf_path: Path to original PDF (for pdfplumber)
            pdf_page_num: PDF page number (0-indexed)
        """
        if page_image is None or page_image.size == 0:
            return {"rows": [], "method": "none", "error": "empty page image"}

        x1, y1, x2, y2 = [int(c) for c in table_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

        table_crop = page_image[y1:y2, x1:x2] if page_image is not None else None
        if table_crop is None or table_crop.size == 0:
            return {"rows": [], "method": "none", "error": "empty table crop"}

        # Save table image
        table_filename = f"table_page_{page_num:02d}_{table_num:02d}.png"
        tables_dir = self.output_paths.get("tables", Path("Output/tables"))
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / table_filename
        save_image(table_crop, table_path, f"table image: {table_filename}")

        # Helper to finalize and refine
        def finalize_result(data, method_name):
            if not data or not data.get("rows") or len(data["rows"]) <= 1:
                return None
            
            # Refine with Ollama if available and not pdfplumber (which is usually trusted)
            # Also refine if pdfplumber produced very few rows or columns, indicating potential failure
            should_refine = (self.ollama_cleaner is not None) and (method_name != "pdfplumber")
            
            if should_refine:
                logger.info(f"Refining {method_name} table with Ollama...")
                data["rows"] = self._refine_with_ollama(data["rows"])
            
            data["table_image_path"] = str(table_path)
            return data

        # SMART EXTRACTION STRATEGY
        table_data = None
        
        # Step 1: Try pdfplumber (best for structured PDFs)
        if self.pdfplumber_available and pdf_path and pdf_page_num is not None and pdf_bbox:
            logger.info("Trying pdfplumber extraction...")
            table_data = self._extract_with_pdfplumber(pdf_path, pdf_page_num, pdf_bbox)
            result = finalize_result(table_data, "pdfplumber")
            if result:
                logger.info(f"✓ pdfplumber extracted {len(result['rows'])} rows")
                return result
        
        # Step 2: Try PPStructureV3 (ML-based table structure recognition)
        if self.ppstructure_available:
            logger.info("Trying PPStructureV3 extraction...")
            table_data = self._extract_with_ppstructure(table_crop)
            result = finalize_result(table_data, "ppstructure")
            if result:
                logger.info(f"✓ PPStructureV3 extracted {len(result['rows'])} rows")
                return result
        
        # Step 3: Fallback to PaddleOCR (simple text extraction)
        logger.info("Trying PaddleOCR extraction...")
        table_data = self._extract_table_with_ocr(table_crop)
        result = finalize_result(table_data, "ocr")
        if result:
            logger.info(f"✓ PaddleOCR extracted {len(result['rows'])} rows")
            return result
        
        # Step 4: Last resort - grid detection
        logger.info("Trying grid detection...")
        table_style = self.style_detector.detect_style(table_crop)
        if table_style == TableStyle.GRID:
            table_data = self._extract_table_with_lines(table_crop)
            if table_data and table_data.get("rows"):
                table_data["table_style"] = table_style.value
                result = finalize_result(table_data, "grid")
                if result:
                    logger.info(f"✓ Grid detection extracted {len(result['rows'])} rows")
                    return result

        # Failed - return empty
        logger.warning(f"All table extraction methods failed for table {table_num} on page {page_num}")
        return {
            "rows": [],
            "method": "failed",
            "table_image_path": str(table_path)
        }

    # --------------------------- INTERNAL METHODS --------------------------- #
    
    def _extract_with_ppstructure(self, table_img: np.ndarray) -> Dict[str, Any]:
        """
        Extract table using PPStructureV3 (ML-based table structure recognition).
        This uses PaddleOCR's built-in table structure model.
        
        Args:
            table_img: Table image as numpy array
            
        Returns:
            Dict with rows and method
        """
        try:
            # Run PPStructureV3 on table image
            result = self.ppstructure.predict(table_img)
            
            if not result or len(result) == 0:
                return {"rows": [], "method": "ppstructure"}
            
            # Get table results
            table_res_list = result[0].get('table_res_list', [])
            
            if not table_res_list:
                return {"rows": [], "method": "ppstructure"}
            
            # Get first table (should be the main one)
            table_result = table_res_list[0]
            
            # Get HTML table structure
            html_table = table_result.get('pred_html', '')
            
            if not html_table:
                return {"rows": [], "method": "ppstructure"}
            
            # Parse HTML to extract rows
            rows = self._parse_html_table(html_table)
            
            logger.info(f"PPStructureV3 extracted {len(rows)} rows from HTML")
            return {"rows": rows, "method": "ppstructure", "html": html_table}
            
        except Exception as e:
            logger.warning(f"PPStructureV3 extraction failed: {e}")
            return {"rows": [], "method": "ppstructure"}
    
    def _parse_html_table(self, html: str) -> List[List[str]]:
        """
        Parse HTML table to extract rows and cells.
        
        Args:
            html: HTML table string
            
        Returns:
            List of rows (each row is a list of cell texts)
        """
        try:
            from html.parser import HTMLParser
            
            class TableParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.rows = []
                    self.current_row = []
                    self.current_cell = []
                    self.in_table = False
                    self.in_row = False
                    self.in_cell = False
                
                def handle_starttag(self, tag, attrs):
                    if tag == 'table':
                        self.in_table = True
                    elif tag == 'tr':
                        self.in_row = True
                        self.current_row = []
                    elif tag == 'td' or tag == 'th':
                        self.in_cell = True
                        self.current_cell = []
                
                def handle_endtag(self, tag):
                    if tag == 'table':
                        self.in_table = False
                    elif tag == 'tr':
                        if self.current_row:
                            self.rows.append(self.current_row)
                        self.in_row = False
                    elif tag == 'td' or tag == 'th':
                        cell_text = ''.join(self.current_cell).strip()
                        self.current_row.append(cell_text)
                        self.in_cell = False
                
                def handle_data(self, data):
                    if self.in_cell:
                        self.current_cell.append(data)
            
            parser = TableParser()
            parser.feed(html)
            return parser.rows
            
        except Exception as e:
            logger.warning(f"HTML parsing failed: {e}")
            return []
    
    def _extract_with_pdfplumber(
        self,
        pdf_path: str,
        page_num: int,
        table_bbox: List[float]
    ) -> Dict[str, Any]:
        """
        Extract table using pdfplumber on detected bbox.
        This is the BEST method for structured PDFs.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            table_bbox: Table bounding box [x1, y1, x2, y2]
            
        Returns:
            Dict with rows and method
        """
        try:
            with self.pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return {"rows": [], "method": "pdfplumber"}
                
                page = pdf.pages[page_num]
                
                # Crop page to table bbox
                x1, y1, x2, y2 = table_bbox
                cropped_page = page.within_bbox((x1, y1, x2, y2))
                
                # Extract tables from cropped region
                tables = cropped_page.extract_tables()
                
                if not tables:
                    return {"rows": [], "method": "pdfplumber"}
                
                # Use the first (largest) table
                table = tables[0]
                
                # Clean up table data
                rows = []
                for row in table:
                    if row and any(cell for cell in row if cell):  # Skip empty rows
                        cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(cleaned_row)
                
                logger.info(f"pdfplumber extracted {len(rows)} rows")
                return {"rows": rows, "method": "pdfplumber"}
                
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return {"rows": [], "method": "pdfplumber"}
    
    def _extract_table_with_ocr(self, table_img: np.ndarray) -> Dict[str, Any]:
        """
        Extract table using our existing OCR engine.
        Returns: {"rows": List[List[str]], "method": "ocr"}
        """
        try:
            # Import here to keep module import cheap if OCR not used
            from .ocr_engine import OCREngine
        except Exception:
            logger.warning("OCR engine not available")
            return {"rows": [], "method": "ocr"}

        try:
            ocr = OCREngine()
            # Use basic OCR to extract text, then try to structure it
            ocr_results = ocr.extract_text_from_image(table_img)
            
            # Simple heuristic: group text blocks into rows based on Y coordinates
            if not ocr_results:
                return {"rows": [], "method": "ocr"}
            
            # Sort by Y coordinate and group into rows
            sorted_blocks = sorted(ocr_results, key=lambda x: x['bbox'][1])
            rows = []
            current_row = []
            current_y = None
            
            for block in sorted_blocks:
                y = block['bbox'][1]
                if current_y is None or abs(y - current_y) < 10:  # Same row threshold
                    current_row.append(block['text'])
                    current_y = y
                else:
                    if current_row:
                        rows.append(current_row)
                    current_row = [block['text']]
                    current_y = y
            
            if current_row:
                rows.append(current_row)
            
            return {"rows": rows, "method": "ocr"}
        except Exception as e:
            logger.exception("OCR extraction failed: %s", e)
            return {"rows": [], "method": "ocr"}

    def _extract_table_with_lines(self, table_img: np.ndarray) -> Dict[str, Any]:
        """
        Line-based table detection using the table_lines module.
        Returns: {"rows": List[List[str]], "method": "lines"}
        """
        try:
            from .table_lines import detect_table_grid
        except Exception:
            logger.warning("Line-based extractor not available")
            return {"rows": [], "method": "lines"}

        try:
            rows = detect_table_grid(table_img)
            return {"rows": rows, "method": "lines"}
        except Exception as e:
            logger.exception("Line-based table extraction failed: %s", e)
            return {"rows": [], "method": "lines"}
    
    def _extract_grid_table(
        self,
        table_img: np.ndarray,
        doc_profile: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract grid table using detected lines and OCR
        
        Args:
            table_img: Table image
            doc_profile: Optional document profile for adaptive parameters
            
        Returns:
            Dict with rows and method
        """
        try:
            # Use the existing line-based extraction
            result = self._extract_table_with_lines(table_img)
            result["method"] = "grid"
            return result
        except Exception as e:
            logger.exception("Grid table extraction failed: %s", e)
            return {"rows": [], "method": "grid"}
    
    def _extract_borderless_table(
        self,
        table_img: np.ndarray,
        doc_profile: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract borderless table using whitespace analysis and alignment detection
        
        Args:
            table_img: Table image
            doc_profile: Optional document profile
            
        Returns:
            Dict with rows and method
        """
        try:
            # Import OCR engine
            from .ocr_engine import OCREngine
            
            ocr = OCREngine()
            ocr_results = ocr.extract_text_from_image(table_img)
            
            if not ocr_results:
                return {"rows": [], "method": "borderless"}
            
            # Analyze whitespace and alignment
            rows = self._group_by_alignment(ocr_results)
            
            return {"rows": rows, "method": "borderless"}
            
        except Exception as e:
            logger.exception("Borderless table extraction failed: %s", e)
            return {"rows": [], "method": "borderless"}
    
    def _group_by_alignment(self, ocr_results: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Group OCR results into table rows and columns based on alignment
        
        Args:
            ocr_results: List of OCR results with bbox and text
            
        Returns:
            List of rows (each row is a list of cell texts)
        """
        if not ocr_results:
            return []
        
        # Sort by Y coordinate
        sorted_results = sorted(ocr_results, key=lambda x: x['bbox'][1])
        
        # Group into rows based on Y proximity
        rows = []
        current_row = [sorted_results[0]]
        current_y = sorted_results[0]['bbox'][1]
        
        for result in sorted_results[1:]:
            y = result['bbox'][1]
            
            # Same row if Y difference is small
            if abs(y - current_y) < 10:
                current_row.append(result)
            else:
                # New row
                if current_row:
                    # Sort row by X coordinate
                    current_row.sort(key=lambda x: x['bbox'][0])
                    row_texts = [r['text'] for r in current_row]
                    rows.append(row_texts)
                current_row = [result]
                current_y = y
        
        # Add last row
        if current_row:
            current_row.sort(key=lambda x: x['bbox'][0])
            row_texts = [r['text'] for r in current_row]
            rows.append(row_texts)
        
        return rows

    def _refine_with_ollama(self, rows: List[List[str]]) -> List[List[str]]:
        """
        Refine table rows using Ollama (Tier 2.5).
        Fixes fragmentation, alignment, and formatting issues.
        """
        if not self.ollama_cleaner or not rows:
            return rows
            
        try:
            # Convert rows to string representation
            table_text = ""
            for row in rows:
                table_text += " | ".join([str(cell) for cell in row]) + "\n"
            
            if len(table_text) > 4000:  # Skip very large tables to avoid context limit
                return rows

            prompt = f"""Refine this extracted table data into a clean JSON list of lists.
Fix fragmentation, merge split cells, and ensure correct column alignment.
Return ONLY the JSON object, no markdown formatting.

Input Table Data:
{table_text}
"""
            
            response = self.ollama_cleaner.client.generate(
                model="mistral",  # Use faster model for speed
                prompt=prompt,
                system="You are a table structure expert. Output valid JSON only.",
                temperature=0.1,
                stream=False,
                timeout=60  # 60s timeout for table refinement
            )
            
            # Extract JSON from response
            import json
            import re
            
            # Find JSON block
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                refined_rows = json.loads(json_str)
                if isinstance(refined_rows, list) and len(refined_rows) > 0:
                    logger.info(f"Ollama refined table: {len(rows)} -> {len(refined_rows)} rows")
                    return refined_rows
            
            return rows
            
        except Exception as e:
            logger.warning(f"Ollama table refinement failed: {e}")
            return rows
