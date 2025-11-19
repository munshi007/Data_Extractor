"""
Table Lines Detection Module
Extracted from the original enhanced_pdf_pipeline.py
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


def detect_table_grid(table_image: np.ndarray) -> List[List[str]]:
    """
    Detect table grid using line detection and extract cell contents.
    
    Args:
        table_image: Input table image as numpy array
        
    Returns:
        List of rows, where each row is a list of cell text strings
    """
    if cv2 is None:
        logger.warning("OpenCV not available for line detection")
        return []
    
    try:
        # Convert to grayscale if needed
        if len(table_image.shape) == 3:
            gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = table_image
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find line coordinates using HoughLinesP
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        # Check if we have sufficient line structure
        if h_lines is None or v_lines is None or len(h_lines) < 2 or len(v_lines) < 2:
            logger.info("Insufficient line structure detected")
            return []
        
        # Extract table using detected lines as anchors
        return _extract_with_line_anchors(table_image, h_lines, v_lines)
        
    except Exception as e:
        logger.error(f"Error in table grid detection: {e}")
        return []


def _extract_with_line_anchors(table_image: np.ndarray, h_lines, v_lines) -> List[List[str]]:
    """Extract table using detected lines as precise anchors for cell boundaries."""
    try:
        # Sort lines by position and clean up duplicates
        h_coords = sorted(set([line[0][1] for line in h_lines]))  # y-coordinates
        v_coords = sorted(set([line[0][0] for line in v_lines]))  # x-coordinates
        
        # Limit table size to prevent massive output
        max_rows = min(len(h_coords) - 1, 20)
        max_cols = min(len(v_coords) - 1, 10)
        
        rows = []
        
        if max_rows > 0 and max_cols > 0:
            # Extract text from each cell defined by line intersections
            for i in range(max_rows):
                row = []
                for j in range(max_cols):
                    # Define cell boundaries using line coordinates
                    x1, x2 = v_coords[j], v_coords[j + 1]
                    y1, y2 = h_coords[i], h_coords[i + 1]
                    
                    # Add small padding to avoid line artifacts
                    padding = 2
                    x1, y1 = max(0, x1 + padding), max(0, y1 + padding)
                    x2 = min(table_image.shape[1], x2 - padding)
                    y2 = min(table_image.shape[0], y2 - padding)
                    
                    # For now, create placeholder cell content
                    # In a full implementation, this would use OCR on the cell region
                    cell_region = table_image[y1:y2, x1:x2]
                    if cell_region.size > 0:
                        # Simple heuristic: if cell has content (not mostly white)
                        if len(cell_region.shape) == 3:
                            gray_cell = cv2.cvtColor(cell_region, cv2.COLOR_RGB2GRAY)
                        else:
                            gray_cell = cell_region
                        
                        # Check if cell has significant content (not empty)
                        mean_intensity = np.mean(gray_cell)
                        if mean_intensity < 240:  # Not mostly white
                            cell_text = f"Cell_{i}_{j}"
                        else:
                            cell_text = ""
                    else:
                        cell_text = ""
                    
                    row.append(cell_text)
                
                # Only add rows with some content
                if any(cell.strip() for cell in row):
                    rows.append(row)
        
        logger.info(f"Extracted {len(rows)} rows with line anchors")
        return rows
        
    except Exception as e:
        logger.error(f"Error in line-anchored extraction: {e}")
        return []