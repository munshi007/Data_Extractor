"""
Reading Order Resolver - Determine correct reading order using adaptive algorithms
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for LayoutLMv3 availability
LAYOUTLM_AVAILABLE = False
try:
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    LAYOUTLM_AVAILABLE = True
except ImportError:
    logger.info("LayoutLMv3 not available, will use adaptive XY-Cut")


class ReadingOrderResolver:
    """Resolve reading order using adaptive XY-Cut with column detection"""
    
    def __init__(self, use_layoutlm: bool = False):
        """
        Initialize reading order resolver
        
        Args:
            use_layoutlm: Whether to use LayoutLMv3 if available
        """
        self.use_layoutlm = use_layoutlm and LAYOUTLM_AVAILABLE
        self.layoutlm_model = None
        self.layoutlm_processor = None
        
        if self.use_layoutlm:
            self._initialize_layoutlm()
    
    def _initialize_layoutlm(self):
        """Initialize LayoutLMv3 model for reading order"""
        try:
            logger.info("Initializing LayoutLMv3 for reading order...")
            # Note: This would require a fine-tuned model for reading order
            # For now, we'll keep it as a placeholder
            # self.layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained("...")
            # self.layoutlm_processor = LayoutLMv3Processor.from_pretrained("...")
            logger.info("LayoutLMv3 initialization skipped (requires fine-tuned model)")
            self.use_layoutlm = False
        except Exception as e:
            logger.warning(f"Failed to initialize LayoutLMv3: {e}")
            self.use_layoutlm = False
    
    def order_regions(
        self, 
        regions: List[Dict[str, Any]], 
        page_image: Optional[np.ndarray] = None,
        doc_profile: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Order regions by reading order using hierarchy of methods
        
        Args:
            regions: List of detected regions with bboxes
            page_image: Optional page image for column detection
            doc_profile: Optional document profile with thresholds
            
        Returns:
            Ordered list of regions
        """
        if not regions:
            return []
        
        # Strategy 1: LayoutLMv3 (if available and enabled)
        if self.use_layoutlm and self.layoutlm_model is not None:
            try:
                return self._order_with_layoutlm(regions, page_image)
            except Exception as e:
                logger.warning(f"LayoutLMv3 ordering failed: {e}, falling back to XY-Cut")
        
        # Strategy 2: Detect reading direction
        reading_direction = self.detect_reading_direction(regions, doc_profile)
        
        # Strategy 3: Adaptive XY-Cut with column detection
        dividers = []
        if page_image is not None:
            dividers = self.detect_vertical_dividers(page_image)
        
        if dividers:
            # Multi-column layout
            logger.info(f"Detected {len(dividers)} column dividers")
            return self._order_multi_column(regions, dividers, reading_direction)
        else:
            # Single column
            return self._order_single_column(regions, reading_direction)
    
    def _order_with_layoutlm(
        self,
        regions: List[Dict[str, Any]],
        page_image: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Order regions using LayoutLMv3 transformer model
        
        Note: This requires a fine-tuned LayoutLMv3 model for reading order prediction
        """
        logger.info("Using LayoutLMv3 for reading order")
        # Placeholder for LayoutLMv3 implementation
        # Would require:
        # 1. Prepare input (image + bboxes + text)
        # 2. Run model inference
        # 3. Extract reading order from predictions
        # For now, fall back to XY-Cut
        raise NotImplementedError("LayoutLMv3 reading order requires fine-tuned model")
    
    def detect_reading_direction(
        self,
        regions: List[Dict[str, Any]],
        doc_profile: Optional[Any] = None
    ) -> str:
        """
        Detect reading direction (LTR, RTL, vertical) from document properties
        
        Args:
            regions: List of regions
            doc_profile: Optional document profile
            
        Returns:
            Reading direction: 'ltr', 'rtl', or 'vertical'
        """
        # For now, default to LTR (left-to-right)
        # Could be enhanced with:
        # - Language detection from text content
        # - Document metadata analysis
        # - Text direction from OCR results
        return 'ltr'
    
    def detect_vertical_dividers(self, image: np.ndarray) -> List[float]:
        """
        Detect vertical lines that separate columns using Hough transform
        
        Args:
            image: Page image as numpy array
            
        Returns:
            List of X coordinates for column dividers
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 
                threshold=100,
                minLineLength=image.shape[0] * 0.5  # At least 50% of page height
            )
            
            if lines is None:
                return []
            
            # Filter vertical lines (angle close to 90 degrees)
            vertical_x_coords = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if 85 <= angle <= 95:  # Nearly vertical
                    x_center = (x1 + x2) / 2
                    vertical_x_coords.append(x_center)
            
            if not vertical_x_coords:
                return []
            
            # Cluster X coordinates to find column boundaries
            from sklearn.cluster import DBSCAN
            X = np.array(vertical_x_coords).reshape(-1, 1)
            clustering = DBSCAN(eps=20, min_samples=2).fit(X)
            
            dividers = []
            for label in set(clustering.labels_):
                if label != -1:  # Ignore noise
                    cluster_points = X[clustering.labels_ == label]
                    dividers.append(float(cluster_points.mean()))
            
            return sorted(dividers)
            
        except Exception as e:
            logger.warning(f"Failed to detect vertical dividers: {e}")
            return []
    
    def _order_single_column(
        self,
        regions: List[Dict[str, Any]],
        reading_direction: str = 'ltr'
    ) -> List[Dict[str, Any]]:
        """
        Order regions in single column layout
        
        Args:
            regions: List of regions
            reading_direction: 'ltr', 'rtl', or 'vertical'
            
        Returns:
            Ordered regions
        """
        if reading_direction == 'rtl':
            # Right-to-left: sort by Y then X descending
            return sorted(regions, key=lambda r: (r['bbox'][1], -r['bbox'][0]))
        elif reading_direction == 'vertical':
            # Vertical: sort by X then Y
            return sorted(regions, key=lambda r: (r['bbox'][0], r['bbox'][1]))
        else:
            # Left-to-right (default): sort by Y then X
            return sorted(regions, key=lambda r: (r['bbox'][1], r['bbox'][0]))
    
    def _order_multi_column(
        self, 
        regions: List[Dict[str, Any]], 
        dividers: List[float],
        reading_direction: str = 'ltr'
    ) -> List[Dict[str, Any]]:
        """
        Order regions in multi-column layout
        
        Args:
            regions: List of regions
            dividers: X coordinates of column dividers
            reading_direction: 'ltr', 'rtl', or 'vertical'
            
        Returns:
            Ordered regions
        """
        # Create column boundaries
        columns = []
        prev_x = 0
        for divider in dividers:
            columns.append((prev_x, divider))
            prev_x = divider
        columns.append((prev_x, float('inf')))  # Last column
        
        # Reverse column order for RTL
        if reading_direction == 'rtl':
            columns = list(reversed(columns))
        
        # Assign regions to columns
        ordered = []
        for col_start, col_end in columns:
            col_regions = []
            for region in regions:
                bbox = region['bbox']
                region_center_x = (bbox[0] + bbox[2]) / 2
                
                # Handle reversed columns for RTL
                if reading_direction == 'rtl':
                    if col_end <= region_center_x < col_start:
                        col_regions.append(region)
                else:
                    if col_start <= region_center_x < col_end:
                        col_regions.append(region)
            
            # Sort regions within column by Y
            col_regions.sort(key=lambda r: r['bbox'][1])
            ordered.extend(col_regions)
        
        return ordered
