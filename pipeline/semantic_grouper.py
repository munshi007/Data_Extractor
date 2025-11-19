"""
Semantic Text Grouper - Group text by semantic meaning and spatial proximity
With Ollama embeddings support (Tier 1 enhancement)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Check for sentence-transformers availability
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.info("sentence-transformers not available, semantic grouping will use spatial-only")

# Check for Ollama embeddings availability
OLLAMA_AVAILABLE = False
try:
    from .ollama.embedder import OllamaEmbedder
    OLLAMA_AVAILABLE = True
except ImportError:
    logger.info("Ollama not available, will use sentence-transformers if available")


@dataclass
class GroupingWeights:
    """Weights for combining spatial and semantic similarity"""
    spatial: float = 0.6
    semantic: float = 0.4


class SemanticTextGrouper:
    """Group text regions by semantic meaning and spatial proximity"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_ollama: bool = True):
        """
        Initialize semantic text grouper
        
        Args:
            model_name: Name of sentence-transformers model to use
            use_ollama: Try to use Ollama embeddings first
        """
        self.embedder = None
        self.model_name = model_name
        self.use_ollama = use_ollama
        
        # Try Ollama first
        if use_ollama and OLLAMA_AVAILABLE:
            self._initialize_ollama_embedder()
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialize_embedder()
    
    def _initialize_ollama_embedder(self):
        """Initialize Ollama embedder (Tier 1)"""
        try:
            logger.info("Initializing Ollama embedder for semantic grouping (Tier 1)")
            self.embedder = OllamaEmbedder(model="nomic-embed-text")
            logger.info("Ollama embedder initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama embedder, falling back to sentence-transformers: {e}")
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._initialize_embedder()
    
    def _initialize_embedder(self):
        """Initialize sentence embedding model"""
        try:
            logger.info(f"Loading sentence embedding model: {self.model_name}")
            self.embedder = SentenceTransformer(self.model_name)
            logger.info("Sentence embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedder = None
    
    def group_paragraphs(
        self,
        text_regions: List[Dict[str, Any]],
        doc_profile: Optional[Any] = None,
        weights: Optional[GroupingWeights] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Group text regions into paragraphs using semantic similarity and spatial proximity
        
        Args:
            text_regions: List of text regions with 'text' and 'bbox' fields
            doc_profile: Optional DocumentProfile with adaptive thresholds
            weights: Optional weights for spatial vs semantic similarity
            
        Returns:
            List of paragraph groups (each group is a list of regions)
        """
        if not text_regions:
            return []
        
        if weights is None:
            weights = GroupingWeights()
        
        # Filter regions with text
        valid_regions = [r for r in text_regions if r.get('text', '').strip()]
        if not valid_regions:
            return []
        
        # Compute combined similarity matrix
        similarity_matrix = self._compute_combined_similarity(
            valid_regions, weights, doc_profile
        )
        
        # Determine threshold
        if doc_profile and hasattr(doc_profile, 'thresholds'):
            threshold = doc_profile.thresholds.merge_distance / 100.0  # Normalize
        else:
            threshold = 0.5  # Default
        
        # Cluster regions
        groups = self._cluster_regions(valid_regions, similarity_matrix, threshold)
        
        logger.info(f"Grouped {len(valid_regions)} text regions into {len(groups)} paragraphs")
        return groups
    
    def _compute_combined_similarity(
        self,
        regions: List[Dict[str, Any]],
        weights: GroupingWeights,
        doc_profile: Optional[Any]
    ) -> np.ndarray:
        """
        Compute combined similarity matrix from spatial and semantic features
        
        Args:
            regions: List of text regions
            weights: Weights for combining similarities
            doc_profile: Optional document profile
            
        Returns:
            Combined similarity matrix (n x n)
        """
        n = len(regions)
        
        # Compute spatial proximity matrix
        spatial_matrix = self._compute_spatial_proximity(regions, doc_profile)
        
        # Compute semantic similarity matrix
        if self.embedder is not None:
            semantic_matrix = self._compute_semantic_similarity(regions)
        else:
            # Fallback: use spatial only
            semantic_matrix = np.zeros((n, n))
            weights = GroupingWeights(spatial=1.0, semantic=0.0)
        
        # Combine matrices
        combined = weights.spatial * spatial_matrix + weights.semantic * semantic_matrix
        
        return combined
    
    def _compute_spatial_proximity(
        self,
        regions: List[Dict[str, Any]],
        doc_profile: Optional[Any]
    ) -> np.ndarray:
        """
        Compute spatial proximity matrix (normalized by document scale)
        
        Args:
            regions: List of text regions
            doc_profile: Optional document profile
            
        Returns:
            Spatial proximity matrix (n x n), values in [0, 1]
        """
        n = len(regions)
        proximity_matrix = np.zeros((n, n))
        
        # Get document scale for normalization
        if doc_profile and hasattr(doc_profile, 'resolution'):
            page_height = doc_profile.resolution.height
        else:
            # Estimate from regions
            max_y = max(r['bbox'][3] for r in regions)
            page_height = max_y if max_y > 0 else 1000
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self._compute_distance(regions[i]['bbox'], regions[j]['bbox'])
                
                # Normalize by page height and convert to similarity (closer = higher)
                normalized_distance = distance / page_height
                similarity = np.exp(-normalized_distance * 5)  # Exponential decay
                
                proximity_matrix[i, j] = similarity
                proximity_matrix[j, i] = similarity
        
        # Set diagonal to 1 (self-similarity)
        np.fill_diagonal(proximity_matrix, 1.0)
        
        return proximity_matrix
    
    def _compute_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Compute distance between two bounding boxes
        Uses vertical distance primarily (for paragraph grouping)
        """
        # Get centers
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        # Vertical distance (primary for paragraph grouping)
        vertical_dist = abs(center2_y - center1_y)
        
        # Horizontal distance (secondary, penalize if far apart horizontally)
        horizontal_dist = abs(center2_x - center1_x)
        
        # Combined distance (weight vertical more heavily)
        distance = vertical_dist + 0.3 * horizontal_dist
        
        return distance
    
    def _compute_semantic_similarity(
        self,
        regions: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Compute semantic similarity matrix using sentence embeddings
        
        Args:
            regions: List of text regions
            
        Returns:
            Semantic similarity matrix (n x n), values in [0, 1]
        """
        # Extract text
        texts = [r.get('text', '') for r in regions]
        
        # Compute embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        # Ensure values are in [0, 1]
        similarity_matrix = (similarity_matrix + 1) / 2
        
        return similarity_matrix
    
    def _cluster_regions(
        self,
        regions: List[Dict[str, Any]],
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster regions based on similarity matrix using agglomerative approach
        
        Args:
            regions: List of text regions
            similarity_matrix: Similarity matrix
            threshold: Similarity threshold for grouping
            
        Returns:
            List of region groups
        """
        n = len(regions)
        
        # Initialize each region as its own cluster
        clusters = [[i] for i in range(n)]
        
        # Agglomerative clustering
        merged = True
        while merged:
            merged = False
            best_sim = -1
            best_pair = None
            
            # Find best pair to merge
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Compute average similarity between clusters
                    sim = self._cluster_similarity(
                        clusters[i], clusters[j], similarity_matrix
                    )
                    
                    if sim > threshold and sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)
            
            # Merge best pair
            if best_pair is not None:
                i, j = best_pair
                clusters[i].extend(clusters[j])
                del clusters[j]
                merged = True
        
        # Convert cluster indices to region groups
        groups = []
        for cluster in clusters:
            group = [regions[idx] for idx in cluster]
            # Sort by Y coordinate within group
            group.sort(key=lambda r: r['bbox'][1])
            groups.append(group)
        
        # Sort groups by Y coordinate of first region
        groups.sort(key=lambda g: g[0]['bbox'][1] if g else 0)
        
        return groups
    
    def _cluster_similarity(
        self,
        cluster1: List[int],
        cluster2: List[int],
        similarity_matrix: np.ndarray
    ) -> float:
        """
        Compute average similarity between two clusters
        
        Args:
            cluster1: List of region indices in cluster 1
            cluster2: List of region indices in cluster 2
            similarity_matrix: Similarity matrix
            
        Returns:
            Average similarity between clusters
        """
        similarities = []
        for i in cluster1:
            for j in cluster2:
                similarities.append(similarity_matrix[i, j])
        
        return np.mean(similarities) if similarities else 0.0
    
    def detect_list_structures(
        self,
        text_regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect list structures using pattern recognition and indentation
        
        Args:
            text_regions: List of text regions
            
        Returns:
            Text regions with 'is_list_item' flag added
        """
        import re
        
        # List patterns
        bullet_pattern = r'^\s*[•\-\*\+◦▪▫]\s+'
        numbered_pattern = r'^\s*\d+[\.\)]\s+'
        letter_pattern = r'^\s*[a-zA-Z][\.\)]\s+'
        
        for region in text_regions:
            text = region.get('text', '')
            
            # Check for list patterns
            is_bullet = bool(re.match(bullet_pattern, text))
            is_numbered = bool(re.match(numbered_pattern, text))
            is_letter = bool(re.match(letter_pattern, text))
            
            if is_bullet or is_numbered or is_letter:
                region['is_list_item'] = True
                region['list_type'] = 'bullet' if is_bullet else ('numbered' if is_numbered else 'letter')
            else:
                region['is_list_item'] = False
        
        return text_regions
    
    def detect_footnotes_and_captions(
        self,
        text_regions: List[Dict[str, Any]],
        doc_profile: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect footnotes and captions using position and font analysis
        
        Args:
            text_regions: List of text regions
            doc_profile: Optional document profile
            
        Returns:
            Text regions with 'is_footnote' or 'is_caption' flags added
        """
        import re
        
        if not text_regions:
            return text_regions
        
        # Get page dimensions
        if doc_profile and hasattr(doc_profile, 'resolution'):
            page_height = doc_profile.resolution.height
        else:
            max_y = max(r['bbox'][3] for r in text_regions)
            page_height = max_y if max_y > 0 else 1000
        
        # Caption patterns
        caption_pattern = r'^(Figure|Fig\.|Table|Diagram|Chart|Image)\s+\d+'
        
        for region in text_regions:
            text = region.get('text', '')
            bbox = region.get('bbox', [0, 0, 0, 0])
            
            # Footnote detection: bottom 10% of page + small font
            y_position = bbox[1]
            is_bottom = y_position > (page_height * 0.9)
            
            font_size = region.get('font_size', 12)
            is_small_font = font_size < 10
            
            if is_bottom and is_small_font:
                region['is_footnote'] = True
            else:
                region['is_footnote'] = False
            
            # Caption detection: pattern matching
            if re.match(caption_pattern, text.strip(), re.IGNORECASE):
                region['is_caption'] = True
            else:
                region['is_caption'] = False
        
        return text_regions
