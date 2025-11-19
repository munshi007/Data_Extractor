
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)


def save_image(img, out_path: Path, msg: str = "") -> None:
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


class PageProcessor:
    """
    Minimal wiring to save region snapshots (Figure/Table).
    Call `attach_region_snapshots(page_image, regions)` after layout detection.
    """

    def __init__(self, output_paths: Dict[str, Path]):
        self.output_paths = output_paths

    def attach_region_snapshots(self, page_image: np.ndarray, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Use separate folders for figures and tables
        figures_dir = self.output_paths.get("figures", Path("Output/extracted_figures"))
        tables_dir = self.output_paths.get("tables", Path("Output/extracted_tables"))
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        figure_count = 1
        for r in regions:
            r_type = str(r.get("type", "")).lower()
            if r_type in ["figure", "table"]:
                bbox = r.get("bbox")
                if not bbox or page_image is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
                crop = page_image[y1:y2, x1:x2]
                
                # Save to appropriate folder
                if r_type == "figure":
                    fname = f"figure_{figure_count:03d}.png"
                    fpath = figures_dir / fname
                    figure_count += 1
                else:  # table
                    # Tables already saved by table extractor, just reference
                    fname = f"region_{r.get('region_id', 'unknown')}.png"
                    fpath = tables_dir / fname
                
                save_image(crop, fpath, f"{r_type} snapshot: {fname}")
                r["snapshot_image"] = str(fpath)
        return regions
