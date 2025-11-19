import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from schema_extraction.converter import SchemaConverter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("schema_runner")

def main():
    # Paths
    base_dir = Path("/local/home/munshi/SECOND_TRY_2_Testing")
    markdown_path = base_dir / "Output/extracted_content.md"
    schema_path = base_dir / "schema.json"
    output_path = base_dir / "Output/structured_output.json"
    
    # Check markdown file
    if not markdown_path.exists():
        logger.error(f"Markdown file not found: {markdown_path}")
        return
        
    # Load content
    logger.info("Loading content...")
    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Load schema if provided, otherwise auto-generate
    schema = None
    if schema_path.exists():
        logger.info(f"Loading schema from {schema_path}")
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
    else:
        logger.info("No schema file found - will auto-generate schema from content")
        
    # Initialize Converter
    converter = SchemaConverter()
    
    # Run Conversion (with or without schema)
    logger.info("Running schema conversion...")
    result = converter.convert(markdown_content, schema)
    
    # Save Output
    logger.info(f"Saving result to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.data, f, indent=2, ensure_ascii=False)
    
    # Save metadata including auto-generated schema if applicable
    metadata_path = base_dir / "Output/extraction_metadata.json"
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(result.metadata, f, indent=2, ensure_ascii=False)
        
    # Save Metadata (The "Thinking" Process)
    meta_path = base_dir / "Output/conversion_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(result.metadata, f, indent=2, ensure_ascii=False)
        
    logger.info("Done!")
    logger.info(f"Valid: {result.is_valid}")
    if not result.is_valid:
        logger.warning(f"Errors: {result.validation_errors}")

if __name__ == "__main__":
    main()
