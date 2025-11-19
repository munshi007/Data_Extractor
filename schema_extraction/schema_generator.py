"""
Automated Schema Generator
Adapted from: /local/home/munshi/FINAL_TRY/data-extractor/schema_recommender.py

Generates JSON schemas automatically from markdown content using LLM analysis.
Uses iterative chunk-based processing to build comprehensive schemas.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    from pipeline.ollama.client import OllamaClient
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from pipeline.ollama.client import OllamaClient

from .chunking.chunker import MarkdownChunker

logger = logging.getLogger(__name__)


@dataclass
class SchemaGenerationResult:
    """Result of automatic schema generation"""
    schema: Dict[str, Any]
    analysis: Dict[str, Any]
    metadata: Dict[str, Any]


class AutoSchemaGenerator:
    """
    Automatically generates JSON schemas from content using LLM analysis.
    Inspired by the data-extractor schema_recommender.py
    """
    
    def __init__(self, host: str = "http://127.0.0.1:11434"):
        self.client = OllamaClient(host=host, timeout=600)
        self.model = "mistral"
        self.chunker = MarkdownChunker(max_tokens=2000)
        self.base_schema: Optional[Dict[str, Any]] = None
        self.processing_notes: List[str] = []
    
    def generate_schema(self, markdown_content: str) -> SchemaGenerationResult:
        """
        Generate a comprehensive JSON schema from markdown content.
        
        Process:
        1. Chunk the content
        2. Analyze first chunk to create base schema
        3. Iteratively refine schema with subsequent chunks
        4. Return final comprehensive schema
        """
        start_time = datetime.now()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔧 Auto Schema Generator: Starting analysis...", flush=True)
        
        # Chunk the content
        chunks = self.chunker.chunk(markdown_content)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Analyzing {len(chunks)} chunks...", flush=True)
        
        # Process first chunk to establish base schema
        self.base_schema = self._analyze_first_chunk(chunks[0])
        
        # Refine schema with remaining chunks
        for i, chunk in enumerate(chunks[1:], 1):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Refining schema with chunk {i+1}/{len(chunks)}...", flush=True)
            self._refine_schema_with_chunk(chunk)
        
        # Finalize and validate schema
        final_schema = self._finalize_schema()
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Schema generation complete in {duration:.1f}s", flush=True)
        
        return SchemaGenerationResult(
            schema=final_schema,
            analysis={
                "chunks_analyzed": len(chunks),
                "processing_notes": self.processing_notes
            },
            metadata={
                "duration_seconds": duration,
                "model_used": self.model
            }
        )
    
    def _analyze_first_chunk(self, chunk: str) -> Dict[str, Any]:
        """Analyze first chunk to create initial schema structure"""
        
        prompt = f"""
        You are a Schema Generation Expert. Analyze this document excerpt and create an initial JSON schema.
        
        Task: Create a comprehensive JSON schema that captures the structure and data types in this content.
        
        Document Content:
        {chunk}
        
        Guidelines:
        1. Identify all data entities (objects, lists, key-value pairs)
        2. Determine appropriate data types (string, number, boolean, array, object)
        3. Note required vs optional fields
        4. Include descriptions for each field
        5. Provide example values where applicable
        
        Output Format - Return ONLY valid JSON matching this structure:
        {{
            "title": "Descriptive schema title",
            "description": "What this data represents",
            "type": "object",
            "properties": {{
                "field_name": {{
                    "type": "string|number|boolean|array|object",
                    "description": "Clear description of this field",
                    "examples": ["example1", "example2"]
                }}
            }},
            "required": ["list", "of", "required", "fields"]
        }}
        
        Generate the schema:
        """
        
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            temperature=0.3,
            num_predict=3072,
            system="You are a JSON schema expert. Output only valid JSON schemas."
        )
        
        schema = self._parse_json_response(response)
        self.processing_notes.append(f"Initial schema from first chunk with {len(schema.get('properties', {}))} properties")
        return schema
    
    def _refine_schema_with_chunk(self, chunk: str):
        """Refine existing schema based on new chunk content"""
        
        current_schema_str = json.dumps(self.base_schema, indent=2)
        
        prompt = f"""
        You are refining an existing JSON schema. A new chunk of content has been provided.
        
        Current Schema:
        {current_schema_str}
        
        New Content Chunk:
        {chunk}
        
        Task: Determine if this new content requires schema modifications.
        
        Rules:
        1. If content fits existing schema → Return: {{"status": "NO_CHANGES_NEEDED"}}
        2. If new fields discovered → Add them to properties
        3. If new data types found → Update type definitions
        4. If new examples found → Add to existing examples (max 3 per field)
        5. NEVER duplicate existing properties
        6. PRESERVE all existing properties
        
        Output Format:
        If changes needed:
        {{
            "status": "SCHEMA_UPDATED",
            "changes": {{
                "new_properties": {{"field_name": {{...field_definition...}}}},
                "updated_properties": {{"field_name": {{...updated_definition...}}}}
            }},
            "reasoning": "Why these changes were made"
        }}
        
        If no changes needed:
        {{
            "status": "NO_CHANGES_NEEDED"
        }}
        
        Analyze and respond:
        """
        
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            temperature=0.2,
            num_predict=2048,
            system="You output only valid JSON."
        )
        
        update = self._parse_json_response(response)
        
        if update.get("status") == "SCHEMA_UPDATED":
            self._apply_schema_updates(update.get("changes", {}))
            self.processing_notes.append(f"Schema updated: {update.get('reasoning', 'No reason provided')}")
        else:
            self.processing_notes.append("No schema changes needed for this chunk")
    
    def _apply_schema_updates(self, changes: Dict[str, Any]):
        """Apply schema updates from chunk analysis"""
        
        # Add new properties
        if "new_properties" in changes:
            for prop_name, prop_def in changes["new_properties"].items():
                if prop_name not in self.base_schema.get("properties", {}):
                    if "properties" not in self.base_schema:
                        self.base_schema["properties"] = {}
                    self.base_schema["properties"][prop_name] = prop_def
        
        # Update existing properties
        if "updated_properties" in changes:
            for prop_name, prop_updates in changes["updated_properties"].items():
                if prop_name in self.base_schema.get("properties", {}):
                    # Merge updates
                    existing = self.base_schema["properties"][prop_name]
                    
                    # Update examples (keep max 3)
                    if "examples" in prop_updates:
                        existing_examples = existing.get("examples", [])
                        new_examples = prop_updates["examples"]
                        combined = list(set(existing_examples + new_examples))
                        existing["examples"] = combined[:3]
                    
                    # Update other fields
                    for key, value in prop_updates.items():
                        if key != "examples":
                            existing[key] = value
    
    def _finalize_schema(self) -> Dict[str, Any]:
        """Finalize and validate the generated schema"""
        
        # Ensure schema has all required JSON Schema fields
        final_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            **self.base_schema
        }
        
        # Ensure required fields list exists
        if "required" not in final_schema:
            final_schema["required"] = []
        
        # Add metadata
        final_schema["$comment"] = f"Auto-generated schema at {datetime.now().isoformat()}"
        
        return final_schema
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks"""
        try:
            # Remove markdown code blocks
            cleaned = response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
            # Find JSON object
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1:
                cleaned = cleaned[start:end+1]
            
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            # Return minimal valid schema as fallback
            return {
                "title": "Fallback Schema",
                "description": "Schema generation encountered an error",
                "type": "object",
                "properties": {},
                "required": []
            }
