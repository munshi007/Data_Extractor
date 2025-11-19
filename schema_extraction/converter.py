"""
OneKE-Inspired Multi-Agent Schema Extraction System
Adapted from: "OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System"
Paper: arXiv:2412.20005v2 [cs.CL]

Architecture:
1. Schema Agent - Analyzes schema and preprocesses data
2. Extraction Agent - Performs knowledge extraction with case retrieval
3. Reflection Agent - Debugs and corrects errors with self-reflection

Key Features:
- Multi-agent collaborative extraction
- Case-based learning (correct/bad case repositories)
- Self-reflection and error correction
- Iterative chunk processing for long documents
"""

import json
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import from the existing pipeline infrastructure
try:
    from pipeline.ollama.client import OllamaClient
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from pipeline.ollama.client import OllamaClient

from .chunking.chunker import MarkdownChunker
from .schema_generator import AutoSchemaGenerator, SchemaGenerationResult

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    is_valid: bool
    validation_errors: List[str]
    extraction_trace: List[Dict[str, Any]]  # OneKE-inspired: track agent interactions

@dataclass
class CaseExample:
    """Stores extraction cases for learning (OneKE Case Repository)"""
    input_text: str
    schema: Dict[str, Any]
    output: Dict[str, Any]
    reasoning: str
    is_correct: bool

class SchemaConverter:
    """
    OneKE-Inspired Multi-Agent Schema Extraction System
    
    Agents:
    1. Schema Agent - Preprocesses and analyzes schema
    2. Extraction Agent - Extracts with case-based learning
    3. Reflection Agent - Self-reflects and corrects errors
    """
    
    def __init__(self, host: str = "http://127.0.0.1:11434"):
        self.client = OllamaClient(host=host, timeout=600)
        
        # Agent models (can be different models for different agents)
        self.schema_agent_model = "mistral"      # Schema Agent
        self.extraction_agent_model = "mistral"  # Extraction Agent
        self.reflection_agent_model = "mistral"  # Reflection Agent
        
        self.max_retries = 3
        self.chunker = MarkdownChunker(max_tokens=2000)
        
        # Auto Schema Generator
        self.schema_generator = AutoSchemaGenerator(host=host)
        
        # OneKE-inspired: Case Repository (in-memory for now)
        self.correct_cases: List[CaseExample] = []
        self.bad_cases: List[CaseExample] = []
        self.extraction_trace: List[Dict[str, Any]] = []

    def convert(self, markdown_content: str, schema: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        start_time = datetime.now()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting OneKE-inspired multi-agent extraction...", flush=True)
        
        # Auto-generate schema if not provided
        if schema is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📋 No schema provided - generating automatically...", flush=True)
            schema_result = self.schema_generator.generate_schema(markdown_content)
            schema = schema_result.schema
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Auto-generated schema with {len(schema.get('properties', {}))} properties", flush=True)
        
        # Agent 1: Schema Agent - Preprocess and analyze
        chunks, schema_analysis = self._schema_agent(markdown_content, schema)
        
        # Agent 2: Extraction Agent - Extract with case retrieval
        current_data = {}
        extraction_logs = []
        
        for i, chunk in enumerate(chunks):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📄 Processing Chunk {i+1}/{len(chunks)}...", flush=True)
            current_data, log = self._extraction_agent(chunk, current_data, schema, schema_analysis)
            extraction_logs.append(log)
        
        # Agent 3: Reflection Agent - Self-reflect and correct
        final_data, reflection_log = self._reflection_agent(current_data, schema, markdown_content)
        
        # Validation
        is_valid, errors = self._validate_output(final_data, schema)
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Extraction complete in {duration:.1f}s", flush=True)
        
        return ExtractionResult(
            data=final_data,
            metadata={
                "duration_seconds": duration,
                "chunks_processed": len(chunks),
                "schema_analysis": schema_analysis,
                "extraction_logs": extraction_logs,
                "reflection_log": reflection_log,
                "schema_auto_generated": schema is None,
                "final_schema": schema
            },
            is_valid=is_valid,
            validation_errors=errors,
            extraction_trace=self.extraction_trace
        )

    def _schema_agent(self, content: str, schema: Dict[str, Any]) -> Tuple[List[str], str]:
        """
        OneKE Schema Agent: Preprocesses data and analyzes schema
        - Chunks the document
        - Identifies field types and extraction strategy
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔍 Schema Agent: Analyzing document structure...", flush=True)
        
        # Chunk the content
        chunks = self.chunker.chunk(content)
        
        # Analyze schema and create extraction strategy
        schema_summary = json.dumps(schema, indent=2)
        content_sample = chunks[0][:3000]
        
        prompt = f"""
        You are the Schema Agent in a multi-agent knowledge extraction system.
        
        Task: Analyze the Target Schema and Document Sample to create an extraction strategy.
        
        Output Format:
        1. List each field in the schema with its type and likely location
        2. Identify any transformations needed
        3. Note any challenging fields
        
        Target Schema:
        {schema_summary}
        
        Document Sample:
        {content_sample}
        
        Provide your analysis:
        """
        
        analysis = self.client.generate(
            model=self.schema_agent_model,
            prompt=prompt,
            temperature=0.3,
            num_predict=1024
        )
        
        self.extraction_trace.append({
            "agent": "Schema Agent",
            "action": "analyze_schema",
            "output": analysis
        })
        
        return chunks, analysis

    def _extraction_agent(self, chunk: str, current_data: Dict[str, Any], 
                          schema: Dict[str, Any], schema_analysis: str) -> Tuple[Dict[str, Any], str]:
        """
        OneKE Extraction Agent: Performs extraction with case-based learning
        - Retrieves similar correct cases for few-shot learning
        - Extracts information from current chunk
        - Merges with existing data
        """
        # Retrieve similar correct cases (OneKE case retrieval)
        few_shot_examples = self._retrieve_similar_cases(chunk, is_correct=True, top_k=2)
        
        # Build few-shot context
        few_shot_context = ""
        if few_shot_examples:
            few_shot_context = "\n\nReference Examples (learn from these):\n"
            for i, case in enumerate(few_shot_examples, 1):
                few_shot_context += f"\nExample {i}:\nInput: {case.input_text[:200]}...\n"
                few_shot_context += f"Output: {json.dumps(case.output, indent=2)}\n"
                few_shot_context += f"Reasoning: {case.reasoning}\n"
        
        # Compact current data for context
        current_data_str = json.dumps(current_data, indent=2)
        if len(current_data_str) > 2000:
            current_data_str = current_data_str[:2000] + "... (truncated)"

        prompt = f"""
        You are the Extraction Agent in a multi-agent knowledge extraction system.
        
        Task: Update the "Current Extracted Data" with information from the "New Text Chunk".
        
        Schema Analysis:
        {schema_analysis}
        
        Target Schema:
        {json.dumps(schema, indent=2)}
        {few_shot_context}
        
        Current Extracted Data:
        {current_data_str}
        
        New Text Chunk:
        {chunk}
        
        Guidelines:
        1. MERGE new information into the existing JSON structure
        2. If the chunk contains new fields, add them
        3. If the chunk contains details for existing fields, refine them
        4. Output ONLY the FULL updated JSON object
        
        Updated JSON Output:
        """
        
        response = self.client.generate(
            model=self.extraction_agent_model,
            prompt=prompt,
            temperature=0.1,
            num_predict=4096,
            system="You output only valid JSON."
        )
        
        self.extraction_trace.append({
            "agent": "Extraction Agent",
            "action": "extract_chunk",
            "chunk_preview": chunk[:100],
            "output": response[:500]
        })
        
        cleaned_json = self._clean_json_string(response)
        try:
            new_data = json.loads(cleaned_json)
            return new_data, response
        except json.JSONDecodeError:
            logger.warning("JSON Decode Error in chunk processing. Returning previous state.")
            return current_data, response

    def _reflection_agent(self, extracted_data: Dict[str, Any], schema: Dict[str, Any], 
                          original_text: str) -> Tuple[Dict[str, Any], str]:
        """
        OneKE Reflection Agent: Self-reflects and corrects errors
        - Retrieves similar bad cases to avoid past mistakes
        - Analyzes extraction for potential issues
        - Corrects and refines the output
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Reflection Agent: Reviewing extraction...", flush=True)
        
        # Retrieve similar bad cases to learn from mistakes
        bad_case_warnings = self._retrieve_similar_cases(original_text[:1000], is_correct=False, top_k=2)
        
        warnings_context = ""
        if bad_case_warnings:
            warnings_context = "\n\nPast Mistakes to Avoid:\n"
            for i, case in enumerate(bad_case_warnings, 1):
                warnings_context += f"\nMistake {i}:\n"
                warnings_context += f"Incorrect Output: {json.dumps(case.output, indent=2)}\n"
                warnings_context += f"Issue: {case.reasoning}\n"
        
        prompt = f"""
        You are the Reflection Agent in a multi-agent knowledge extraction system.
        
        Task: Review and correct the extracted data for potential errors.
        
        Target Schema:
        {json.dumps(schema, indent=2)}
        
        Extracted Data:
        {json.dumps(extracted_data, indent=2)}
        {warnings_context}
        
        Instructions:
        1. Check if all schema fields are properly populated
        2. Verify data types match the schema
        3. Look for incomplete or malformed entries
        4. Ensure no hallucinated information
        5. Fix any issues and output the CORRECTED JSON
        
        If extraction is already correct, return it as-is.
        
        Corrected JSON Output:
        """
        
        response = self.client.generate(
            model=self.reflection_agent_model,
            prompt=prompt,
            temperature=0.2,
            num_predict=4096,
            system="You output only valid JSON with corrections applied."
        )
        
        self.extraction_trace.append({
            "agent": "Reflection Agent",
            "action": "reflect_and_correct",
            "output": response[:500]
        })
        
        cleaned_json = self._clean_json_string(response)
        try:
            corrected_data = json.loads(cleaned_json)
            return corrected_data, response
        except json.JSONDecodeError:
            logger.warning("Reflection Agent JSON error. Returning original extraction.")
            return extracted_data, response
    
    def _retrieve_similar_cases(self, query_text: str, is_correct: bool, top_k: int = 2) -> List[CaseExample]:
        """
        OneKE-inspired case retrieval: Find similar past cases for learning
        Uses simple keyword matching (can be enhanced with embeddings)
        """
        case_repository = self.correct_cases if is_correct else self.bad_cases
        
        if not case_repository:
            return []
        
        # Simple keyword-based retrieval (can upgrade to semantic similarity)
        query_words = set(query_text.lower().split()[:50])
        
        scored_cases = []
        for case in case_repository:
            case_words = set(case.input_text.lower().split()[:50])
            overlap = len(query_words & case_words)
            scored_cases.append((overlap, case))
        
        scored_cases.sort(reverse=True, key=lambda x: x[0])
        return [case for _, case in scored_cases[:top_k]]
    
    
    def add_case_to_repository(self, case: CaseExample):
        """Add a case to the appropriate repository for future learning"""
        if case.is_correct:
            self.correct_cases.append(case)
        else:
            self.bad_cases.append(case)
    
    def _clean_json_string(self, json_str: str) -> str:
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start != -1 and end != -1:
            return json_str[start:end+1]
        return json_str

    def _validate_output(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Basic validation - can be enhanced"""
        errors = []
        
        # Check if required schema keys exist
        for key in schema.keys():
            if key not in data:
                errors.append(f"Missing required field: {key}")
        
        is_valid = len(errors) == 0
        return is_valid, errors

