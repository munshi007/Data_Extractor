# Ollama Integration for PDF Extraction Pipeline

A 4-tier LLM-powered enhancement system for the PDF document extraction pipeline.

## Architecture Overview

```
Tier 1: Semantic Embeddings (nomic-embed-text)
         ↓
         Semantic Text Grouper
         Better paragraph grouping, reduces duplication
         
Tier 2: Text Cleaning (mistral)
         ↓
         Markdown Renderer
         Cleans OCR errors, normalizes text
         
Tier 3: Region Validation (deepsee-r1)
         ↓
         Region Processor
         Validates classifications, fixes errors
         
Tier 4: Post-Processing (mistral)
         ↓
         Enhanced Pipeline
         Final markdown quality improvements
```

## Directory Structure

```
pipeline/ollama/
├── __init__.py              # Module exports
├── config.py                # Configuration settings
├── client.py                # Base Ollama client
├── embedder.py              # Tier 1: Semantic embeddings
├── text_cleaner.py          # Tier 2: Text cleaning
├── region_validator.py      # Tier 3: Region validation
└── post_processor.py        # Tier 4: Post-processing
```

## Installation & Setup

### 1. Start Ollama Server

```bash
OLLAMA_MODELS=/local/home/munshi/TESTING/.ollama/models \
/local/home/munshi/TESTING/.local/bin/ollama serve
```

### 2. Run Pipeline

```bash
cd /local/home/munshi/SECOND_TRY_2_Testing
python3 main.py --debug
```

## Features

### Tier 1: Semantic Embeddings
**Model:** `nomic-embed-text`  
**Replaces:** sentence-transformers  
**Benefits:**
- GPU-accelerated embedding generation
- Embedding caching for performance
- Better semantic similarity computations
- Faster paragraph grouping

**Code Location:** `pipeline/semantic_grouper.py`

```python
# Automatic initialization if Ollama is available
grouper = SemanticTextGrouper(use_ollama=True)
```

### Tier 2: Text Cleaning
**Model:** `mistral`  
**Purpose:** Clean extracted PDF text  
**Features:**
- Regex-based fast cleaning
- LLM-based intelligent cleaning for complex text
- Specific cleaners for headings, paragraphs, tables
- Common OCR error fixes

**Code Location:** `pipeline/markdown_renderer.py`

```python
# Automatic initialization in MarkdownRenderer
renderer = MarkdownRenderer(use_ollama_cleaning=True)
```

### Tier 3: Region Validation
**Model:** `deepsee-r1`  
**Purpose:** Validate region classifications  
**Features:**
- Reasoning-based classification verification
- Quick heuristic checks for captions
- Batch validation support
- Confidence scores

**Code Location:** `pipeline/region_processor.py`

```python
# Automatic initialization in RegionProcessor
processor = RegionProcessor(use_ollama_validation=True)
```

### Tier 4: Post-Processing
**Model:** `mistral`  
**Purpose:** Improve final markdown quality  
**Features:**
- Markdown formatting fixes
- Structure validation
- Table of contents generation
- LLM-based quality improvements

**Code Location:** `pipeline/enhanced_pipeline.py`

```python
# Automatic initialization in EnhancedPipeline
post_processor = OllamaPostProcessor()
```

## Configuration

Edit `pipeline/ollama/config.py` to customize:

```python
# Enable/disable specific tiers
USE_OLLAMA_EMBEDDINGS = True
AGGRESSIVE_TEXT_CLEANING = False
VALIDATE_ALL_REGIONS = True
POST_PROCESS_MARKDOWN = True

# Performance tuning
EMBEDDING_CACHE_SIZE = 1000
LLM_TIMEOUT = 300
LLM_MAX_TOKENS = 512
```

## Performance Impact

### Before Ollama Integration
- Duplicate text blocks at output start
- OCR artifacts in text
- Occasional classification errors
- Paragraph grouping suboptimal

### After Ollama Integration
- ✓ Cleaner output with less duplication
- ✓ Professional markdown formatting
- ✓ Better region classification
- ✓ Improved semantic grouping
- ✓ GPU acceleration via embeddings

## Troubleshooting

### "ollama server not responding"
```bash
# Make sure Ollama is running
ps aux | grep ollama

# Start it:
OLLAMA_MODELS=/local/home/munshi/TESTING/.ollama/models \
/local/home/munshi/TESTING/.local/bin/ollama serve
```

### Model not found
```bash
# List available models
OLLAMA_MODELS=/local/home/munshi/TESTING/.ollama/models \
/local/home/munshi/TESTING/.local/bin/ollama list
```

### Slow performance
- Disable aggressive text cleaning: `AGGRESSIVE_TEXT_CLEANING = False`
- Reduce embedding cache size if RAM is limited
- Use smaller models (gemma3:4b instead of gpt-oss:20b)

## Logging

Enable detailed logging to see each tier in action:

```bash
export LOGLEVEL=DEBUG
python3 main.py --debug
```

Watch for these log messages:
- `"Tier 1: Semantic embeddings initialized"` → semantic_grouper.py
- `"Tier 2: Text cleaner initialized"` → markdown_renderer.py
- `"Tier 3: Region validator initialized"` → region_processor.py
- `"Tier 4: Post-processor initialized"` → enhanced_pipeline.py

## Models Used

| Tier | Model | Size | Purpose | GPU Memory |
|------|-------|------|---------|------------|
| 1 | nomic-embed-text | ~274MB | Embeddings | ~300MB |
| 2 | mistral | ~4GB | Text cleaning | ~4.5GB |
| 3 | deepsee-r1 | ~8GB | Reasoning | ~8.5GB |
| 4 | mistral | ~4GB | Post-processing | ~4.5GB |

**Total GPU Memory Required:** ~17GB (uses model sharing/unloading)

## API Reference

### OllamaClient
```python
from pipeline.ollama import OllamaClient

client = OllamaClient()
response = client.generate("mistral", "What is 2+2?")
embedding = client.embed("nomic-embed-text", "Some text")
```

### OllamaEmbedder
```python
from pipeline.ollama import OllamaEmbedder

embedder = OllamaEmbedder()
emb1 = embedder.embed("text1")
similarity = embedder.similarity("text1", "text2")
```

### OllamaTextCleaner
```python
from pipeline.ollama import OllamaTextCleaner

cleaner = OllamaTextCleaner()
cleaned = cleaner.clean_text("messy text")
heading = cleaner.clean_heading("heading text")
```

### OllamaRegionValidator
```python
from pipeline.ollama import OllamaRegionValidator

validator = OllamaRegionValidator()
result = validator.validate_region(region_dict)
corrected_regions = validator.validate_regions_batch(regions_list)
```

### OllamaPostProcessor
```python
from pipeline.ollama import OllamaPostProcessor

post_processor = OllamaPostProcessor()
improved_md = post_processor.improve_markdown(markdown_text)
validation = post_processor.validate_markdown_structure(markdown_text)
```

## Next Steps

1. **Run the pipeline with Ollama integration:**
   ```bash
   OLLAMA_MODELS=/local/home/munshi/TESTING/.ollama/models \
   /local/home/munshi/TESTING/.local/bin/ollama serve &
   
   cd /local/home/munshi/SECOND_TRY_2_Testing
   python3 main.py
   ```

2. **Compare output with original pipeline:**
   - Check `Output/extracted_content.md` for cleaner text
   - Look at JSON logs for validation results

3. **Fine-tune based on results:**
   - Disable tiers that don't improve output
   - Adjust timeouts for your hardware
   - Use different models if needed

## License & Credits

Ollama Integration uses:
- [Ollama](https://ollama.ai) - Local LLM inference
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text) - Embeddings
- [Mistral](https://mistral.ai) - Text processing
- [DeepSeek-R1](https://github.com/deepseek-ai) - Reasoning
