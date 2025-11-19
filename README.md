# Enhanced PDF Processing Pipeline - Modular Architecture

## 🎯 Overview

# Document Layout Analysis & Schema-Based Extraction Pipeline

A comprehensive PDF document processing pipeline that combines advanced layout detection, OCR, and **OneKE-inspired multi-agent schema extraction** with automated schema generation capabilities.

## 🌟 Key Features

### Document Processing
- **Advanced Layout Detection**: DocLayout-YOLO for precise element identification
- **Multi-OCR Engine Support**: EasyOCR, Tesseract, PaddleOCR integration
- **Figure & Table Extraction**: Automated extraction with caption detection
- **Markdown Rendering**: Clean, structured markdown output

### Schema Extraction (OneKE-Inspired)
- **Multi-Agent Architecture**: 
  - **Schema Agent**: Preprocesses and analyzes document structure
  - **Extraction Agent**: Performs knowledge extraction with case-based learning
  - **Reflection Agent**: Self-reflects and corrects errors
- **Automated Schema Generation**: LLM-powered schema inference when no schema provided
- **Case-Based Learning**: Learns from correct/incorrect examples (Case Repository)
- **Iterative Chunk Processing**: Handles large documents efficiently
- **Self-Reflection & Error Correction**: Continuous improvement through feedback

### LLM Integration (Ollama)
- **4-Tier Enhancement System**:
  1. Semantic Embeddings (nomic-embed-text)
  2. Text Cleaning (Mistral)
  3. Region Validation (Mistral)
  4. Post-Processing (Mistral)
- **GPU-Accelerated**: Optimized for local LLM inference
- **Flexible Model Support**: Easy model swapping

## 📋 Architecture

```
OneKE-Inspired Multi-Agent System
├── Schema Agent: Document structure analysis
├── Extraction Agent: Knowledge extraction + case retrieval
└── Reflection Agent: Error detection + self-correction

Pipeline Components
├── Layout Detection: DocLayout-YOLO
├── OCR Processing: EasyOCR/Tesseract/PaddleOCR
├── Content Analysis: Enhanced with Ollama LLMs
└── Output Rendering: Markdown + Structured JSON
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd SECOND_TRY_2_Testing

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Prerequisites

1. **Ollama Server** (for LLM features):
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull mistral
ollama pull nomic-embed-text

# Start Ollama server
ollama serve
```

2. **System Dependencies**:
```bash
# For OpenCV (headless for servers)
pip install opencv-python-headless

# GPU support (optional)
export CUDA_VISIBLE_DEVICES=""  # Force CPU if needed
```

### Basic Usage

#### 1. PDF Processing (Layout + OCR + Markdown)

```bash
python main.py
```

This processes PDFs from the `PDF/` folder and generates:
- Layout analysis with bounding boxes
- OCR-extracted text
- Markdown-formatted output in `Output/extracted_content.md`

#### 2. Schema-Based Extraction

**Option A: With User-Provided Schema**
```bash
# Place your schema in the root directory as schema.json
python schema_extraction/run.py
```

**Option B: Automated Schema Generation**
```bash
# Remove or rename schema.json - system will auto-generate
mv schema.json schema.json.backup
python schema_extraction/run.py
```

Output files:
- `Output/structured_output.json` - Extracted structured data
- `Output/extraction_metadata.json` - Processing metadata

## 📁 Project Structure

```
SECOND_TRY_2_Testing/
├── main.py                      # Main PDF processing pipeline
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── pipeline/                    # Core pipeline components
│   ├── enhanced_pipeline.py    # Main orchestrator
│   ├── layout_detector.py      # DocLayout-YOLO integration
│   ├── ocr_engine.py           # Multi-OCR engine support
│   ├── markdown_renderer.py    # Markdown generation
│   ├── figure_caption_processor.py
│   ├── page_processor.py
│   └── ollama/                 # LLM integration
│       ├── client.py           # Ollama HTTP client
│       ├── embedder.py         # Semantic embeddings (Tier 1)
│       ├── text_cleaner.py     # Text cleaning (Tier 2)
│       ├── region_validator.py # Validation (Tier 3)
│       └── post_processor.py   # Post-processing (Tier 4)
│
├── schema_extraction/          # OneKE-inspired extraction
│   ├── converter.py            # Multi-agent schema converter
│   ├── schema_generator.py    # Automated schema generation
│   ├── run.py                  # Extraction runner
│   └── chunking/
│       └── chunker.py          # Document chunking
│
├── examples/                   # Example files
│   ├── input/
│   │   └── example_schema.json # Sample schema
│   └── output/
│       └── example_structured_output.json
│
├── Output/                     # Generated outputs (gitignored)
├── PDF/                        # Input PDFs (gitignored)
└── .gitignore
```

## 🔧 Configuration

### Pipeline Configuration (`config.py`)

```python
# OCR Engine Selection
OCR_ENGINE = "easyocr"  # Options: easyocr, tesseract, paddleocr

# Layout Detection
LAYOUT_MODEL = "doclayout_yolo"
CONFIDENCE_THRESHOLD = 0.5

# Ollama Integration
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_ENABLED = True
```

### Schema Extraction

Models can be customized in `schema_extraction/converter.py`:

```python
self.schema_agent_model = "mistral"
self.extraction_agent_model = "mistral" 
self.reflection_agent_model = "mistral"
```

## 📊 Example Usage

### Example Schema

See `examples/input/example_schema.json` for a complete technical manual schema with:
- Document metadata
- Hierarchical sections
- Technical specifications
- Tables and figures
- Safety warnings
- Step-by-step procedures

### Example Output

See `examples/output/example_structured_output.json` for:
- Fully populated structured data
- Multi-level nested objects
- Arrays of specifications and procedures
- Extracted table data

## 🧪 Testing

```bash
# Test Ollama integration
python test_ollama_integration.py

# Process a single PDF
python main.py

# Run schema extraction with examples
cp examples/input/example_schema.json ./schema.json
python schema_extraction/run.py
```

## 📚 Implementation Details

### OneKE-Inspired Multi-Agent System

Based on the paper: "OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System" (arXiv:2412.20005v2)

**Key Adaptations:**

1. **Schema Agent**:
   - Analyzes document structure
   - Creates extraction strategy
   - Handles chunking for large documents

2. **Extraction Agent**:
   - Retrieves similar cases from Case Repository
   - Uses few-shot learning with correct examples
   - Iteratively builds structured output

3. **Reflection Agent**:
   - Reviews extraction for errors
   - Learns from past mistakes (bad cases)
   - Applies corrections and refinements

4. **Case Repository**:
   - Stores correct extraction examples
   - Maintains error cases with reflections
   - Enables continuous learning

### Automated Schema Generation

When no schema is provided:
1. Analyzes first chunk to create base schema
2. Iteratively refines with subsequent chunks
3. Identifies new properties and data types
4. Merges findings into comprehensive schema
5. Validates and finalizes structure

## 🎯 Use Cases

- **Technical Documentation**: Extract specifications, procedures, warnings
- **Scientific Papers**: Structure methodology, results, references
- **Financial Reports**: Parse tables, metrics, statements
- **Legal Documents**: Extract clauses, definitions, references
- **Medical Records**: Structure patient data, diagnoses, treatments

## 🔄 Workflow

1. **PDF → Markdown**: Layout detection + OCR → Clean markdown
2. **Markdown → Schema**: (Optional) Auto-generate schema from content
3. **Markdown + Schema → JSON**: Multi-agent extraction → Structured data
4. **Review & Iterate**: Use case repository to improve future extractions

## 🐛 Troubleshooting

### Common Issues

**GPU Errors (CUDA)**:
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

**OpenCV Import Error**:
```bash
# Use headless version
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless
```

**Ollama Connection**:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

**Timeout Issues**:
- Increase timeout in `pipeline/ollama/client.py`
- Reduce chunk size in `schema_extraction/chunking/chunker.py`

## 📝 License

[Specify your license]

## 🙏 Acknowledgments

- **OneKE Paper**: Multi-agent architecture inspiration
- **DocLayout-YOLO**: Layout detection model
- **Ollama**: Local LLM inference
- **EasyOCR/Tesseract/PaddleOCR**: OCR engines

## 📧 Contact

[Your contact information]

## 🗺️ Roadmap

- [ ] Web interface for easier usage
- [ ] Support for more document types (Word, HTML)
- [ ] Enhanced table extraction with structure preservation
- [ ] Multi-language support
- [ ] Batch processing capabilities
- [ ] Schema validation and suggestions
- [ ] Export to multiple formats (CSV, XML, etc.)
 The architecture is designed for maintainability, testability, and future enhancements.

## 📁 Project Structure

```
project_root/
├── pipeline/                          # Core processing modules
│   ├── __init__.py                   # Package initialization
│   ├── layout_detector.py            # YOLO / fallback layout detection
│   ├── ocr_engine.py                 # PaddleOCR integration
│   ├── table_extractor.py            # OCRFlux-style table processing
│   ├── region_merger.py              # IoU-based region merging
│   ├── markdown_renderer.py          # Markdown formatting
│   ├── page_processor.py             # Page processing orchestration
│   └── utils.py                      # Utility functions
├── main.py                           # Main entry point
├── config.py                         # Centralized configuration
├── requirements.txt                  # Dependencies
├── test_modular_pipeline.py          # Module testing
└── README_MODULAR.md                 # This file
```

## 🧩 Module Architecture

### 1. **Layout Detector** (`layout_detector.py`)
- **Purpose**: Advanced document layout detection
- **Features**:
  - DocLayout-YOLO integration (latest model)
  - Fallback computer vision methods
  - Confidence-based filtering
  - Multi-class region detection (Title, Text, Table, Figure, etc.)

### 2. **OCR Engine** (`ocr_engine.py`)
- **Purpose**: Text extraction from images
- **Features**:
  - PaddleOCR integration (PP-OCRv4)
  - Batch text processing
  - Cell-level OCR for tables
  - Confidence scoring

### 3. **Table Extractor** (`table_extractor.py`)
- **Purpose**: OCRFlux-style table structure reconstruction
- **Features**:
  - **Spatial clustering** of text boxes into rows
  - **Column bin alignment** using header detection
  - **Line-anchored extraction** using detected H/V lines
  - **Heuristic fixes** for merged cells and alignment
  - Multiple extraction methods with fallbacks

### 4. **Region Merger** (`region_merger.py`)
- **Purpose**: Intelligent region combination and deduplication
- **Features**:
  - IoU-based overlap detection
  - Semantic region linking
  - Quality filtering
  - Metadata enhancement

### 5. **Markdown Renderer** (`markdown_renderer.py`)
- **Purpose**: Convert processed regions to structured markdown
- **Features**:
  - Reading order detection (multi-column support)
  - Table formatting with validation
  - Heading level detection
  - Figure and formula handling

### 6. **Page Processor** (`page_processor.py`)
- **Purpose**: Orchestrate the complete processing pipeline
- **Features**:
  - Module coordination
  - Debug visualization generation
  - Error handling and recovery
  - Statistics aggregation

## ⚙️ Configuration System

All settings are centralized in `config.py`:

```python
# Model Configuration
MODEL_CONFIG = {
    'confidence_threshold': 0.25,
    'layout_detection_dpi': 300,
    'table_detection_threshold': 0.6,
}

# OCR Configuration  
OCR_CONFIG = {
    'use_angle_cls': True,
    'lang': 'en',
    'confidence_threshold': 0.5,
}

# Table Processing (OCRFlux)
TABLE_CONFIG = {
    'max_text_boxes': 1000,
    'max_rows': 100,
    'y_threshold': 8,  # Row clustering threshold
}
```

## 🚀 Usage

### Basic Usage
```bash
python3 main.py PDF/your_document.pdf
```

### Debug Mode (with visualizations)
```bash
python3 main.py PDF/your_document.pdf --debug
```

### Process Specific Page
```bash
python3 main.py PDF/your_document.pdf --debug-page 2
```

### Custom Output Directory
```bash
python3 main.py PDF/your_document.pdf --output custom_output/
```

## 🔧 OCRFlux Enhancements

The table processing implements OCRFlux-style spatial clustering:

### 1. **Row Clustering**
- Improved Y-coordinate grouping with overlap detection
- Handles misaligned text boxes in the same row

### 2. **Column Alignment** 
- Creates column bins from header row
- Assigns text boxes to closest column center
- Handles variable cell widths

### 3. **Line Anchoring**
- Uses detected horizontal/vertical lines as precise bounda