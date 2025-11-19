# Examples

This folder contains example inputs and outputs for the document processing pipeline.

## 📁 Structure

```
examples/
├── input/
│   ├── example_schema.json              # Sample JSON schema for extraction
│   └── example_extracted_markdown.md    # Sample markdown from PDF processing
└── output/
    └── example_structured_output.json   # Sample structured data extraction result
```

## 🔄 Workflow

### 1. PDF Processing → Markdown

**Input**: PDF document (place in `../PDF/`)
**Process**: Run `python main.py`
**Output**: `example_extracted_markdown.md` (similar format)

The pipeline:
- Detects document layout (headers, paragraphs, tables, figures)
- Applies OCR to extract text
- Renders clean markdown with preserved structure

### 2. Markdown + Schema → Structured Data

**Input**: 
- `example_extracted_markdown.md` (from step 1)
- `example_schema.json` (your data schema)

**Process**: Run `python schema_extraction/run.py`

**Output**: `example_structured_output.json`

The extraction system:
- Analyzes document structure (Schema Agent)
- Extracts information matching schema (Extraction Agent)
- Reviews and corrects errors (Reflection Agent)

### 3. Auto-Generate Schema (Optional)

If you don't have a schema, the system can generate one:

**Input**: `example_extracted_markdown.md`
**Process**: Run `python schema_extraction/run.py` (without schema.json)
**Output**: Auto-generated schema + structured data

## 📋 Example Schema

The `example_schema.json` demonstrates a technical manual schema with:

- **document_metadata**: Title, version, date, document type
- **sections**: Hierarchical document structure with subsections
- **specifications**: Technical parameters with values and units
- **tables**: Structured table data with headers and rows
- **figures**: Figure references with captions
- **warnings_and_cautions**: Safety information
- **procedures**: Step-by-step instructions

## 📊 Example Output

The `example_structured_output.json` shows:

- Fully populated fields matching the schema
- Nested objects (sections with subsections)
- Arrays of data (specifications, tables, procedures)
- Preserved table structures
- Extracted procedural steps

## 🚀 Try It Yourself

### Use Example Files

```bash
# Copy example schema to root
cp examples/input/example_schema.json ./schema.json

# Copy example markdown (or use your own from Output/)
cp examples/input/example_extracted_markdown.md ./Output/extracted_content.md

# Run extraction
python schema_extraction/run.py

# View results
cat Output/structured_output.json
```

### Process Your Own PDF

```bash
# 1. Place your PDF in PDF/ folder
cp /path/to/your/document.pdf PDF/

# 2. Extract to markdown
python main.py

# 3. (Optional) Create your own schema or let system auto-generate
# If creating manually, use example_schema.json as template

# 4. Run extraction
python schema_extraction/run.py

# 5. View results
cat Output/structured_output.json
```

## 🎯 Schema Design Tips

1. **Start Simple**: Begin with basic fields, expand as needed
2. **Use Descriptive Names**: `operating_temperature` not `temp`
3. **Include Examples**: Helps LLM understand expected format
4. **Nest Related Data**: Group related fields in objects
5. **Mark Required Fields**: Specify which fields are mandatory
6. **Add Descriptions**: Clear descriptions improve extraction accuracy

## 📝 Customization

### Modify Example Schema

Edit `example_schema.json` to match your document structure:

```json
{
  "properties": {
    "your_custom_field": {
      "type": "string",
      "description": "Description of what this field contains",
      "examples": ["example1", "example2"]
    }
  }
}
```

### Create New Schema from Scratch

Use JSON Schema draft-07 format:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Your Schema Title",
  "type": "object",
  "properties": {
    // Your fields here
  },
  "required": ["field1", "field2"]
}
```

## 🔍 Expected Results

With the example files, you should get:

- **Extraction Time**: 30-60 seconds (depends on document size)
- **Accuracy**: 90%+ for well-structured documents
- **Fields Populated**: All fields with available data from markdown
- **Missing Data**: Empty strings or arrays for unavailable data

## 💡 Tips for Best Results

1. **Clear Markdown Structure**: Better markdown = better extraction
2. **Appropriate Schema**: Schema should match document content
3. **Chunk Size**: Adjust in `schema_extraction/chunking/chunker.py` for very large docs
4. **Model Selection**: Try different models (mistral, llama3, etc.) for your use case
5. **Iterative Refinement**: Use case repository to improve over time

## 🐛 Troubleshooting

**Empty Output**:
- Check markdown file exists and has content
- Verify schema matches document structure
- Review extraction logs for errors

**Incomplete Data**:
- Markdown may be missing information
- Schema fields may be too specific
- Try auto-generation to see what system detects

**Slow Processing**:
- Large documents need more time
- Reduce chunk size for faster processing
- Use CPU-only mode if GPU issues

## 📚 Further Reading

- [Main README](../README.md) - Full documentation
- [OneKE Paper](https://arxiv.org/abs/2412.20005) - Multi-agent inspiration
- [JSON Schema](https://json-schema.org/) - Schema specification
