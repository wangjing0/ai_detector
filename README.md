# AI Text Detector

A Python package for detecting AI-generated text using dual language model analysis.

## Features

- Detects AI-generated text using perplexity and cross-entropy analysis
- Supports multiple detection modes (accuracy vs low false positive rate)
- Highlights suspicious words in AI-generated content
- Command-line interface and Python API
- Interactive mode for multiple predictions

## Installation

### Using Poetry (Fixed)

Poetry installation now works with proper dependency resolution:

```bash
# First, restore to working versions if needed
pip install torch==2.0.1 transformers==4.49.0 bitsandbytes==0.41.3

# Install using pip with pyproject.toml  
pip install -e .
```

### Using pip (Original)

```bash
pip install -r requirements.txt
```

**Note**: There are complex dependency conflicts between newer versions of torch/transformers. The package is configured to work with the current environment versions.

## Quick Start

### Command Line

```bash
# Single prediction
ai-detector "Your text here"

# Interactive mode
ai-detector --interactive

# With highlighting
ai-detector --display-highlights "Suspicious text"
```

### Python API

```python
from src.detector import Detector

detector = Detector(mode='accuracy')
result = detector.predict("Your text here")

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Configuration

The detector uses two language models:
- **Observer Model**: Analyzes text patterns
- **Performer Model**: Generates reference probabilities

Default models:
- Observer: `unsloth/Meta-Llama-3.1-8B-bnb-4bit`
- Performer: `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`

## Development

```bash
# Install with dev dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black src/

# Type checking
poetry run mypy src/
```

## License

MIT License