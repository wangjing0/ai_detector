# AI Text Detector

A Python package for detecting AI-generated text using dual language model analysis.

## Features

- Detects AI-generated text using perplexity and cross-entropy analysis
- Supports multiple detection modes (accuracy vs low false positive rate)
- Highlights suspicious words in AI-generated content
- Command-line interface and Python API
- Interactive mode for multiple predictions

## Installation

### Using Poetry

```bash
poetry install
```

### Using pip

```bash
pip install -e .
```

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