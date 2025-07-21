# AI Text Detector

A Python package for detecting AI-generated text using dual language model analysis with perplexity and cross-entropy metrics.

## Features

- Detects AI-generated text using perplexity and cross-entropy analysis
- Supports multiple detection modes (accuracy vs low false positive rate)
- Highlights suspicious words in AI-generated content
- Command-line interface and interactive mode for multiple predictions

## Installation

### Prerequisites

Install core dependencies with specific versions to avoid conflicts:

```bash
pip install torch==2.0.1 transformers==4.49.0 bitsandbytes==0.41.3
```


```bash
git clone https://github.com/wangjing0/ai_detector.git
cd ai_detector
```

```bash
# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

> **Note**: This package requires specific PyTorch and Transformers versions due to dependency compatibility requirements.

## Quick Start

### Command Line Interface

```bash
# Analyze a single text
ai-detector "Your text here"

# Interactive mode for multiple texts
ai-detector --interactive

# Display word-level highlights for suspicious content
ai-detector --display-highlights "Text to analyze"
```

### Python API

```python
from src.detector import Detector

# Initialize detector with preferred models and metrics mode
detector = Detector(observer_name_or_path="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
              performer_name_or_path="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
              mode='accuracy')

# Analyze text
result = detector.predict("Your text here")

# Display results
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Development

### Setup Development Environment

```bash
# Install with development dependencies
poetry install --with dev
```

## License

MIT License