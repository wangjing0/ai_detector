# AI Text Detector

Zero-shot detecting AI-generated content with appropriate baselines.

## Features

- Detects AI-generated text using perplexity and cross-entropy analysis
- Supports multiple detection modes (accuracy vs low false-positive-rate)
- Highlights suspicious words in AI-generated content
- Command-line interface and interactive mode for multiple predictions

## Installation

### Prerequisites

Install core dependencies with specific versions to avoid conflicts:

```bash
git clone https://github.com/wangjing0/ai_detector.git
cd ai_detector
pip install -e .
```

or use poetry:
```bash
poetry lock
poetry install
```

> **Note**: This package requires specific PyTorch and Transformers versions due to dependency compatibility requirements.

## Quick Start

### Command Line Interface

```bash
# Analyze a single text
ai-detector "Your text here"

# Interactive mode for multiple texts
ai-detector --interactive --display-highlights

# Display word-level highlights for suspicious content
ai-detector --display-highlights "Text to analyze"

# List available model configurations
ai-detector --list-configs

# Get model configuration recommendation
ai-detector --recommend-config --vram-gb 16 --speed-priority

# Use a predefined model configuration
ai-detector --model-config llama_3_1_8b
```

### Development

```bash 
python -m ai_detector.detector_app --interactive --display-highlights
```

or

```python
from src.detector import Detector

# Initialize detector with preferred models and metrics mode
detector = Detector(observer_name_or_path="google/gemma-2b",
              performer_name_or_path="google/gemma-2b-it",
              mode='accuracy')

# Analyze text
result = detector.predict("Your text here")

# Display results
print(f"Prediction: {result['prediction']}")
print(f"Score: {result['score'] :.3f}")
print(f"Confidence: {result['confidence']:.3f}")
```

## License

MIT License (c) 2025 Jing Wang