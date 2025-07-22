import pytest
import torch
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def mock_models():
    """Mock the heavy model loading for faster tests"""
    with patch('ai_detector.detector.AutoModelForCausalLM') as mock_model_class, \
         patch('ai_detector.detector.AutoTokenizer') as mock_tokenizer_class:
        
        mock_observer = Mock()
        mock_performer = Mock()
        mock_tokenizer = Mock()
        
        mock_observer.device = "cpu"
        mock_performer.device = "cpu"
        mock_observer.eval.return_value = None
        mock_performer.eval.return_value = None
        
        mock_tokenizer.vocab = {"test": 1, "token": 2}
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 0
        
        mock_model_class.from_pretrained.side_effect = [mock_observer, mock_performer]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        yield {
            'observer': mock_observer,
            'performer': mock_performer, 
            'tokenizer': mock_tokenizer
        }

@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return {
        'short': "This is a test.",
        'medium': "This is a longer test text that should be sufficient for testing purposes.",
        'ai_like': "As an AI language model, I can assist you with various tasks and provide information on a wide range of topics.",
        'human_like': "I went to the store yesterday and bought some groceries for dinner tonight."
    }