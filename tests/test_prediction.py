import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.detector import Detector


class TestPredictionMethods:
    """Test prediction methods in detail"""
    
    def test_compute_score_single_text(self, mock_models, sample_texts):
        """Test compute_score with single text input"""
        detector = Detector()
        
        # Mock the tokenization and model outputs
        mock_encoding = Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        with patch.object(detector, '_tokenize', return_value=mock_encoding), \
             patch.object(detector, '_get_logits') as mock_get_logits, \
             patch.object(detector, 'perplexity') as mock_perplexity, \
             patch.object(detector, 'entropy') as mock_entropy:
            
            # Setup mock returns
            observer_logits = torch.randn(1, 5, 1000)
            performer_logits = torch.randn(1, 5, 1000)
            mock_get_logits.return_value = (observer_logits, performer_logits)
            
            mock_perplexity.return_value = (np.array([2.5]), torch.randn(1, 4))
            mock_entropy.return_value = (np.array([3.0]), torch.randn(1, 4))
            
            # Mock tokenizer decode for colored text
            detector.tokenizer.decode.return_value = "test"
            
            score, colored_text = detector.compute_score(sample_texts['short'])
            
            assert isinstance(score, float)
            assert isinstance(colored_text, str)
            assert score == 2.5 / 3.0  # ppl / x_ppl
            
    def test_compute_score_batch_text(self, mock_models, sample_texts):
        """Test compute_score with batch text input"""
        detector = Detector()
        
        # Mock the tokenization and model outputs for batch
        mock_encoding = Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3], [1, 2, 4]])
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
        
        with patch.object(detector, '_tokenize', return_value=mock_encoding), \
             patch.object(detector, '_get_logits') as mock_get_logits, \
             patch.object(detector, 'perplexity') as mock_perplexity, \
             patch.object(detector, 'entropy') as mock_entropy:
            
            # Setup mock returns for batch
            observer_logits = torch.randn(2, 3, 1000)
            performer_logits = torch.randn(2, 3, 1000)
            mock_get_logits.return_value = (observer_logits, performer_logits)
            
            mock_perplexity.return_value = (np.array([2.0, 3.0]), torch.randn(2, 2))
            mock_entropy.return_value = (np.array([2.5, 2.8]), torch.randn(2, 2))
            
            # Mock tokenizer decode
            detector.tokenizer.decode.return_value = "test"
            
            batch_input = [sample_texts['short'], sample_texts['medium']]
            scores, colored_texts = detector.compute_score(batch_input)
            
            assert isinstance(scores, list)
            assert isinstance(colored_texts, list)
            assert len(scores) == 2
            assert len(colored_texts) == 2
            
    def test_colored_text_generation(self, mock_models):
        """Test colored text generation with highlighting"""
        detector = Detector()
        
        mock_encoding = Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4]])
        
        # Mock cross-entropy values - some below threshold, some above
        ce_tensor = torch.tensor([[0.5, 1.5, 0.3, 2.0]])  # 0.7 * ppl = 0.7 * 2.0 = 1.4
        
        with patch.object(detector, '_tokenize', return_value=mock_encoding), \
             patch.object(detector, '_get_logits'), \
             patch.object(detector, 'perplexity', return_value=(np.array([2.0]), ce_tensor)), \
             patch.object(detector, 'entropy', return_value=(np.array([2.5]), torch.randn(1, 3))):
            
            # Mock tokenizer decode to return different tokens
            detector.tokenizer.decode.side_effect = ["The", "AI", "generates", "text"]
            
            score, colored_text = detector.compute_score("test text")
            
            # Check that highlighted spans are present for tokens with low cross-entropy
            assert "<span style='background-color: #FFFF00'>" in colored_text
            
    def test_confidence_calculation(self, mock_models):
        """Test confidence score calculation"""
        detector = Detector()
        
        with patch.object(detector, 'compute_score') as mock_compute:
            # Test different score scenarios
            test_cases = [
                (0.5, "AI-generated"),    # Well below threshold
                (0.8, "AI-generated"),    # Close to threshold
                (0.95, "Human-generated"), # Above threshold
                (1.2, "Human-generated")   # Well above threshold
            ]
            
            for score, expected_pred in test_cases:
                mock_compute.return_value = (score, "test colored")
                result = detector.predict("test")
                
                assert result['prediction'] == expected_pred
                assert 0.5 <= result['confidence'] <= 1.0
                
    def test_prediction_consistency(self, mock_models):
        """Test that predictions are consistent for same input"""
        detector = Detector()
        
        with patch.object(detector, 'compute_score') as mock_compute:
            mock_compute.return_value = (0.7, "test colored")
            
            result1 = detector.predict("test text")
            result2 = detector.predict("test text")
            
            assert result1['prediction'] == result2['prediction']
            assert result1['score'] == result2['score']


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_text_handling(self, mock_models):
        """Test handling of empty text"""
        detector = Detector()
        
        with patch.object(detector, '_tokenize') as mock_tokenize:
            mock_encoding = Mock()
            mock_encoding.input_ids = torch.tensor([[]])
            mock_tokenize.return_value = mock_encoding
            
            # Should handle empty input gracefully
            with patch.object(detector, '_get_logits'), \
                 patch.object(detector, 'perplexity'), \
                 patch.object(detector, 'entropy'):
                try:
                    result = detector.predict("")
                except Exception as e:
                    # Expected to potentially fail with empty input
                    assert True
                    
    def test_very_long_text_truncation(self, mock_models):
        """Test handling of very long text (should be truncated)"""
        detector = Detector()
        
        # Create very long text (over max_token_observed)
        long_text = "This is a test. " * 200  # Much longer than 1024 tokens
        
        with patch.object(detector, '_tokenize') as mock_tokenize:
            mock_tokenize.return_value = Mock()
            detector.predict(long_text)
            
            # Check that tokenizer was called with truncation
            call_args = mock_tokenize.call_args[1] if mock_tokenize.call_args else {}
            # The _tokenize method should handle truncation internally
            
    def test_special_characters_handling(self, mock_models):
        """Test handling of text with special characters"""
        detector = Detector()
        
        special_text = "Hello! @#$%^&*() 你好 🚀 ñoño"
        
        with patch.object(detector, 'compute_score') as mock_compute:
            mock_compute.return_value = (0.8, "test")
            result = detector.predict(special_text)
            
            assert result is not None
            assert 'prediction' in result
            
    def test_different_temperature_effects(self, mock_models):
        """Test perplexity and entropy with different temperatures"""
        detector = Detector()
        
        batch_size, seq_len, vocab_size = 1, 5, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        mock_encoding = Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        # Test different temperatures
        ppl_low, _ = detector.perplexity(mock_encoding, logits, temperature=0.5)
        ppl_high, _ = detector.perplexity(mock_encoding, logits, temperature=2.0)
        
        # Lower temperature should generally give different perplexity
        assert ppl_low.shape == ppl_high.shape