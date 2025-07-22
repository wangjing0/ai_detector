import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from ai_detector.detector import Detector, ACCURACY_THRESHOLD, FPR_THRESHOLD


class TestDetectorInit:
    """Test Detector class initialization"""
    
    def test_init_with_default_params(self, mock_models):
        """Test initialization with default parameters"""
        detector = Detector()
        
        assert detector.threshold == ACCURACY_THRESHOLD
        assert detector.max_token_observed == 1024
        assert detector.tokenizer.pad_token == "</s>"
        
    def test_init_with_custom_params(self, mock_models):
        """Test initialization with custom parameters"""
        observer_model = "test/observer"
        performer_model = "test/performer"
        max_tokens = 512
        mode = "low-fpr"
        
        detector = Detector(
            observer_name_or_path=observer_model,
            performer_name_or_path=performer_model,
            max_token_observed=max_tokens,
            mode=mode
        )
        
        assert detector.threshold == FPR_THRESHOLD
        assert detector.max_token_observed == max_tokens
        
    def test_threshold_mode_accuracy(self, mock_models):
        """Test threshold mode setting to accuracy"""
        detector = Detector(mode="accuracy")
        assert detector.threshold == ACCURACY_THRESHOLD
        
    def test_threshold_mode_low_fpr(self, mock_models):
        """Test threshold mode setting to low-fpr"""
        detector = Detector(mode="low-fpr")
        assert detector.threshold == FPR_THRESHOLD
        
    def test_invalid_threshold_mode(self, mock_models):
        """Test invalid threshold mode raises error"""
        with pytest.raises(ValueError, match="Invalid mode"):
            Detector(mode="invalid_mode")
            
    def test_tokenizer_consistency_check_pass(self, mock_models):
        """Test tokenizer consistency check passes with identical tokenizers"""
        # This should not raise an error with mocked identical tokenizers
        detector = Detector()
        
    def test_tokenizer_consistency_check_fail(self, mock_models):
        """Test tokenizer consistency check fails with different tokenizers"""
        with patch('ai_detector.detector.AutoTokenizer') as mock_tokenizer_class:
            # Create different tokenizers
            mock_tokenizer1 = Mock()
            mock_tokenizer2 = Mock()
            mock_tokenizer1.vocab = {"test": 1}
            mock_tokenizer2.vocab = {"different": 2}
            
            mock_tokenizer_class.from_pretrained.side_effect = [
                mock_tokenizer1, mock_tokenizer2, mock_tokenizer1
            ]
            
            with pytest.raises(ValueError, match="Tokenizers are not identical"):
                Detector()


class TestDetectorTokenization:
    """Test Detector tokenization methods"""
    
    def test_tokenize_single_text(self, mock_models):
        """Test tokenizing single text"""
        detector = Detector()
        
        # Mock tokenizer return
        mock_encoding = Mock()
        mock_encoding.to.return_value = mock_encoding
        detector.tokenizer.return_value = mock_encoding
        
        result = detector._tokenize(["test text"])
        
        detector.tokenizer.assert_called_once()
        mock_encoding.to.assert_called_once_with(detector.DEVICE_1)
        
    def test_tokenize_batch_text(self, mock_models):
        """Test tokenizing batch of texts"""
        detector = Detector()
        
        mock_encoding = Mock()
        mock_encoding.to.return_value = mock_encoding
        detector.tokenizer.return_value = mock_encoding
        
        batch = ["text1", "text2"]
        result = detector._tokenize(batch)
        
        detector.tokenizer.assert_called_once()
        call_args = detector.tokenizer.call_args
        assert call_args[0][0] == batch
        assert call_args[1]['padding'] == 'longest'


class TestDetectorInference:
    """Test Detector inference methods"""
    
    def test_get_logits(self, mock_models):
        """Test getting logits from both models"""
        detector = Detector()
        
        # Mock model outputs
        mock_observer_output = Mock()
        mock_performer_output = Mock()
        mock_observer_logits = torch.randn(1, 10, 1000)
        mock_performer_logits = torch.randn(1, 10, 1000)
        
        mock_observer_output.logits = mock_observer_logits
        mock_performer_output.logits = mock_performer_logits
        
        detector.observer_model.return_value = mock_observer_output
        detector.performer_model.return_value = mock_performer_output
        
        # Mock encoding
        mock_encoding = Mock()
        mock_encoding.to.return_value = mock_encoding
        
        observer_logits, performer_logits = detector._get_logits(mock_encoding)
        
        assert torch.equal(observer_logits, mock_observer_logits)
        assert torch.equal(performer_logits, mock_performer_logits)
        
    def test_perplexity_computation(self, mock_models):
        """Test perplexity computation"""
        detector = Detector()
        
        # Create mock data
        batch_size, seq_len, vocab_size = 1, 5, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        mock_encoding = Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        ppl, ce = detector.perplexity(mock_encoding, logits)
        
        assert isinstance(ppl, np.ndarray)
        assert ppl.shape == (batch_size,)
        assert isinstance(ce, torch.Tensor)
        
    def test_entropy_computation(self, mock_models):
        """Test cross-entropy computation"""
        detector = Detector()
        
        batch_size, seq_len, vocab_size = 1, 5, 1000
        p_logits = torch.randn(batch_size, seq_len, vocab_size)
        q_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        mock_encoding = Mock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        agg_ce, ce = detector.entropy(p_logits, q_logits, mock_encoding, pad_token_id=0)
        
        assert isinstance(agg_ce, np.ndarray)
        assert agg_ce.shape == (batch_size,)
        assert isinstance(ce, torch.Tensor)


class TestDetectorPrediction:
    """Test Detector prediction methods"""
    
    @patch('ai_detector.detector.Detector.compute_score')
    def test_predict_single_text(self, mock_compute_score, mock_models):
        """Test prediction for single text"""
        detector = Detector()
        
        # Mock compute_score return
        test_score = 0.5
        test_colored = "test colored text"
        mock_compute_score.return_value = (test_score, test_colored)
        
        result = detector.predict("test text")
        
        assert result['prediction'] == 'AI-generated'  # 0.5 < ACCURACY_THRESHOLD
        assert result['score'] == test_score
        assert result['colored_text'] == test_colored
        assert result['text'] == "test text"
        assert isinstance(result['confidence'], np.floating)
        
    @patch('ai_detector.detector.Detector.compute_score') 
    def test_predict_batch_text(self, mock_compute_score, mock_models):
        """Test prediction for batch of texts"""
        detector = Detector()
        
        # Mock compute_score return for batch
        test_scores = [0.5, 0.95]  # One AI, one human
        test_colored = ["colored1", "colored2"]
        mock_compute_score.return_value = (test_scores, test_colored)
        
        batch_text = ["text1", "text2"]
        result = detector.predict(batch_text)
        
        assert result['prediction'] == ['AI-generated', 'Human-generated']
        assert result['score'] == test_scores
        assert result['colored_text'] == test_colored
        assert result['text'] == batch_text
        
    def test_predict_human_threshold(self, mock_models):
        """Test prediction returns human for high scores"""
        detector = Detector()
        
        with patch.object(detector, 'compute_score') as mock_compute:
            mock_compute.return_value = (0.95, "test")  # Above threshold
            result = detector.predict("test")
            assert result['prediction'] == 'Human-generated'
            
    def test_predict_ai_threshold(self, mock_models):
        """Test prediction returns AI for low scores"""
        detector = Detector()
        
        with patch.object(detector, 'compute_score') as mock_compute:
            mock_compute.return_value = (0.5, "test")  # Below threshold
            result = detector.predict("test")
            assert result['prediction'] == 'AI-generated'