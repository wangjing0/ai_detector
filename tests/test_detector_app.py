import pytest
import sys
from io import StringIO
from unittest.mock import Mock, patch, MagicMock
import subprocess
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_detector.detector_app import get_detector, predict_text, main


class TestDetectorApp:
    """Test detector_app CLI functionality"""
    
    def test_get_detector_singleton(self):
        """Test that get_detector implements singleton pattern"""
        with patch('ai_detector.detector_app.Detector') as mock_detector_class:
            mock_instance = Mock()
            mock_detector_class.return_value = mock_instance
            
            # First call should create instance
            detector1 = get_detector()
            assert detector1 == mock_instance
            mock_detector_class.assert_called_once()
            
            # Second call should return same instance
            detector2 = get_detector()
            assert detector2 == mock_instance
            assert detector1 is detector2
            # Detector class should still only be called once
            mock_detector_class.assert_called_once()
            
    def test_get_detector_with_custom_params(self):
        """Test get_detector with custom parameters"""
        with patch('ai_detector.detector_app.Detector') as mock_detector_class:
            mock_instance = Mock()
            mock_detector_class.return_value = mock_instance
            
            observer_model = "custom/observer"
            performer_model = "custom/performer"
            mode = "low-fpr"
            
            detector = get_detector(observer_model, performer_model, mode)
            
            mock_detector_class.assert_called_once_with(
                observer_name_or_path=observer_model,
                performer_name_or_path=performer_model,
                mode=mode
            )
            
    def test_predict_text_function(self):
        """Test predict_text wrapper function"""
        with patch('ai_detector.detector_app.get_detector') as mock_get_detector:
            mock_detector = Mock()
            mock_result = {
                'prediction': 'AI-generated',
                'score': 0.7,
                'confidence': 0.8,
                'colored_text': 'test text',
                'text': 'test input'
            }
            mock_detector.predict.return_value = mock_result
            mock_get_detector.return_value = mock_detector
            
            result = predict_text("test input", display_highlights=True)
            
            assert result == mock_result
            mock_detector.predict.assert_called_once_with("test input", display_text=True)


class TestCLIArguments:
    """Test CLI argument parsing and handling"""
    
    @patch('ai_detector.detector_app.get_detector')
    def test_main_with_text_argument(self, mock_get_detector):
        """Test main function with text argument"""
        mock_detector = Mock()
        mock_result = {
            'prediction': 'Human-generated',
            'score': 0.95,
            'confidence': 0.9,
            'colored_text': 'test text',
            'text': 'test input'
        }
        mock_detector.predict.return_value = mock_result
        mock_get_detector.return_value = mock_detector
        
        test_args = ['detector_app.py', 'This is test text']
        
        with patch('sys.argv', test_args), \
             patch('builtins.print') as mock_print:
            main()
            
            # Check that detector was called with the text
            mock_detector.predict.assert_called_once_with('This is test text', display_text=False)
            
            # Check print calls for output
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any('Human-generated' in call for call in print_calls)
            assert any('0.9500' in call for call in print_calls)
            
    @patch('ai_detector.detector_app.get_detector')
    def test_main_with_stdin_input(self, mock_get_detector):
        """Test main function with stdin input"""
        mock_detector = Mock()
        mock_result = {
            'prediction': 'AI-generated', 
            'score': 0.6,
            'confidence': 0.7,
            'colored_text': 'input from stdin',
            'text': 'input from stdin'
        }
        mock_detector.predict.return_value = mock_result
        mock_get_detector.return_value = mock_detector
        
        test_args = ['detector_app.py']
        mock_stdin = StringIO('input from stdin')
        
        with patch('sys.argv', test_args), \
             patch('sys.stdin', mock_stdin), \
             patch('sys.stdin.isatty', return_value=False), \
             patch('builtins.print') as mock_print:
            main()
            
            mock_detector.predict.assert_called_once_with('input from stdin', display_text=False)
            
    @patch('ai_detector.detector_app.get_detector')
    def test_main_with_mode_arguments(self, mock_get_detector):
        """Test main function with different mode arguments"""
        mock_detector = Mock()
        mock_result = {'prediction': 'AI-generated', 'score': 0.8, 'confidence': 0.85, 'colored_text': '', 'text': ''}
        mock_detector.predict.return_value = mock_result
        mock_get_detector.return_value = mock_detector
        
        test_args = ['detector_app.py', '--mode', 'low-fpr', 'test text']
        
        with patch('sys.argv', test_args), \
             patch('builtins.print'):
            main()
            
            # Check that get_detector was called with low-fpr mode
            mock_get_detector.assert_called_with(
                'unsloth/Meta-Llama-3.1-8B-bnb-4bit',
                'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit', 
                'low-fpr'
            )
            
    @patch('ai_detector.detector_app.get_detector')
    def test_main_with_custom_models(self, mock_get_detector):
        """Test main function with custom model arguments"""
        mock_detector = Mock()
        mock_result = {'prediction': 'Human-generated', 'score': 0.9, 'confidence': 0.95, 'colored_text': '', 'text': ''}
        mock_detector.predict.return_value = mock_result
        mock_get_detector.return_value = mock_detector
        
        test_args = [
            'detector_app.py',
            '--observer-model', 'custom/observer',
            '--performer-model', 'custom/performer',
            'test text'
        ]
        
        with patch('sys.argv', test_args), \
             patch('builtins.print'):
            main()
            
            mock_get_detector.assert_called_with(
                'custom/observer',
                'custom/performer',
                'accuracy'
            )
            
    @patch('ai_detector.detector_app.get_detector')
    def test_main_display_highlights(self, mock_get_detector):
        """Test main function with display highlights option"""
        mock_detector = Mock()
        mock_result = {
            'prediction': 'AI-generated',
            'score': 0.5,
            'confidence': 0.8,
            'colored_text': '<span>highlighted text</span>',
            'text': 'test text'
        }
        mock_detector.predict.return_value = mock_result
        mock_get_detector.return_value = mock_detector
        
        test_args = ['detector_app.py', '--display-highlights', 'test text']
        
        with patch('sys.argv', test_args), \
             patch('builtins.print') as mock_print:
            main()
            
            mock_detector.predict.assert_called_once_with('test text', display_text=True)
            
            # Check that highlights message is shown for AI-generated text
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any('Highlighted suspicious words' in call for call in print_calls)


class TestInteractiveMode:
    """Test interactive mode functionality"""
    
    @patch('ai_detector.detector_app.get_detector')
    def test_interactive_mode_basic(self, mock_get_detector):
        """Test basic interactive mode functionality"""
        mock_detector = Mock()
        mock_result = {
            'prediction': 'AI-generated',
            'score': 0.7,
            'confidence': 0.8,
            'colored_text': 'test',
            'text': 'test'
        }
        mock_detector.predict.return_value = mock_result
        mock_get_detector.return_value = mock_detector
        
        test_args = ['detector_app.py', '--interactive']
        
        # Simulate user input: one test input then quit
        mock_input_sequence = ['test input', 'quit']
        
        with patch('sys.argv', test_args), \
             patch('builtins.input', side_effect=mock_input_sequence), \
             patch('builtins.print') as mock_print:
            main()
            
            # Check that detector was called with user input
            mock_detector.predict.assert_called_with('test input', display_text=False)
            
            # Check that results were printed
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any('AI-generated' in call for call in print_calls)
            assert any('Goodbye!' in call for call in print_calls)
            
    @patch('ai_detector.detector_app.get_detector')
    def test_interactive_mode_empty_input(self, mock_get_detector):
        """Test interactive mode handles empty input"""
        mock_detector = Mock()
        mock_get_detector.return_value = mock_detector
        
        test_args = ['detector_app.py', '--interactive']
        
        # Simulate empty input then quit
        mock_input_sequence = ['', 'quit']
        
        with patch('sys.argv', test_args), \
             patch('builtins.input', side_effect=mock_input_sequence), \
             patch('builtins.print'):
            main()
            
            # Detector should not be called for empty input
            mock_detector.predict.assert_not_called()
            
    @patch('ai_detector.detector_app.get_detector') 
    def test_interactive_mode_keyboard_interrupt(self, mock_get_detector):
        """Test interactive mode handles keyboard interrupt gracefully"""
        mock_detector = Mock()
        mock_get_detector.return_value = mock_detector
        
        test_args = ['detector_app.py', '--interactive']
        
        with patch('sys.argv', test_args), \
             patch('builtins.input', side_effect=KeyboardInterrupt), \
             patch('builtins.print') as mock_print:
            main()
            
            # Should exit gracefully without error
            print_calls = [call.args[0] for call in mock_print.call_args_list]
            assert any('Goodbye!' in call for call in print_calls)


class TestErrorHandling:
    """Test error handling in CLI"""
    
    def test_no_text_provided_error(self):
        """Test error when no text is provided"""
        test_args = ['detector_app.py']
        mock_stdin = StringIO('')  # Empty stdin
        
        with patch('sys.argv', test_args), \
             patch('sys.stdin', mock_stdin), \
             patch('sys.stdin.isatty', return_value=False), \
             patch('sys.exit') as mock_exit, \
             patch('builtins.print') as mock_print:
            main()
            
            # Should exit with error code 1
            mock_exit.assert_called_once_with(1)
            
            # Should print error message
            print_calls = [call.args for call in mock_print.call_args_list]
            error_messages = [call[0] for call in print_calls if len(call) > 1 and call[1] == sys.stderr]
            assert any('No text provided' in msg for msg in error_messages)