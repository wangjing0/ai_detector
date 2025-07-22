"""
AI Text Detector Package

This package provides tools for detecting AI-generated text using dual language model analysis.
"""

from .detector import Detector
from .detector_app import main as run_detector_app, predict_text, get_detector
from .model_configs import get_model_config, list_available_configs, get_model_info, recommend_config

__version__ = "0.1.0"
__author__ = "Jing Wang"
__email__ = "jingwang.physics@gmail.com"

__all__ = [
    "Detector",
    "run_detector_app", 
    "predict_text",
    "get_detector",
    "get_model_config",
    "list_available_configs", 
    "get_model_info",
    "recommend_config"
]