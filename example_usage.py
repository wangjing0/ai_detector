#!/usr/bin/env python3
"""
Example usage of the AI Detector package with different model configurations.
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_model_configs():
    """Demonstrate different model configurations"""
    from ai_detector.model_configs import (
        list_available_configs, 
        get_model_info, 
        get_model_config, 
        recommend_config,
        get_configs_by_size
    )
    
    print("=== AI Detector Model Configurations Demo ===\n")
    
    # List all available configurations
    print("Available Model Configurations:")
    configs = list_available_configs()
    for config in configs[:5]:  # Show first 5
        info = get_model_info(config)
        print(f"  {config}:")
        print(f"    Description: {info['description']}")
        if 'speed' in info:
            print(f"    Performance: Speed={info['speed']}, Accuracy={info['accuracy']}, VRAM={info['vram']}")
    
    print(f"\n... and {len(configs) - 5} more configurations available.\n")
    
    # Show configurations by size
    print("Small Models (low VRAM):")
    small_models = get_configs_by_size('small')
    for model in small_models:
        observer, performer = get_model_config(model)
        print(f"  {model}: {observer} + {performer}")
    
    # Get recommendations
    print(f"\nRecommended for 8GB VRAM: {recommend_config(vram_gb=8.0)}")
    print(f"Recommended for speed: {recommend_config(speed_priority=True)}")
    
    # Show specific config details
    print(f"\nDetailed info for 'llama_3_1_8b':")
    config_info = get_model_info('llama_3_1_8b')
    for key, value in config_info.items():
        print(f"  {key}: {value}")

def demo_detector_usage():
    """Demonstrate basic detector usage without loading heavy models"""
    print("\n=== AI Detector Usage Demo ===\n")
    
    # Example of how to use the detector (without actually loading models)
    print("Example usage (pseudo-code, models not loaded):")
    print("""
from ai_detector import Detector, get_model_config

# Use a predefined configuration
observer, performer = get_model_config('gemma_2b')  # Lightweight option
detector = Detector(
    observer_name_or_path=observer,
    performer_name_or_path=performer,
    mode='accuracy'
)

# Analyze text
result = detector.predict("This is a sample text to analyze.")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Score: {result['score']:.3f}")
""")

def demo_cli_usage():
    """Demonstrate CLI usage examples"""
    print("\n=== CLI Usage Examples ===\n")
    
    print("Command line usage examples:")
    print("# List available model configurations")
    print("python -m ai_detector.detector_app --list-configs")
    print()
    print("# Get model recommendation")
    print("python -m ai_detector.detector_app --recommend-config --vram-gb 8")
    print()
    print("# Use a specific model configuration")
    print("python -m ai_detector.detector_app --model-config gemma_2b 'Your text here'")
    print()
    print("# Use custom models directly")
    print("python -m ai_detector.detector_app --observer-model 'model1' --performer-model 'model2' 'Text'")
    print()
    print("# Interactive mode")
    print("python -m ai_detector.detector_app --interactive --model-config llama_3_2_3b")

if __name__ == "__main__":
    try:
        demo_model_configs()
        demo_detector_usage() 
        demo_cli_usage()
        
        print("\n✅ Package structure appears to be working correctly!")
        print("\nTo run the actual detector, use:")
        print("python -m ai_detector.detector_app --help")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()