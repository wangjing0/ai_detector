#!/usr/bin/env python3

import argparse
import sys
import os
from .detector import Detector

try:
    from .model_configs import get_model_config, list_available_configs, get_model_info, recommend_config
except ImportError:
    # Fallback if model_configs not available
    def get_model_config(config_name):
        raise ValueError(f"Model configs not available. Use --observer-model and --performer-model directly.")
    def list_available_configs():
        return []
    def get_model_info(config_name):
        return {}
    def recommend_config(**kwargs):
        return "llama_3_1_8b"

# Global detector instance
_detector = None

def get_detector(observer_model='unsloth/Meta-Llama-3.1-8B-bnb-4bit',
                performer_model='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
                mode='accuracy'
) -> Detector:
    """Get or create detector instance (singleton pattern)"""
    global _detector
    if _detector is None:
        print("Loading models... This may take a moment.")
        _detector = Detector(
            observer_name_or_path=observer_model,
            performer_name_or_path=performer_model,
            mode=mode
        )
        print("Models loaded successfully!")
    return _detector

def predict_text(text, display_highlights=False):
    """Predict if text is AI-generated using cached detector"""
    detector = get_detector()
    return detector.predict(text, display_text=display_highlights)

def main():
    parser = argparse.ArgumentParser(description='AI Text Detection App')
    parser.add_argument('text', nargs='?', help='Text to analyze (or use stdin)')
    parser.add_argument('--mode', choices=['accuracy', 'low-fpr'], default='accuracy',
                        help='Detection mode (default: accuracy)')
    parser.add_argument('--display-highlights', action='store_true',
                        help='Display highlighted suspicious words (works with both single and interactive mode)')
    parser.add_argument('--observer-model', default='unsloth/Meta-Llama-3.1-8B-bnb-4bit',
                        help='Observer model path or HuggingFace model ID')
    parser.add_argument('--performer-model', default='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
                        help='Performer model path or HuggingFace model ID')
    parser.add_argument('--model-config', 
                        help=f'Use predefined model configuration. Available: {", ".join(list_available_configs()) if list_available_configs() else "none"}')
    parser.add_argument('--list-configs', action='store_true',
                        help='List available model configurations and exit')
    parser.add_argument('--recommend-config', action='store_true',
                        help='Get model configuration recommendation and exit')
    parser.add_argument('--vram-gb', type=float,
                        help='Available VRAM in GB (for model recommendation)')
    parser.add_argument('--speed-priority', action='store_true',
                        help='Prioritize inference speed in model recommendation')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode for multiple predictions (can be combined with --display-highlights)')

    args = parser.parse_args()
    
    # Handle utility commands first
    if args.list_configs:
        print("Available model configurations:")
        for config_name in list_available_configs():
            try:
                info = get_model_info(config_name)
                print(f"  {config_name}: {info.get('description', 'No description available')}")
                if 'speed' in info:
                    print(f"    Speed: {info['speed']}, Accuracy: {info['accuracy']}, VRAM: {info['vram']}")
            except Exception as e:
                print(f"  {config_name}: Error getting info - {e}")
        return
    
    if args.recommend_config:
        try:
            recommended = recommend_config(vram_gb=args.vram_gb, speed_priority=args.speed_priority)
            info = get_model_info(recommended)
            print(f"Recommended configuration: {recommended}")
            print(f"Description: {info.get('description', 'No description available')}")
            if 'speed' in info:
                print(f"Performance: Speed={info['speed']}, Accuracy={info['accuracy']}, VRAM={info['vram']}")
            observer, performer = get_model_config(recommended)
            print(f"Observer model: {observer}")
            print(f"Performer model: {performer}")
        except Exception as e:
            print(f"Error getting recommendation: {e}")
        return
    
    # Determine which models to use
    observer_model = args.observer_model
    performer_model = args.performer_model
    
    if args.model_config:
        try:
            observer_model, performer_model = get_model_config(args.model_config)
            print(f"Using model config '{args.model_config}':")
            print(f"  Observer: {observer_model}")
            print(f"  Performer: {performer_model}")
        except Exception as e:
            print(f"Error loading model config: {e}")
            print("Falling back to default models.")

    # Initialize detector once
    detector = get_detector(observer_model, performer_model, args.mode)
    
    if args.interactive:
        print("\n=== Interactive Mode ===")
        if args.display_highlights:
            print("Highlighting enabled - suspicious words will be highlighted for AI-generated text")
        print("Enter text to analyze (type 'quit' to exit):")
        while True:
            try:
                text_input = input("> ").strip()
                if text_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not text_input:
                    continue
                    
                result = detector.predict(text_input, display_text=args.display_highlights)
                print(f"Prediction: {result['prediction']}")
                print(f"Score: {result['score']:.4f}")
                print(f"Confidence: {result['confidence']:.4f}")
                
                if args.display_highlights and result['prediction'] == 'AI-generated':
                    print("(Highlighted suspicious words shown above)")
                elif args.display_highlights and result['prediction'] == 'Human-generated':
                    print("(No highlighting for human-generated text)")
                
                print("-" * 50)
            except KeyboardInterrupt:
                break
        print("\nGoodbye!")
        return

    # Single prediction mode
    if args.text:
        text_input = args.text
    else:
        if sys.stdin.isatty():
            print("Enter text to analyze (Ctrl+D to finish):")
        text_input = sys.stdin.read().strip()

    if not text_input:
        print("No text provided.", file=sys.stderr)
        sys.exit(1)

    result = detector.predict(text_input, display_text=args.display_highlights)
    
    print(f"\nPrediction: {result['prediction']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    if args.display_highlights and result['prediction'] == 'AI-generated':
        print("\nHighlighted suspicious words are shown above.")

if __name__ == "__main__":
    main()