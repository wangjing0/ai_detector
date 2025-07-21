#!/usr/bin/env python3

import argparse
import sys
from src.detector import Detector

# Global detector instance
_detector = None

def get_detector(observer_model='unsloth/Meta-Llama-3.1-8B-bnb-4bit',
                performer_model='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
                mode='accuracy'):
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
                        help='Display highlighted suspicious words')
    parser.add_argument('--observer-model', default='unsloth/Meta-Llama-3.1-8B-bnb-4bit',
                        help='Observer model path')
    parser.add_argument('--performer-model', default='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
                        help='Performer model path')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode for multiple predictions')

    args = parser.parse_args()

    # Initialize detector once
    detector = get_detector(args.observer_model, args.performer_model, args.mode)
    
    if args.interactive:
        print("\n=== Interactive Mode ===")
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