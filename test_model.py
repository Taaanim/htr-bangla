"""
test_model.py - Quick Interactive Model Tester for Bangla HTR

Allows testing the trained model by:
1. Typing any Bangla text string (renders synthetic handwriting image and tests recognition).
2. Providing a handwriting image file path.
"""

import os
import argparse
from predict import BanglaHTRPredictor
from dataset import BanglaSyntheticTextGenerator


def main():
    parser = argparse.ArgumentParser(description="Test Trained Bangla HTR Model")
    parser.add_argument("--image", type=str, help="Path to input handwriting image file")
    parser.add_argument("--text", type=str, help="Bangla text string to render and test model recognition")
    parser.add_argument("--model-path", type=str, default="checkpoints/final_htr_model.pth", help="Model checkpoint path")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("BANGLA HTR MODEL TESTER")
    print("=" * 50)

    # Initialize Predictor
    predictor = BanglaHTRPredictor(model_path=args.model_path)

    # 1. Test by Image Path
    if args.image and os.path.exists(args.image):
        print(f"\nTesting Image File: {args.image}")
        result = predictor.predict(args.image)
        print("-" * 40)
        print(f"Raw Model Output      : '{result['raw_prediction']}'")
        print(f"Corrected Bangla Text : '{result['corrected_prediction']}'")
        print(f"Confidence Score      : {result['confidence_score']}%")
        print(f"In Bangla Dictionary  : {result['is_in_dictionary']}")
        print("-" * 40)

    # 2. Test by Text String (Text -> Render -> Model Recognition)
    elif args.text:
        print(f"\nTesting Text String: '{args.text}'")
        generator = BanglaSyntheticTextGenerator()
        rendered_img = generator.render_text(args.text)

        result = predictor.predict(rendered_img)
        print("-" * 40)
        print(f"Target Input Text     : '{args.text}'")
        print(f"Raw Model Output      : '{result['raw_prediction']}'")
        print(f"Corrected Bangla Text : '{result['corrected_prediction']}'")
        print(f"Confidence Score      : {result['confidence_score']}%")
        print(f"In Bangla Dictionary  : {result['is_in_dictionary']}")
        print("-" * 40)

    # 3. Default Interactive Demo
    else:
        demo_texts = ["বাংলাদেশ", "শিক্ষা", "বিজ্ঞান", "নতুন"]
        print("\nNo specific image or text provided. Running demonstration on test words:")
        generator = BanglaSyntheticTextGenerator()

        for word in demo_texts:
            rendered_img = generator.render_text(word)
            result = predictor.predict(rendered_img)
            print("-" * 40)
            print(f"Ground Truth Target   : '{word}'")
            print(f"Raw Model Output      : '{result['raw_prediction']}'")
            print(f"Corrected Bangla Text : '{result['corrected_prediction']}'")
            print(f"Confidence Score      : {result['confidence_score']}%")
            print(f"In Bangla Dictionary  : {result['is_in_dictionary']}")


if __name__ == "__main__":
    main()
