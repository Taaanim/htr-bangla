"""
test_model.py — Quick testing and validation for Bangla HTR model

Supports:
1. Test on a single image file
2. Test on a dataset folder
3. Generate full classification report
"""

import os
import json
import argparse
import numpy as np
import torch
import joblib
from sklearn.metrics import classification_report

from model import BestCNN
from dataset import prepare_data, IMG_SIZE
from predict import BanglaPredictor


def test_single_image(image_path: str, weights: str, as_json: bool = False):
    """Test model on a single image."""
    predictor = BanglaPredictor(weights_path=weights)
    result = predictor.predict_file(image_path)

    if as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"\n{'='*50}")
        print(f"BANGLA HTR — Single Image Test")
        print(f"{'='*50}")
        print(f"Image:      {image_path}")
        print(f"Prediction: {result.get('prediction', '?')}")
        print(f"Confidence: {result.get('confidence', 0)}%")
        if "top5" in result:
            print(f"\nTop 5 Predictions:")
            for i, item in enumerate(result["top5"], 1):
                bar = "█" * int(item["confidence"] / 5) + "░" * (20 - int(item["confidence"] / 5))
                print(f"  {i}. {item['label']}  {bar} {item['confidence']}%")
        print(f"{'='*50}")


def test_validation_set(weights: str):
    """Run full validation and generate classification report."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_loader, val_loader, le, num_classes = prepare_data()

    model = BestCNN(num_classes).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    accuracy = 100 * (all_preds == all_true).mean()

    print(f"\n{'='*60}")
    print(f"VALIDATION REPORT — Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")

    class_names = list(le.classes_)
    print(classification_report(all_true, all_preds, target_names=class_names, digits=3))

    # Per-class accuracy breakdown
    per_class = []
    for i, cls in enumerate(class_names):
        mask = all_true == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == i).mean() * 100
            per_class.append((cls, acc, mask.sum()))

    per_class.sort(key=lambda x: x[1])

    below_90 = sum(1 for _, a, _ in per_class if a < 90)
    between = sum(1 for _, a, _ in per_class if 90 <= a < 95)
    above_95 = sum(1 for _, a, _ in per_class if a >= 95)
    perfect = sum(1 for _, a, _ in per_class if a == 100)

    print(f"\nPer-Class Summary:")
    print(f"  Below 90%:  {below_90} classes")
    print(f"  90-95%:     {between} classes")
    print(f"  Above 95%:  {above_95} classes")
    print(f"  Perfect:    {perfect} classes")

    if below_90 > 0:
        print(f"\nWeakest classes:")
        for cls, acc, count in per_class[:10]:
            print(f"  {cls}: {acc:.1f}% ({count} samples)")


def test_folder(folder_path: str, weights: str):
    """Test on all images in a folder."""
    predictor = BanglaPredictor(weights_path=weights)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    results = []

    for fname in sorted(os.listdir(folder_path)):
        if os.path.splitext(fname)[1].lower() in image_exts:
            fpath = os.path.join(folder_path, fname)
            result = predictor.predict_file(fpath)
            result["file"] = fname
            results.append(result)
            print(f"  {fname}: {result.get('prediction', '?')} ({result.get('confidence', 0)}%)")

    print(f"\nProcessed {len(results)} images")
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Bangla HTR Model")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--folder", type=str, help="Folder of images to test")
    parser.add_argument("--report", action="store_true", help="Full validation report")
    parser.add_argument("--weights", type=str, default="checkpoints/best_cnn_model_weights.pth")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.image:
        test_single_image(args.image, args.weights, args.json)
    elif args.folder:
        test_folder(args.folder, args.weights)
    elif args.report:
        test_validation_set(args.weights)
    else:
        print("Usage:")
        print("  python3 test_model.py --image <path>     # Test single image")
        print("  python3 test_model.py --folder <path>    # Test folder of images")
        print("  python3 test_model.py --report           # Full validation report")


if __name__ == "__main__":
    main()
