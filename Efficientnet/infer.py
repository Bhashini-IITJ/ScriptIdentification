import os
import argparse
import time
import csv
import json
import torch
from efficientnet_identifier import EfficientNetIdentifier

# Define the default classes
CLASSES = ["hindi", "english", "assamese", "bengali", "gujarati", "kannada", "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu"]

def get_config_image_size():
    try:
        with open('config.py', 'r') as f:
            config = json.load(f)
        return config.get("hyperparameters", {}).get("image_size", 480)
    except Exception:
        return 480

def predict(image_path, identifier):
    predicted_label = identifier.identify(image_path)
    return {"predicted_class": predicted_label.capitalize() if predicted_label else None}

def predict_batch(image_dir, identifier, time_show, output_csv="prediction.csv"):
    start_time = time.time()
    results = []
    
    # Get all valid images
    valid_extensions = (".png", ".jpg", ".jpeg")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(valid_extensions)]
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    
    if not image_paths:
        print("No images found in the directory.")
        return output_csv
        
    # Use identify_batch from EfficientNetIdentifier
    batch_size = 32
    predicted_labels = identifier.identify_batch(image_paths, batch_size=batch_size)
    
    for filename, predicted_label in zip(image_files, predicted_labels):
        results.append({
            "Filepath": filename, 
            "Language": predicted_label.capitalize() if predicted_label else "Unknown"
        })
    
    elapsed_time = time.time() - start_time

    if time_show:
        print(f"Time taken to process {len(image_files)} images: {elapsed_time:.2f} seconds")
    
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Filepath", "Language"])
        writer.writeheader()
        writer.writerows(results)
    
    return output_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification using EfficientNet model")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--image_dir", type=str, help="Path to the input image directory")
    parser.add_argument("--batch", action="store_true", help="Process images in batch mode if specified")
    parser.add_argument("--time", type=bool, nargs="?", const=True, default=False, help="Prints the time required to process a batch of images")
    parser.add_argument("--model_name", type=str, default="efficientnet_v2m_sliced", help="Model name as defined in efficientnet_identifier.py")

    args = parser.parse_args()

    # Initialize identifier once
    checkpoint_path = "models/efficientnet/efficientnetv2_m_sliced.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    identifier = EfficientNetIdentifier(
        checkpoint_path=checkpoint_path,
        classes=CLASSES,
        image_size=get_config_image_size(),
        device=device,
        model_name=args.model_name
    )

    if args.batch:
        if not args.image_dir:
            print("Error: image_dir is required when batch is set to True.")
        else:
            result = predict_batch(args.image_dir, identifier, args.time)
            print(f"Predictions saved to: {result}")
    else:
        if not args.image_path:
            print("Error: image_path is required when batch is not set.")
        else:
            result = predict(args.image_path, identifier)
            print(result)
